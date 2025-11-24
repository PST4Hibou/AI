import os
import cv2
import argparse
import torch
import numpy as np
import traceback
from sam3.model_builder import build_sam3_video_predictor

# -------- argparse ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="sam2.1", help="모델 버전 (e.g., sam2, sam2.1)")
args = parser.parse_args()
# ---------------------------

# Check CUDA
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

# Optional: enable TF32 only when CUDA is available and device supports it
if use_cuda:
    try:
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# Build predictor (ensure it supports CPU fallback if no CUDA)
predictor = build_sam3_video_predictor()

# globals
drawing = False
ix = iy = fx = fy = -1
bbox = None
enter_pressed = False

def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, bbox, enter_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        bbox = (ix, iy, fx, fy)
        enter_pressed = True

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera")

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Camera", draw_rectangle)

if_init = False

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]  # standard: (h, w)

        if not enter_pressed:
            # draw selection UI
            disp = frame.copy()
            if drawing and ix >= 0 and iy >= 0:
                cv2.rectangle(disp, (ix, iy), (fx, fy), (255, 0, 0), 2)
            cv2.putText(disp, "Select an object by drawing a box (press q to quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            continue

        # enter_pressed is True -> initialize once then track
        if not if_init:
            if_init = True
            # move predictor initialization inside autocast if using bfloat16 on CUDA
            if use_cuda:
                # autocast only during inference calls
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictor.load_first_frame(frame)
            else:
                predictor.load_first_frame(frame)

            ann_frame_idx = 0
            ann_obj_id = 1
            # Draw original bbox for user feedback
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            bbox_np = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)

            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox_np
                    )
            else:
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox_np
                )
        else:
            # tracking step
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_obj_ids, out_mask_logits = predictor.track(frame)
            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)

            # assemble mask
            # assume out_mask_logits is a list/iterable of tensors (logits) on the same device
            all_mask = np.zeros((height, width), dtype=np.uint8)

            for i in range(len(out_obj_ids)):
                mask_logits = out_mask_logits[i]  # tensor
                # produce boolean mask on CPU without forcing needless cuda moves
                # If mask_logits is on CUDA, move to CPU once
                if mask_logits.device.type == "cuda":
                    mask_cpu = (mask_logits > 0.0).squeeze(0).to("cpu")
                else:
                    mask_cpu = (mask_logits > 0.0).squeeze(0)

                # convert to numpy
                mask_np = mask_cpu.byte().numpy()  # shape (H, W)
                # combine (here we overwrite with last; modify if multiple objects)
                all_mask = (mask_np * 255).astype(np.uint8)

            # colorize mask and overlay
            mask_rgb = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 1.0, mask_rgb, 0.5, 0)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception:
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    if use_cuda:
        torch.cuda.empty_cache()