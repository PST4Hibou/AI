from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
import torch
import torchcodec
from datasets import ClassLabel, Audio
from collections import Counter

labels = ['other', 'drone']

old_dataset = load_dataset("Usernameeeeee/df_462700_2")
old_dataset = old_dataset.remove_columns("sampling_rate")

print(old_dataset)

added_ds = load_dataset("audiofolder", data_dir="data/raw/new/drone_output")
added_ds = added_ds.cast_column("label", ClassLabel(names=labels))
# added_ds = added_ds.cast_column("audio", Audio(sampling_rate=16000))

def process_new_dataset(examples):
    all_audios = []
    all_labels = []

    for audio, label in zip(examples["audio"], examples["label"]):
        audio_array = audio["array"]
        audio_array = torch.tensor(audio_array).float()

        all_audios.append(audio_array.numpy())
        all_labels.append(label)

    return {
        "audio": all_audios,
        "label": all_labels,
    }


added_ds = added_ds["train"].map(
    process_new_dataset,
    batched=True,
    num_proc=12,
    batch_size=16,
    remove_columns=added_ds["train"].column_names,
)
new_ds = concatenate_datasets([old_dataset["train"], added_ds])


new_ds.push_to_hub("Usernameeeeee/df_462700_3")