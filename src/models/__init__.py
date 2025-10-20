import src.models.AudioCNN2d as AudioCNN2d
import src.models.AudioCNN2d_improved as AudioCNN2d_improved


models = {"audiocnn2d": AudioCNN2d, "audiocnn2d_improved": AudioCNN2d_improved}


def select_model(model_name: str):
    if model_name.lower() in models:
        return models[model_name.lower()]
    raise KeyError(f"Unknown model name: {model_name}")
