# from src.data.preprocess import preprocess_data
# from src.models.train import train
from src.data.load_data import load_hibou_audio_files
#
# if __name__ == "__main__":
#     preprocess_data("data/raw/data.csv", "data/processed/data.csv")
#     train("configs/train_config.yaml")'

if __name__ == "__main__":
    load_hibou_audio_files("data/raw/hibou_dataset")