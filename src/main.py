import os
from pathlib import Path

from src.data import load_data, preprocess_image
from src.models import YoloV3

INPUT_SHAPE = (416, 416)

if __name__ == '__main__':

    project_dir = Path(__file__).parent.parent

    # Paths to data directories
    train_path = os.path.join(project_dir, "dataset/Train")
    val_path = os.path.join(project_dir, "dataset/Val")

    # Load training and validation data
    #train_images, train_labels = load_data(train_path)
    #val_images, val_labels = load_data(val_path)

    #print(f"The size of training dataset: {len(train_images)}")
    #print(f"The size of validation dataset: {len(val_images)}")


    #train_images = preprocess_image(train_images)
    #val_images = preprocess_image(val_images)

    yolo = YoloV3(
        n_classes=4,
        img_size=INPUT_SHAPE,
        max_output_size=10,
        iou_threshold=0.5,
        confidence_threshold=0.5
    )()
    yolo.summary()

