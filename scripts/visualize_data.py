import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import cv2

from src.utils import draw_boxes_from_labels, CLASS_NAMES


def plot_random_images(directory: str, num_images: int = 3):
    for class_name in os.listdir(directory):
        images_path = os.path.join(directory, class_name, 'images')
        labels_path = os.path.join(directory, class_name, 'labels')
        images = os.listdir(images_path)
        selected_images = random.sample(images, num_images)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Displaying Images from {class_name}", fontsize=16, fontweight='bold')

        for ax, image in zip(axes, selected_images):
            image_path = os.path.join(images_path, image)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label_file = image.replace('.jpg', '.txt')
            label_file_path = os.path.join(labels_path, label_file)
            with open(label_file_path, 'r') as file:
                labels = file.readlines()
            img = draw_boxes_from_labels(image=img, labels=labels, class_names=CLASS_NAMES)
            ax.imshow(img)
            ax.set_title(f"{class_name} - {os.path.basename(image)}", fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.savefig(f"random_images_{class_name}.png")



if __name__ == '__main__':
    project_dir = Path(__file__).parent.parent

    # Paths to data directories
    train_path = os.path.join(project_dir, "dataset/Train")
    val_path = os.path.join(project_dir, "dataset/Val")

    plot_random_images(directory=train_path)