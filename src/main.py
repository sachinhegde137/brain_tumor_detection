import h5py
import os
from pathlib import Path
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import DataVisualizer
from src.models import YoloV3, ANCHORS, ANCHOR_MASKS, loss_function
from src.data import transform_targets, tf_transform_imgs_labels

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

INPUT_SHAPE = (416, 416)
n_classes = 4
LEARNING_RATE = 1e-3


def create_dataset(dir_path: str) -> tf.data.Dataset:
    """
    Creates Tensorflow dataset object
    :param dir_path: Path to train or validation directory
    :return: Tensorflow dataset object
    """
    # Get the list of images and labels
    images_list = []
    label_file_list = []
    for class_name in os.listdir(dir_path):
        images_path = os.path.join(dir_path, class_name, 'images')
        labels_path = os.path.join(dir_path, class_name, 'labels')
        for image in os.listdir(images_path):
            image_path = os.path.join(images_path, image)
            label_file = image.replace('.jpg', '.txt')
            label_file_path = os.path.join(labels_path, label_file)
            if os.path.exists(label_file_path):
                images_list.append(image_path)
                label_file_list.append(label_file_path)

    # Create TensorFlow Dataset
    img_dataset = tf.data.Dataset.from_tensor_slices(images_list)
    labels_dataset = tf.data.Dataset.from_tensor_slices(label_file_list)

    dataset = tf.data.Dataset.zip((img_dataset, labels_dataset))

    dataset = dataset.map(lambda x, y: tf_transform_imgs_labels(x, y, INPUT_SHAPE))

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object detection using YoloV3 model.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the train dataset.")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for the training.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for the training.")
    args = parser.parse_args()

    batch_size = args.batch_size
    n_epochs = args.epochs

    project_dir = Path(__file__).parent.parent

    # Paths to data directories
    train_path = os.path.join(project_dir, args.train_path)
    val_path = os.path.join(project_dir, args.val_path)

    train_dataset = create_dataset(train_path)
    val_dataset = create_dataset(val_path)

    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.map(
        lambda x, y: (x, transform_targets(y, INPUT_SHAPE, ANCHORS, ANCHOR_MASKS, n_classes)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size=batch_size)
    val_dataset = val_dataset.map(
        lambda x, y: (x, transform_targets(y, INPUT_SHAPE, ANCHORS, ANCHOR_MASKS, n_classes)))

    train_dataset = train_dataset.skip(250)
    model = YoloV3(
        n_classes=4,
        img_size=INPUT_SHAPE,
        max_output_size=10,
        iou_threshold=0.5,
        confidence_threshold=0.5,
        trainable=True
    )()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = [loss_function(ANCHORS[mask], n_classes=n_classes) for mask in ANCHOR_MASKS]
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_{epoch}.h5', verbose=1, save_weights_only=False),
        CSVLogger('training.log'),
        DataVisualizer(val_dataset, result_dir='train_results')
    ]

    history = model.fit(train_dataset, epochs=n_epochs, callbacks=callbacks, validation_data=val_dataset, verbose=True)
    model.save(f"YoloV3_{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}.h5")



