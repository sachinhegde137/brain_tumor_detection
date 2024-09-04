import os
import shutil
from typing import List, Tuple, Any
import cv2
import uuid
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Lambda
import numpy as np
from src.models import ANCHOR_MASKS,ANCHORS, build_boxes, non_max_suppression

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumour", "Pituitary"]


def draw_outputs(img, outputs, class_names=None):
    boxes, objectness, classes = outputs
    #boxes, objectness, classes = boxes[0], objectness[0], classes[0]
    wh = np.flip(img.shape[0:2])
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_wh = np.amin(wh)
    if min_wh <= 100:
        font_size = 0.5
    else:
        font_size = 1
    for i in range(classes.shape[0]):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 1)
        img = cv2.putText(img, '{}'.format(int(classes[i])), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size,
                          (0, 0, 255), 1)
    return img



def draw_boxes_from_labels(image, labels, class_names):
    height, width, _ = image.shape
    for label in labels:
        # Split the line into its components
        parts = label.strip().split()
        class_index = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])

        # Convert normalized coordinates to pixel values
        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        bbox_width_pixel = int(bbox_width * width)
        bbox_height_pixel = int(bbox_height * height)

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center_pixel - bbox_width_pixel / 2)
        y1 = int(y_center_pixel - bbox_height_pixel / 2)
        x2 = int(x_center_pixel + bbox_width_pixel / 2)
        y2 = int(y_center_pixel + bbox_height_pixel / 2)

        # Draw the bounding box on the image
        cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        cv2.putText(
            img=image, text=CLASS_NAMES[class_index], org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)

    return image


def make_eval_model_from_trained_model(model, anchors, masks, num_classes=4, tiny=True):
    output_0, output_1, output_2 = model.outputs

    boxes_0 = Lambda(lambda x: build_boxes(x, anchors[masks[0]], num_classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: build_boxes(x, anchors[masks[1]], num_classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: build_boxes(x, anchors[masks[2]], num_classes), name='yolo_boxes_2')(output_2)
    outputs = Lambda(lambda x: non_max_suppression(x, max_output_size=100, iou_threshold=0.5, confidence_threshold=0.5),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    model = tf.keras.Model(model.inputs, outputs, name='yolov3')

    return model


class DataVisualizer(Callback):
    def __init__(self, dataset, result_dir='train_results', num_batches=8):
        self.result_dir = result_dir
        self.dataset = dataset
        self.num_batches = num_batches
        super(DataVisualizer, self).__init__()

    def on_train_begin(self, logs=None):
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir, ignore_errors=True)
        else:
            os.makedirs(self.result_dir)

    def on_epoch_end(self, epoch, logs=None):
        anchors = ANCHORS
        masks = ANCHOR_MASKS
        model = make_eval_model_from_trained_model(self.model, anchors, masks)

        epoch_dir = os.path.join(self.result_dir, str(epoch))
        os.makedirs(epoch_dir)
        for batch, (images, labels) in enumerate(self.dataset):
            images = images.numpy()
            for i in range(images.shape[0]):
                boxes, scores, classes, valid_detections = model.predict(images[i:i + 1, ...])
                img_for_this = (images[i, ...] * 255).astype(np.uint8)

                boxes_for_this, scores_for_this, classes_for_this = boxes[0, ...], scores[0, ...], classes[0, ...]

                img_for_this = draw_outputs(img_for_this, (boxes_for_this, scores_for_this, classes_for_this))
                cv2.imwrite(os.path.join(epoch_dir, '{0}.jpg'.format(uuid.uuid4())), img_for_this)
            if batch == self.num_batches:
                break



