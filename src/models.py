from typing import Union, Tuple, List, Any
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, Lambda, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU = 0.1
ANCHORS = np.array([
    (10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
    (59, 119), (116, 90), (156, 198), (373, 326)],
    np.float32) / 416
ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])



def darknet_convolution_block(
        inputs: tf.Tensor,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        trainable: bool,
        strides: Union[int, Tuple[int, int]] = 1,
        use_batch_norm: bool = True
) -> tf.Tensor:
    """
    Performs 2D Convolution operation with standard set of parameters.

    :param inputs: Input tensor
    :param filters: The dimension of the output space
    :param kernel_size: Integer specifying the size of the convolution window.
    :param strides: Integer specifying the stride length of the convolution
    :param trainable: Boolean value to indicate if the layer is trainable.
    :param use_batch_norm: Boolean value to indicate if batch normalization must be performed
    :return: Output tensor after convolution operation
    """
    inputs = Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
    )(inputs)
    if use_batch_norm:
        inputs = BatchNormalization(
            axis=3,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            scale=True,
            trainable=trainable
        )(inputs)
        inputs = tf.nn.leaky_relu(inputs, alpha=LEAKY_RELU)

    return inputs


def darknet53_residual_block(
        inputs: tf.Tensor,
        filters: int,
        trainable: bool,
        strides: Union[int, Tuple[int, int]] = 1,
) -> tf.Tensor:
    """
    Creates a residual block for Darknet.

    :param inputs: Input tensor
    :param filters: The dimension of the output space
    :param trainable: Boolean value to indicate if the layer is trainable.
    :param strides: Integer specifying the stride length of the convolution
    :return: Output tensor after darknet residual block
    """
    shortcut = inputs

    inputs = darknet_convolution_block(
        inputs=inputs, filters=filters, kernel_size=1, strides=strides, trainable=trainable
    )
    inputs = darknet_convolution_block(
        inputs, filters=2 * filters, kernel_size=3, strides=strides, trainable=trainable
    )

    inputs += shortcut

    return inputs


def darknet53(
        name: str, trainable: bool
) -> Model:
    """
    Creates Darknet53 model for feature extraction.
    :param name: A string
    :param trainable: Boolean value to indicate if the layer is trainable.
    :return: 3 Output tensors
    """

    inputs = Input([416, 416, 3])

    out = darknet_convolution_block(inputs, filters=32, kernel_size=3, trainable=trainable)
    out = darknet_convolution_block(out, filters=64, kernel_size=3, strides=2, trainable=trainable)
    out = darknet53_residual_block(out, filters=32, trainable=trainable)

    out = darknet_convolution_block(out, filters=128, kernel_size=3, strides=2, trainable=trainable)
    for _ in range(2):
        out = darknet53_residual_block(out, filters=64, trainable=trainable)

    out = darknet_convolution_block(out, filters=256, kernel_size=3, strides=2, trainable=trainable)
    for _ in range(8):
        out = darknet53_residual_block(out, filters=128, trainable=trainable)

    route1 = out

    out = darknet_convolution_block(out, filters=512, kernel_size=3, strides=2, trainable=trainable)
    for _ in range(8):
        out = darknet53_residual_block(out, filters=256, trainable=trainable)

    route2 = out

    out = darknet_convolution_block(out, filters=1024, kernel_size=3, strides=2, trainable=trainable)
    for _ in range(4):
        out = darknet53_residual_block(out, filters=512, trainable=trainable)

    darknet_model = tf.keras.Model(inputs, (route1, route2, out), name=name)
    return darknet_model


def yolo_convolution_block(
        name: str,
        input_tensor: tf.Tensor,
        filters: int,
        trainable: bool,
) -> Model:
    """
    Creates convolution operations layer used after Darknet.

    :param name: A string
    :param input_tensor: Input tensor
    :param filters: The dimension of the output space
    :param trainable: Boolean value to indicate if the layer is trainable.
    :return: 2 Output tensors
    """
    inputs = Input(input_tensor.shape[1:])
    out = darknet_convolution_block(inputs, filters=filters, kernel_size=1, trainable=trainable)
    out = darknet_convolution_block(out, filters=2 * filters, kernel_size=3, trainable=trainable)
    out = darknet_convolution_block(out, filters=filters, kernel_size=1, trainable=trainable)
    out = darknet_convolution_block(out, filters=2 * filters, kernel_size=3, trainable=trainable)
    out = darknet_convolution_block(out, filters=filters, kernel_size=1, trainable=trainable)

    route = out

    out = darknet_convolution_block(out, filters=2 * filters, kernel_size=3, trainable=trainable)
    yolo_conv_block = tf.keras.Model(inputs, (route, out), name=name)
    return yolo_conv_block


def yolo_output_layer(
        name: str,
        input_tensor: tf.Tensor,
        filters: int,
        trainable: bool,
        n_classes: int,
        anchors: int,
) -> Model:
    """
    Creates Yolo final detection layer.
    Detects boxes with respect to anchors.

    :param name: A string
    :param input_tensor: Input tensor
    :param filters: The dimension of the output space
    :param trainable: Boolean value to indicate if the layer is trainable.
    :param n_classes: Number of classes
    :param anchors: A list of anchor sizes.
    :return: Output tensor
    """
    inputs = Input(input_tensor.shape[1:])
    out = darknet_convolution_block(inputs, filters=filters, kernel_size=3, trainable=trainable)
    out = darknet_convolution_block(
        out, filters=anchors * (n_classes + 5), kernel_size=1, use_batch_norm=False, trainable=trainable
    )
    out = Lambda(
        lambda out: tf.reshape(
            out, (-1, tf.shape(out)[1], tf.shape(out)[2], anchors, n_classes + 5)
        )
    )(out)
    return tf.keras.Model(inputs, out, name=name)


def build_boxes(
        inputs: tf.Tensor,
        anchors: np.ndarray,
        n_classes: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes top left and bottom right points of the boxes.
    :param anchors: A list of anchors
    :param inputs: Input tensor
    :param n_classes: Number of classes
    :return: Output tensor
    """

    # inputs: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(inputs)[1:3][::-1]
    grid_y, grid_x = tf.shape(inputs)[1], tf.shape(inputs)[2]
    stack = inspect.stack()
    box_xy, box_wh, objectness, class_probs = tf.split(inputs, (2, 2, 1, n_classes), axis=-1)
    box_xy = tf.sigmoid(box_xy)

    objectness = tf.math.sigmoid(objectness)
    class_probs = tf.nn.softmax(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    grid = tf.meshgrid(tf.range(grid_x), tf.range(grid_y))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.math.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def non_max_suppression(
        inputs: tf.Tensor,
        max_output_size: int,
        iou_threshold: float,
        confidence_threshold: float,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Performs non-max suppression separately for each class.

    :param inputs: Input tensor
    :param max_output_size: Max number of boxes to be selected for each class.
    :param iou_threshold: Threshold for the IOU.
    :param confidence_threshold: Threshold for the confidence score.
    :return: A list containing class-to-boxes dictionaries
            for each sample in the batch.
    """
    b, c, t = [], [], []
    for o in inputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    scores = confidence * class_probs
    nms_boxes, nms_scores, nms_classes, nms_valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores,
            (tf.shape(scores)[0], -1, tf.shape(scores)[-1])
        ),
        max_output_size_per_class=max_output_size,
        max_total_size=100,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return nms_boxes, nms_scores, nms_classes, nms_valid_detections


def broadcast_iou(box_1: tf.Tensor, box_2: tf.Tensor) -> Any:
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def loss_function(
        anchors: np.ndarray,
        n_classes: int = 4,
        ignore_thresh: float = 0.5
) -> Any:
    """
    Loss function to train the model

    :param anchors: A list of anchors
    :param n_classes: Number of classes
    :param ignore_thresh: Ignore threshold
    :return: yolo loss function
    """

    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = build_boxes(
            inputs=y_pred,
            anchors=anchors,
            n_classes=n_classes
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]
        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]
        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                  tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(
            pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
                   (1 - obj_mask) * ignore_mask * obj_loss
        # Could also use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)
        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss




class YoloV3:
    """
    Yolo v3 model class.
    """

    def __init__(
            self,
            n_classes: int,
            img_size: Tuple[int, int],
            max_output_size: int = 10,
            iou_threshold: float = 0.5,
            confidence_threshold: float = 0.5,
            trainable: bool = False):
        """
        Creates the model.

        :param n_classes: Number of classes
        :param img_size: The input size of the model.
        :param max_output_size: Max number of boxes to be selected for each class.
        :param iou_threshold: Threshold for the IOU.
        :param confidence_threshold: Threshold for the confidence score.
        :return: A list containing class-to-boxes dictionaries
                for each sample in the batch.
        """
        self.n_classes = n_classes
        self.img_size = img_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.trainable = trainable

    def __call__(self):
        """
        Add operations to detect boxes for a batch of input images.

        :param trainable: A boolean, whether to use in training or inference mode.
        :return:
        """
        inputs = Input([self.img_size[0], self.img_size[1], 3])

        route1, route2, out = darknet53(
            name="Darknet_Block",
            trainable=self.trainable,
        )(inputs)
        route, out = yolo_convolution_block(
            name="Yolo_Conv_0",
            input_tensor=out,
            filters=512,
            trainable=self.trainable
        )(out)
        detect1 = yolo_output_layer(
            name="Yolo_Output_0",
            input_tensor=out,
            filters=512,
            trainable=self.trainable,
            n_classes=self.n_classes,
            anchors=len(ANCHOR_MASKS[0]),
        )(out)

        out = darknet_convolution_block(out, filters=256, kernel_size=1, trainable=self.trainable)
        out = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")(out)
        out = Concatenate(axis=3)([out, route2])

        route, out = yolo_convolution_block(
            name="Yolo_Conv_1",
            input_tensor=out,
            filters=256,
            trainable=self.trainable
        )(out)
        detect2 = yolo_output_layer(
            name="Yolo_Output_1",
            input_tensor=out,
            filters=256,
            trainable=self.trainable,
            n_classes=self.n_classes,
            anchors=len(ANCHOR_MASKS[1]),
        )(out)

        out = darknet_convolution_block(out, filters=128, kernel_size=1, trainable=self.trainable)
        out = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")(out)
        out = Concatenate(axis=3)([out, route1])

        route, out = yolo_convolution_block(
            name="Yolo_Conv_2",
            input_tensor=out,
            filters=128,
            trainable=self.trainable
        )(out)
        detect3 = yolo_output_layer(
            name="Yolo_Output_2",
            input_tensor=out,
            filters=128,
            trainable=self.trainable,
            n_classes=self.n_classes,
            anchors=len(ANCHOR_MASKS[2]),
        )(out)

        if self.trainable:
            return Model(inputs, (detect1, detect2, detect3), name='Yolov3')

        boxes_0 = Lambda(lambda x: build_boxes(x, ANCHORS[ANCHOR_MASKS[0]], self.n_classes), name='yolo_boxes_0')(detect1)
        boxes_1 = Lambda(lambda x: build_boxes(x, ANCHORS[ANCHOR_MASKS[1]], self.n_classes), name='yolo_boxes_1')(detect2)
        boxes_2 = Lambda(lambda x: build_boxes(x, ANCHORS[ANCHOR_MASKS[2]], self.n_classes), name='yolo_boxes_2')(detect3)

        outputs = Lambda(
            lambda x: non_max_suppression(
                x, max_output_size=100, iou_threshold=0.5, confidence_threshold=0.5
            ),
            name='yolo_nms'
        )((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return Model(inputs, outputs, name='Yolov3')








