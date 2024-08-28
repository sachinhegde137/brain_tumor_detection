from typing import Union, Tuple, List, Any
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, Lambda, Concatenate
from tensorflow.keras.regularizers import l2

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU = 0.1
ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]


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
    print(f"(DARKNET)The shape of the output after layer 1: {out.shape}")

    out = darknet_convolution_block(out, filters=64, kernel_size=3, strides=2, trainable=trainable)
    print(f"(DARKNET)The shape of the output after layer 2: {out.shape}")

    out = darknet53_residual_block(out, filters=32, trainable=trainable)
    print(f"(DARKNET)The shape of the output after residual layer 1: {out.shape}")

    out = darknet_convolution_block(out, filters=128, kernel_size=3, strides=2, trainable=trainable)
    print(f"(DARKNET)The shape of the output after layer 3: {out.shape}")

    for _ in range(2):
        out = darknet53_residual_block(out, filters=64, trainable=trainable)

    print(f"(DARKNET)The shape of the output after residual layer 2: {out.shape}")

    out = darknet_convolution_block(out, filters=256, kernel_size=3, strides=2, trainable=trainable)
    print(f"(DARKNET)The shape of the output after layer 4: {out.shape}")

    for _ in range(8):
        out = darknet53_residual_block(out, filters=128, trainable=trainable)

    print(f"(DARKNET)The shape of the output after residual layer 3 (route1): {out.shape}")

    route1 = out

    out = darknet_convolution_block(out, filters=512, kernel_size=3, strides=2, trainable=trainable)
    print(f"(DARKNET)The shape of the output after layer 5: {out.shape}")

    for _ in range(8):
        out = darknet53_residual_block(out, filters=256, trainable=trainable)
    print(f"(DARKNET)The shape of the output after residual layer 4 (route2): {out.shape}")

    route2 = out

    out = darknet_convolution_block(out, filters=1024, kernel_size=3, strides=2, trainable=trainable)

    print(f"(DARKNET)The shape of the output after layer 6: {out.shape}")

    for _ in range(4):
        out = darknet53_residual_block(out, filters=512, trainable=trainable)

    print(f"(DARKNET)The shape of the output after residual layer 5 (output): {out.shape}")
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
    print(f"(YOLOCONV)The shape of the input: {input_tensor.shape}")
    out = darknet_convolution_block(inputs, filters=filters, kernel_size=1, trainable=trainable)
    print(f"(YOLOCONV)The shape of the output after layer 1: {out.shape}")

    out = darknet_convolution_block(out, filters=2 * filters, kernel_size=3, trainable=trainable)
    print(f"(YOLOCONV)The shape of the output after layer 2: {out.shape}")

    out = darknet_convolution_block(out, filters=filters, kernel_size=1, trainable=trainable)
    print(f"(YOLOCONV)The shape of the output after layer 3: {out.shape}")

    out = darknet_convolution_block(out, filters=2 * filters, kernel_size=3, trainable=trainable)
    print(f"(YOLOCONV)The shape of the output after layer 4: {out.shape}")

    out = darknet_convolution_block(out, filters=filters, kernel_size=1, trainable=trainable)
    print(f"(YOLOCONV)The shape of the output after layer 5(route): {out.shape}")

    route = out

    out = darknet_convolution_block(out, filters=2 * filters, kernel_size=3, trainable=trainable)
    print(f"(YOLOCONV)The shape of the output after layer 6(output): {out.shape}")

    yolo_conv_block = tf.keras.Model(inputs, (route, out), name=name)
    return yolo_conv_block


def yolo_output_layer(
        name: str,
        input_tensor: tf.Tensor,
        filters: int,
        trainable: bool,
        n_classes: int,
        anchors: List[Tuple[int, int]],
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
    n_anchors = len(anchors)
    inputs = Input(input_tensor.shape[1:])
    out = darknet_convolution_block(inputs, filters=filters, kernel_size=3, trainable=trainable)
    print(f"(YOLOOUT)The shape of the output after layer 1: {out.shape}")
    out = darknet_convolution_block(
        out, filters=n_anchors * (n_classes + 5), kernel_size=1, use_batch_norm=False, trainable=trainable
    )
    print(f"(YOLOOUT)The shape of the output after layer 2: {out.shape}")
    out = Lambda(
        lambda out: tf.reshape(
            out, (-1, tf.shape(out)[1], tf.shape(out)[2], n_anchors, n_classes + 5)
        )
    )(out)
    print(f"(YOLOOUT)The shape of the output after layer 3: {out.shape}")
    return tf.keras.Model(inputs, out, name=name)


def build_boxes(
        inputs: tf.Tensor,
        anchors: List[Tuple[int, int]],
        n_classes: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes top left and bottom right points of the boxes.
    :param anchors: A list of anchor sizes.
    :param inputs: Input tensor
    :param n_classes: Number of classes
    :return: Output tensor
    """
    n_anchors = len(anchors)

    # inputs: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(inputs)[1:3][::-1]
    grid_y, grid_x = tf.shape(inputs)[1], tf.shape(inputs)[2]

    box_xy, box_wh, objectness, class_probs = tf.split(inputs, (2, 2, 1, n_classes), axis=-1)
    box_xy = tf.sigmoid(box_xy)

    objectness = tf.math.sigmoid(objectness)
    class_probs = tf.nn.softmax(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_x), tf.range(grid_y))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * n_anchors

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


class YoloV3:
    """
    Yolo v3 model class.
    """

    def __init__(self, n_classes, img_size, max_output_size, iou_threshold,
                 confidence_threshold):
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

    def __call__(self, trainable=False):
        """
        Add operations to detect boxes for a batch of input images.

        :param trainable: A boolean, whether to use in training or inference mode.
        :return:
        """
        inputs = Input([self.img_size[0], self.img_size[1], 3])

        route1, route2, out = darknet53(
            name="Darknet_Block",
            trainable=trainable,
        )(inputs)
        print(f"The shape of the output after darknet block: {out.shape}")
        route, out = yolo_convolution_block(
            name="Yolo_Conv_0",
            input_tensor=out,
            filters=512,
            trainable=trainable
        )(out)
        print(f"The shape of the output after yolo convolution block: {out.shape}")
        detect1 = yolo_output_layer(
            name="Yolo_Output_0",
            input_tensor=out,
            filters=512,
            trainable=trainable,
            n_classes=self.n_classes,
            anchors=ANCHORS[6:9]
        )(out)
        print(f"The shape of the output after yolo output layer 0 (detect 1): {detect1.shape}")

        out = darknet_convolution_block(out, filters=256, kernel_size=1, trainable=trainable)

        out = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")(out)
        print(f"The shape of the output after upsampling layer: {out.shape}")
        out = Concatenate(axis=3)([out, route2])

        route, out = yolo_convolution_block(
            name="Yolo_Conv_1",
            input_tensor=out,
            filters=256,
            trainable=trainable
        )(out)
        detect2 = yolo_output_layer(
            name="Yolo_Output_1",
            input_tensor=out,
            filters=256,
            trainable=trainable,
            n_classes=self.n_classes,
            anchors=ANCHORS[3:6]
        )(out)
        print(f"The shape of the output after yolo output layer 1 (detect 2): {detect2.shape}")

        out = darknet_convolution_block(out, filters=128, kernel_size=1, trainable=trainable)

        out = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")(out)
        print(f"The shape of the output after upsampling layer 2: {out.shape}")
        out = Concatenate(axis=3)([out, route1])

        route, out = yolo_convolution_block(
            name="Yolo_Conv_2",
            input_tensor=out,
            filters=128,
            trainable=trainable
        )(out)
        detect3 = yolo_output_layer(
            name="Yolo_Output_2",
            input_tensor=out,
            filters=128,
            trainable=trainable,
            n_classes=self.n_classes,
            anchors=ANCHORS[0:3]
        )(out)
        print(f"The shape of the output after yolo output layer 2 (detect 3): {detect3.shape}")

        if trainable:
            return Model(inputs, (detect1, detect2, detect3), name='Yolov3')

        print(f"The shape of the 3 output layers: Layer 82: {detect1.shape}, "
              f"Layer 94: {detect2.shape}, Layer 106: {detect3.shape}")

        boxes_0 = Lambda(lambda x: build_boxes(x, ANCHORS[6:9], self.n_classes), name='yolo_boxes_0')(detect1)
        boxes_1 = Lambda(lambda x: build_boxes(x, ANCHORS[3:6], self.n_classes), name='yolo_boxes_1')(detect2)
        boxes_2 = Lambda(lambda x: build_boxes(x, ANCHORS[0:3], self.n_classes), name='yolo_boxes_2')(detect3)

        print(f"The shape of the output before NMS: BBOX: {boxes_0[0].shape}"
              f" OBJECTNESS: {boxes_0[1].shape} CLASS_PROB: {boxes_0[2].shape}"
              f" PRED: {boxes_0[3].shape}")
        outputs = Lambda(
            lambda x: non_max_suppression(
                x, max_output_size=100, iou_threshold=0.5, confidence_threshold=0.5
            ),
            name='yolo_nms'
        )((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        print(f"The shape of the output after NMS: BBOX: {outputs[0].shape}"
              f" OBJECTNESS: {outputs[1].shape} CLASS_PROB: {outputs[2].shape}"
              f" PRED: {outputs[3].shape}")

        return Model(inputs, outputs, name='Yolov3')






