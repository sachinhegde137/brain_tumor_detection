import numpy as np
import tensorflow as tf

import cv2


INPUT_SHAPE = (416, 416)
BATCH_SIZE = 8
MAX_BOXES = 100

@tf.function
def transform_targets_for_output(y_true, grid_y, grid_x, anchor_idxs, classes):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((N, grid_y, grid_x, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    # Specify the element shapes for TensorArrays
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True, element_shape=(4,))
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True, element_shape=(6,))
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2.

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_size = tf.cast(tf.stack([grid_x, grid_y], axis=-1), tf.float32)
                grid_xy = tf.cast(box_xy * grid_size, tf.int32)

                assert grid_xy.shape == (2,), "Tensor must have exactly two values."

                # Apply the condition to limit the tensor values
                grid_xy = tf.where(grid_xy >= grid_x, grid_x-1, grid_xy)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1
    indices = indexes.stack()
    updates = updates.stack()
    y_true_out = tf.tensor_scatter_nd_update(y_true_out, indices, updates)
    return y_true_out


def transform_targets(y_train, size, anchors, anchor_masks, classes):
    #y_train = targets.numpy()
    y_outs = []

    grid_y, grid_x = size[0] // 32, size[1] // 32
    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    # y_train: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_out = transform_targets_for_output(y_train, grid_y, grid_x, anchor_idxs, classes)
        y_outs.append(y_out)
        grid_x *= 2
        grid_y *= 2

    return tuple(y_outs)


def transform_img_and_labels(image_path_tensor: tf.Tensor, label_path_tensor: tf.Tensor):
    image_path = bytes.decode(image_path_tensor.numpy())
    label_path = bytes.decode(label_path_tensor.numpy())

    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, INPUT_SHAPE)
    image_normalized = tf.cast(image, tf.float32) / 255.0

    with open(label_path, 'r') as file:
        raw_label_data = file.readlines()

        if len(raw_label_data) > 0:
            xmins, xmaxs, ymins, ymaxs, classes = [], [], [], [], []
            for line in raw_label_data:
                parts = line.strip().split()
                class_idx = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                xmin = x_center - (width / 2)
                ymin = y_center - (height / 2)
                xmax = x_center + (width / 2)
                ymax = y_center + (height / 2)
                xmins.append(xmin)
                ymins.append(ymin)
                xmaxs.append(xmax)
                ymaxs.append(ymax)
                classes.append(class_idx)
            processed_label_data = np.stack((xmins, ymins, xmaxs, ymaxs, classes), axis=1)

            paddings = [[0, 100 - processed_label_data.shape[0]], [0, 0]]
            processed_label_data = np.pad(processed_label_data, paddings, mode='constant')
            label_tensor = tf.convert_to_tensor(processed_label_data, dtype=tf.float32)
            # Adjust bounding boxes
            scale_x = INPUT_SHAPE[1] / orig_image.shape[1]
            scale_y = INPUT_SHAPE[0] / orig_image.shape[0]

            transformed_label_tensor = label_tensor * tf.constant(
                [scale_x, scale_y, scale_x, scale_y, 1], dtype=tf.float32
            )
        else:
            # Handle case with no labels
            transformed_label_tensor = tf.zeros([100, 5], dtype=tf.float32)

    return image_normalized, transformed_label_tensor

def tf_transform_imgs_labels(image_path_tensor: tf.Tensor, label_path_tensor: tf.Tensor):
    img, label = tf.py_function(
        func=transform_img_and_labels,
        inp=[image_path_tensor, label_path_tensor],
        Tout=[tf.float32, tf.float32]
    )
    img.set_shape([INPUT_SHAPE[0], INPUT_SHAPE[1], 3])
    label.set_shape([MAX_BOXES, 5])
    return img, label


