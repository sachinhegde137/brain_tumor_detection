import shutil
import time
import os
import sys
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import preprocess_image, draw_outputs, make_eval_model_from_trained_model
from src.models import loss_function, ANCHORS, ANCHOR_MASKS


output_img_name = "output.png"

def main(args):
    #yolo = YoloV3(
    #    n_classes= 4,
    #    img_size=args.input_shape,
    #    trainable=False
    #)

    # Load the model
    model = load_model(args.model_path, custom_objects={'yolo_loss': loss_function(np.zeros((3, 2), np.float32))})
    model = make_eval_model_from_trained_model(model, ANCHORS, ANCHOR_MASKS)
    print(f"Model loaded")

    # Preprocess image
    original_image, preprocessed_image = preprocess_image(
        image_path=args.image_path,
        input_shape=(args.input_shape, args.input_shape)
    )
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    t1 = time.time()
    boxes, scores, classes,valid_detections = model.predict(preprocessed_image)
    print(f"The shape of the boxes: {boxes.shape}")
    print(f"The shape of the scores: {scores.shape}")
    print(f"The shape of the classes: {classes.shape}")
    print(f"The shape of the valid_detections: {valid_detections.shape}")
    t2 = time.time()
    print(f"time: {t2 - t1}")

    print("Detections:")
    for i in range(boxes.shape[1]):
        print(f"\t{int(classes[0][i])}, {np.array(scores[0][i])}, {np.array(boxes[0][i])}")

    print(f"Number of valid detections: {int(valid_detections)}")

    results_dir = "inference_results/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir, ignore_errors=True)
    else:
        os.makedirs(results_dir)

    output_path = os.path.join(results_dir, output_img_name)
    img = draw_outputs(original_image, (boxes[0], scores[0], classes[0]))
    cv2.imwrite(output_path, img)
    print(f"output saved to: {output_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object detection using YoloV3 model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--input_shape", type=int, required=True, help="Input shape (width and height) of the model.")

    args = parser.parse_args()
    main(args)