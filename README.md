# Brain_tumor_detection using Yolov3
This repository implements Yolov3 architecture using Tensorflow. The object detection model is trained on brain tumor
dataset consisting of MRI images for the purpose of brain tumor detection.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Dataset
The dataset used in this project consists of MRI images labeled with the presence and location of brain tumors. Each image is annotated with bounding boxes around the detected tumors. The dataset is split into training and validation sets. 
The images in the dataset are from different angles of MRI scans including sagittal, axial, and coronal views. This variety ensures comprehensive coverage of brain anatomy, enhancing the robustness of models trained on this dataset.


- **Dataset Source:** https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes
- **Number of Images:** 5,249 images
- **Image Dimensions:** 512x512 pixels
- **Number of Classes:** 4 (Glioma, Meningioma, No Tumour, Pituitary)

### Data Split
1. Training Set:
- Glioma: 1,153 images
- Meningioma: 1,449 images
- No Tumor: 711 images
- Pituitary: 1,424 images
2. Validation Set:
- Glioma: 136 images
- Meningioma: 140 images
- No Tumor: 100 images
- Pituitary: 136 images

## Model Architecture
The model used in this project is based on the YOLOv3 architecture, which is a state-of-the-art object detection model. YOLOv3 is designed for both speed and accuracy, making it highly effective for real-time object detection tasks, including medical imaging applications like brain tumor detection.

### Key Components of YOLOv3

1. **Backbone - Darknet-53:**
   - YOLOv3 uses Darknet-53 as its backbone network, a 53-layer convolutional neural network that is more powerful and efficient compared to the previous Darknet-19 used in YOLOv2.
   - Darknet-53 is characterized by its use of residual connections, inspired by ResNet, which help in training deeper networks by mitigating the vanishing gradient problem.
   - This backbone extracts hierarchical features from the input image, which are then used by the detection layers to predict bounding boxes and class probabilities.

2. **Detection at Multiple Scales:**
   - YOLOv3 predicts objects at three different scales, improving its ability to detect smaller objects like brain tumors that might be missed by single-scale detectors.
   - This is achieved by using feature maps from three different layers in the network, each corresponding to a different resolution. The model detects larger objects on lower resolution layers and smaller objects on higher resolution layers.

3. **Bounding Box Predictions:**
   - YOLOv3 divides the input image into an S x S grid. For each grid cell, it predicts several bounding boxes along with confidence scores and class probabilities.
   - Bounding boxes are predicted using anchor boxes, predefined boxes that serve as references for the size and aspect ratio of detected objects. YOLOv3 uses dimension clusters as anchor boxes, which are calculated based on the dataset.

4. **Loss Function:**
   - The loss function used in YOLOv3 is a combination of multiple components:
     - **Binary Cross-Entropy Loss:** Used for class predictions and objectness score (which indicates whether an object is present in a grid cell).
     - **Mean Squared Error (MSE):** Applied to the bounding box coordinates (x, y, width, height).
   - This loss function helps in accurately localizing the tumor regions in MRI images and assigning the correct class labels

6. **Anchor Boxes:**
   - The model utilizes nine anchor boxes of varying sizes, determined using k-means clustering on the dataset’s bounding boxes. This allows the network to predict bounding boxes more accurately by using anchors that closely match the dimensions of tumors in the dataset.

### Advantages of YOLOv3 for Brain Tumor Detection

- **Real-Time Detection:** YOLOv3’s architecture allows for the processing of multiple images per second, making it suitable for real-time detection tasks.
- **High Accuracy:** The use of multi-scale detection and the Darknet-53 backbone improves the accuracy of detecting small, complex objects like brain tumors.
- **End-to-End Learning:** YOLOv3 is an end-to-end learning model, meaning it can be trained directly on raw images and their corresponding annotations without needing to pre-process images into multiple scales or aspect ratios.

Overall, YOLOv3's architecture is well-suited for the task of detecting brain tumors from MRI scans, offering a robust balance between speed and detection accuracy.


## Installation
To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/sachinhegde137/brain_tumor_detection.git
   cd brain_tumor_detection

2. Create a virtual environment and activate it:
   ```bash
   python -m venv my_env
   source my_env/bin/activate  # On linux
   venv\Scripts\activate  # On windows
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
4. Download the dataset from this link: [MRI Dataset for Brain tumor detection](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes).
The dataset has 'Train' and 'Val' directories. Each of these directories contain 4 sub-directories corresponding to 4 classes.
Place the 'Train' and 'Val' folders in the 'dataset' directory in the project directory.

## Acknowledgements
- This project is inspired by the original [YOLOv3 paper](https://arxiv.org/pdf/1804.02767) by Joseph Redmon and Ali Farhadi.
- Thanks to the [dataset provider](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes) for making the brain tumor dataset available.