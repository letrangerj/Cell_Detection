# Cell Analysis Pipeline

This project is a pipeline for analyzing cellular images using YOLO (You Only Look Once) for cell detection and SAM (Segment Anything Model) for segmentation and intensity measurement.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [File Structure](#file-structure)
4. [Usage](#usage)
5. [Scripts](#scripts)
6. [Model Training](#model-training)

## Overview

This pipeline is designed to process multi-round fluorescence microscopy images of cells. It performs the following tasks:
- Detects cells using a custom-trained YOLO model
- Segments cells using the SAM model
- Measures cell intensities across different imaging rounds
- Optionally generates cell contours

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install ultralytics segment-anything opencv-python numpy torch tqdm roboflow
   ```
3. Download the SAM model checkpoint and place it in the `weights` directory:
   ```
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/
   ```

## File Structure

```
project_root/
│
├── main.py
├── SAM_Prediction.py
├── Yolo_Prediction.py
├── Yolo_Training.py
│
├── weights/
│   └── sam_vit_h_4b8939.pth
│
└── dataset/
    └── (YOLO dataset files)
```

## Usage

1. Prepare your image data in the following structure:
   ```
   Group_path/
   ├── frame_1/
   │   ├── R1/
   │   │   ├── 1.png
   │   │   ├── 2.png
   │   │   └── ...
   │   ├── R2/
   │   ├── R3/
   │   ├── R4/
   │   └── channels/
   │       ├── R1ch0.png
   │       ├── R2ch0.png
   │       └── ...
   ├── frame_2/
   └── ...
   ```

2. Update the `Group_path` and `Result_path` variables in `main.py`.

3. Run the main script:
   ```
   python main.py
   ```

## Scripts

### main.py
The entry point of the pipeline. It processes all frames and generates results.

### SAM_Prediction.py
Handles cell segmentation and intensity measurement using the SAM model.

### Yolo_Prediction.py
Performs cell detection using the custom-trained YOLO model.

### Yolo_Training.py
Contains the code for training the custom YOLO model on cell images.

## Model Training

To train the YOLO model on your own dataset:

1. Prepare your dataset using Roboflow or a similar tool.
2. Update the Roboflow API key and project details in `Yolo_Training.py`.
3. Run the training script:
   ```
   python Yolo_Training.py
   ```

This will train the model, validate it, and optionally deploy it to Roboflow.

## Output

The pipeline generates the following outputs in the `Result_path`:
- CSV files with cell intensities for each frame
- Optional CSV files with cell contours
- YOLO prediction visualization images

## Note

This pipeline is designed for research purposes and may require adjustments based on your specific imaging setup and cellular analysis needs.