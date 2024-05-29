# Traffic Counter System

This Python script utilizes the YOLO (You Only Look Once) object detection model to detect cars in a given video, tracks them, and counts the number of cars passing through designated lines.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)

## Overview
The script takes input from a video file, applies YOLO object detection to detect cars within the frame, and tracks the detected cars using the DeepSORT (Deep Simple Online and Realtime Tracking) algorithm. The script counts the number of cars passing through predefined lines and annotates the video feed with bounding boxes around the detected cars and a counter for the total number of cars.

## Features
- Real-time car detection and tracking.
- Counting the number of cars passing through designated lines.
- Annotating the video feed with bounding boxes and counters.
- Customizable parameters for thresholding, line positions, and video paths.

## Prerequisites
Ensure you have the following installed:
- Python 3.6 or later
- `pip` package installer
- GPU with CUDA support (optional for faster processing)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/morsechim/TrafficCounter.git
    cd TrafficCounter
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your environment:
    - Place the YOLOv9c weights file in the `weights` directory.
    - Place your input video file in the `videos` directory.
    - Ensure the `mask.png` image is in the `images` directory.
    - Ensure the `classes.names` file is in the `data` directory.

2. Run the script:
    ```bash
    python main.py
    ```

3. View the output:
    - The processed video with detections will be saved as `output.mp4` in the `videos` directory.
    - During processing, the script will display the video with annotated detections and count overlays.

## Configuration
- **Model and Device:**
    - The YOLO model weights are expected to be located at `./weights/yolov9c.pt`.
    - The script automatically selects the processing device (`mps` if available, otherwise `cpu`).

- **Video Input/Output:**
    - Input video path: `./videos/traffic.mp4`
    - Output video path: `./videos/output.mp4`

- **Counter Lines:**
    - Upward counting line coordinates: `[459, 265, 657, 280]`
    - Downward counting line coordinates: `[684, 288, 885, 282]`

## Acknowledgments
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- **[YOLOv9 by Ultralytics](https://github.com/WongKinYiu/yolov9)**
- **[DeepSORT: Deep Simple Online and Realtime Tracking](https://github.com/levan92/deep_sort_realtime)**

This project is inspired by the need for efficient traffic monitoring systems utilizing computer vision techniques.

---

*Note: Customize the repository URL, paths, and any other project-specific details as needed.*
