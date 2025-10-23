# Real-time Object Detection using YOLOv8 and OpenCV

This project uses your computer's webcam to capture a live video stream and performs real-time object detection, identifying objects like people, cars, books, bottles, etc. It utilizes the popular [YOLOv8](https://github.com/ultralytics/ultralytics) (You Only Look Once) deep learning model.



---

## Technologies Used

* **Python**
* **OpenCV** - For capturing and displaying the live video stream.
* **Ultralytics (YOLOv8)** - For the pre-trained object detection model.

---

## How to Run

### 1. Prerequisites

Make sure you have **Python** and **pip** installed on your system.

### 2. Install Libraries

Open your terminal or command prompt and run the following command to install the necessary libraries:

```bash
pip install opencv-python ultralytics
3. Run the Project
Run the Python script from your terminal:

Bash

python detect.py
A window will open showing your webcam feed with detection boxes drawn around objects.

To quit the program, make sure the video window is active and press the 'q' key on your keyboard.