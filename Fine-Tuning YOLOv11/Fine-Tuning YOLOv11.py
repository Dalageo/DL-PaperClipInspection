''' 
# YOLOv11 Implementation

YOLO (You Only Look Once) is a real-time object detection algorithm 
that can identify and classify multiple objects in an image or video frame in a single pass.
Unlike traditional object detection methods that perform region proposals and then classify each region, 
YOLO treats object detection as a single regression problem, directly predicting bounding boxes and 
class probabilities from the full image in one step. YOLO is exceptionally fast because it processes 
the entire image in one pass through the network, rather than analyzing smaller parts independently. 
By considering the global context of the image, it reduces false positives while maintaining high detection 
accuracy. This efficiency and precision make YOLO ideal for real-time applications such as video analysis 
and self-driving cars.

## How YOLO Works

YOLO operates by dividing the input image into a grid, where each grid cell is responsible for detecting 
objects whose center lies within that cell. In other words, if an object's center is inside a particular 
grid cell, that cell predicts the bounding box for the object, including its position, size, and class. 
Each grid cell predicts several key pieces of information: the coordinates of bounding boxes (including 
the center, width, and height), a confidence score indicating how confident the model is that an object 
exists in the bounding box, and class probabilities for identifying which object the box contains. 
After the model makes its predictions, it applies Non-Maximum Suppression (NMS) to filter out redundant 
bounding boxes, keeping only the most confident ones based on their scores.

## How YOLO Learns (Training Phase)

During training, YOLO learns to predict bounding boxes and classify objects by analyzing labeled images. 
These images contain the ground truth for bounding box coordinates and object classes. YOLO uses a loss 
function that includes three components: localization loss to measure the accuracy of the bounding box 
coordinates, confidence loss to evaluate the correctness of the object confidence score, and classification 
loss to assess the accuracy of the class predictions. The training process adjusts the network’s weights to 
minimize these losses and improve detection accuracy.
'''

# Import YOLO from ultralytics
from ultralytics import YOLO

# Load the pretrained obb model since the dataset is annotated with oriented bounding boxes
model = YOLO("yolo11m-obb.pt")

# Train the model for 200 epochs on GPU
model.train( data="PaperClipInspection.yaml", imgsz = 640, 
            batch = 8, epochs = 200, workers = 1, device = 0)

# After training the best YOLO model (based on validation performance) will be saved to:
# 'path_to_your_yolo_folder/runs/obb/train/weights/best.pt'