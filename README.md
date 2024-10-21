# Autonomous Driving Object Detection

This project focuses on object detection for autonomous driving using YOLOv5, a state-of-the-art object detection model. It covers data preparation, image resizing, annotation formatting for YOLO, model training, and evaluation of results.

## Table of Contents
1. [Project Structure](#project-structure)
   - [1. Data Preparation](#1-data-preparation)
   - [2. Image Preprocessing](#2-image-preprocessing)
   - [3. YOLOv5 Annotation and Formatting](#3-yolov5-annotation-and-formatting)
   - [4. Model Training](#4-model-training)
   - [5. Inference and Results](#5-inference-and-results)
2. [Key Insights](#key-insights)
3. [Suggestions for Improvement](#suggestions-for-improvement)
4. [Conclusion](#conclusion)
5. [Future Work](#future-work)

## Project Structure

### 1. Data Preparation
The project starts by importing the dataset and preparing it for training. The dataset consists of images and corresponding annotations, including bounding boxes and object labels.


### 2. Image Preprocessing
The images are resized to fit the model's input requirements (640x640), and all the files are organized into 'train', 'validation', and 'test' folders.


### 3. YOLOv5 Annotation and Formatting
The annotation data is converted into YOLO format, which includes object classes and bounding box coordinates normalized to fit within the [0, 1] range.



### 4. Model Training
The YOLOv5 model is trained on the dataset, with results including performance metrics for each object class. Key performance metrics include Precision, Recall, and mAP (mean Average Precision).


### 5. Inference and Results
The trained model is used to perform inference on a new set of images. The results show the bounding boxes around detected objects along with their class labels.

**Screenshots**:
- Example of inference results with detected objects in images.

![InferenceResults](https://github.com/user-attachments/assets/3d5d9a50-ed49-42d3-9f88-7fd8c59c676f)

## Key Insights
- **Performance**: The model performs well on classes like 'work_van' and 'single_unit_truck', but improvements could be made for other vehicle types.
- **YOLO Formatting**: Converting annotations to the YOLO format ensured compatibility with the YOLOv5 model, facilitating smooth training and inference processes.

## Suggestions for Improvement
1. **Data Augmentation**: Introducing techniques like rotation, flipping, or brightness adjustment could improve model robustness.
2. **Hyperparameter Tuning**: Adjusting learning rate and batch size could lead to better performance.
3. **Additional Features**: Including weather and lighting conditions in the dataset could help the model generalize better.

## Conclusion
This project demonstrates the use of YOLOv5 for object detection in the context of autonomous driving. The model shows promising results, though further tuning and data augmentation may yield improved performance.

## Future Work
1. **Fine-Tuning the Model**: Further training on a larger dataset or using a pre-trained model could enhance detection accuracy.
2. **Incorporating External Data**: Adding contextual information like traffic signals or road conditions may improve the model's decision-making ability.
3. **Real-Time Deployment**: Integrating this object detection model into an actual autonomous vehicle system for real-time object detection.
