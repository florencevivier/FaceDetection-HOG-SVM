# Face Detection with HOG + SVM

## Project Overview

This project implements a lightweight face detection system built from scratch using classical computer vision techniques.

The objective is to detect human faces in images without using pre-trained deep learning models and while keeping computational requirements low.

The project is inspired by:

- Navneet Dalal and Bill Triggs – *Histograms of Oriented Gradients for Human Detection (2005)*
- Adrian Rosebrock – Sliding Window & Image Pyramid techniques

The system combines:

- HOG (Histogram of Oriented Gradients) feature extraction
- Linear Support Vector Machine (SVM)
- Image Pyramid for multi-scale detection
- Sliding Window scanning
- Non-Maximum Suppression (NMS)

The final pipeline detects zero, one, or multiple faces in real-world images.

---

## Objective

The goal is to design a face detector that:

- Works without pre-trained models
- Uses limited computational resources
- Is interpretable and modular
- Can generalize from cropped training faces to full real-world images

From a computer vision perspective:

- False Negatives (missed faces) are undesirable
- False Positives (detecting faces where none exist) must be controlled
- Multi-scale detection is required for robustness

---

## Dataset

### Positive Samples

- 400 grayscale face images (64x64)
- Source: Scikit-learn Olivetti Faces dataset
- Images already normalized
- Faces centered and tightly cropped

### Negative Samples

- Extracted from TensorFlow Datasets Caltech101 dataset
- Classes containing faces removed ("faces", "faces_easy", "buddha")
- Images:
  - Converted to grayscale
  - Centrally cropped
  - Resized to 64x64
  - Normalized
- Balanced dataset (~400 negative samples selected)

Final dataset:
- Binary classification (face / no face)
- Balanced classes
- Images flattened for model training

---

## Data Preparation

Preprocessing steps:

- Grayscale conversion
- Central square cropping
- Resize to 64x64
- Pixel normalization (0–1)
- Label encoding (1 = face, 0 = no face)
- Train/Test split (70/30)

Visualization checks were performed at each stage.

---

## Feature Engineering

### HOG (Histogram of Oriented Gradients)

Each image is transformed into HOG features with:

- 9 orientations
- 8x8 pixels per cell
- 2x2 cells per block
- L2-Hys normalization

HOG captures edge orientation patterns, which are highly discriminative for face structure.

A custom `HOGTransformer` class was implemented to integrate feature extraction into a Scikit-learn pipeline.

---

## Model

### Linear SVM Classifier

Pipeline structure:

HOG → StandardScaler → Linear SVC

Why Linear SVM?

- Efficient for high-dimensional data
- Works well with HOG features
- Low computational cost
- Good interpretability

### Training Results

Both training and test sets achieved:

- Accuracy: 100%
- Precision: 1.00
- Recall: 1.00
- F1-score: 1.00

This indicates perfect separation on the curated dataset.

## From Classification to Detection

The trained classifier only works on 64x64 cropped images.

To detect faces in real-world images, additional techniques were implemented.

### 1. Image Pyramid

Generates progressively smaller versions of the image to detect faces at different scales.

### 2. Sliding Window

Scans the image using a fixed 64x64 window across all pyramid levels.

For each window:

- Preprocessing applied
- HOG features extracted
- SVM decision score computed
- Windows above threshold retained

### 3. Non-Maximum Suppression (NMS)

Multiple overlapping detections are reduced to a single bounding box per face.

Implemented using OpenCV’s NMS method.


## Final Detection Pipeline

The final `face_detection_pipeline()` function performs:

1. Multi-scale image generation
2. Sliding window scanning
3. Classification via HOG + SVM
4. Threshold filtering
5. Non-Maximum Suppression

Input:
- RGB image

Output:
- List of bounding boxes (x, y, width, height, score)


## Experimental Results

The pipeline was tested on:

- Landscape images (no faces) → No detections
- Single-face images → Correct localization
- Multi-face images → All faces detected
- Tilted faces → Detected with adjusted threshold

The detector generalizes well despite being trained only on tightly cropped faces.


## Key Learning Points

- Classical computer vision vs deep learning approaches
- HOG feature extraction theory and implementation
- Linear SVM for image classification
- Multi-scale object detection
- Sliding window computational trade-offs
- Non-Maximum Suppression mechanics
- From classification to object detection pipeline design


## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- TensorFlow Datasets
- PIL
- imutils
