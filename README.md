# Computer-Vision-CMU-16720A

**Instructor:** Deva Ramanan<br>
**Semester:** Fall 2022

> *This course introduces the fundamental techniques used in computer vision, that is, the analysis of patterns in visual images to reconstruct and understand the objects and scenes that generated them. Topics covered include image formation and representation, camera geometry, and calibration, computational imaging, multi-view geometry, stereo, 3D reconstruction from images, motion analysis, physics-based vision, image segmentation and object recognition. The material is based on graduate-level texts augmented with research papers, as appropriate.*

<hr style="border:1px solid gray">

## Homework 1: Spatial Pyramid Matching for Scene Classification

### **Procedure:**<br>
![bow_diagram](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/BOW-diagram.png)


**Topics Covered:**
1. Feature Extraction based on Filter Banks
2. K Means Clustering
3. Visual Word Dictionary
4. Scene Classification
5. Hyperparameters Tuning

### **Results:** <br>
![visual-words](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/visual-words.png)

**Description.** <br>*Visual words for three sample images from the SUN database.*

<hr style="border:1px solid gray">

## Homework 2: Lucas-Kanade Object Tracking

### **Procedure:**<br>
![lk-tracking-graphic](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/Lk-tracking-schematic.png)

**Topics Covered:**
1. Simple Lucas & Kanade Tracker with Naive Template Update
2. Lucas & Kanade Tracker with Template Correction
3. Two-dimensional Tracking with a Pure Translation Warp Function
4. Two-dimensional Tracking with a Plane Affine Warp Function
5. Lucas & Kanade Forward Additive Approach
6. Lucas & Kanade Inverse Compositional Approach

### **Results:**

![lk-results](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/Drift_Correction.png)

**Description:** <br>*Lucas-Kanade tracking using Naive Template Update (purple) versus Template Correction (Red).*

<hr style="border:1px solid gray">

## Homework 3: Augmented Reality with Planar Homographies

### **Procedure:**<br>
![homographies-graphic](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/homography-diagram.png)

**Topics Covered:**
1. Direct Linear Transform
2. Matrix Decomposition to calculate Homography
3. Limitations of Planar Homography
4. FAST Detector and BRIEF Descriptors
5. Feature Matching
6. Compute Homography via RANSAC
7. Automated Homography Estimation and Warping
8. Augmented Reality Application using Homography
9. Real-Time Augmented Reality with High FPS
10. Panorama Generation based on Homography

### **Results:**<br>
![kf-panda-graphic](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/ar_kungfu_panda.gif)

**Description:** <br>*Augmented reality clip, superimposing a video sequence onto a book cover - using Planar Homographies.*

<hr style="border:1px solid gray">

## Homework 4: 3D Reconstruction


### **Procedure:**<br>
![3d-graphic](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/3d_recon_diag.png)


**Topics Covered:**
1. Fundamental Matrix Estimation using Point Correspondence
2. Metric Reconstruction
3. Retrieval of Camera Matrices up to a Scale and Four-Fold Rotation Ambiguity
4. Triangulation using the Homogeneous Least Squares Solution
5. 3D Visualization from a Stereo-Pair by Triangulation and 3D Locations Rendering
6. Bundle Adjustment
7. Estimated fundamental matrix through RANSAC for noisy correspondences
8. Jointly optmized reprojection error w.r.t 3D estimated points and camera matrices
9. Non-linear optimization using SciPy least square optimizer

### **Results:**<br>
![temple](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/temples.png)
![temple-reconstr](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/3d-reconst.png)

**Description:** <br>*Temple (top) reconstructed in 3D (bottom).*


<hr style="border:1px solid gray">

## Homework 5: Neural Networks for Recognition

### **Procedure:**<br>
![neural-net](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/deep_nn.png)

**Topics Covered:**
1. Manual Implementation of a Fully Connected Network
2. Text Extraction from Images of Handwritten Characters
3. PyTorch Implementation of a Convolutional Neural Network
4. Fine Tuning of SqueezeNet in PyTorch
5. Comparison between Fine Tuning and Training from Scratch

### **Results:**<br>
![raw-note](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/03_haiku.jpg)
![nn-reading](https://github.com/artrela/Computer-Vision-CMU-16720A/blob/master/Results/dl-results.png)

**Description:** <br>*Neural network text recognition results, based on raw images (example on top).*
