# Computer Vision Projects

> This repository is made for the Computer Vision course projects - Fall 2018.

**Dependencies:**
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

---

This repository contains implementation of the following projects in Python using OpenCV:
- [Connected Components](/connected_components)
  - Detecting a specific type of connected components.
  - Counting the number of connected components in an input image.
- [Corner Detection (Harris - Sobel)](/corner_detection_harris_sobel)
  - Implementing `Harris` corner detection algorithm using `Sobel`.
- [Face Detection (Haar cascade)](/face_detection_haar)
  - Detecting faces of an input image by using `Haar cascade`.
- [Histogram Stretching and Histogram Clipping](histogram_stretching_clipping)
  - Processing an image with `Histogram Streching` and `Histogram Clipping`.
- [Line segmentation (RANSAC - LSD - Hough)](line_RANSAC_LSD_Hough)
  - Circle detection using OpenCV's `HoughCircles`.
  - Implementing `RANSAC` line detection in Python.
  - Detecting rectangles using `Line Segment Detector` and `Hough` Algorithm.
- [Live Face Detection (Kalman)](/live_face_detection_Kalman)
  - Live face detection using built-in OpenCV's methods and `Kalman` filter.
- [Noise Reduction (Smoothing - Median)](noise_reduction_1)
  - Removing Gaussian and Salt-and-pepper noise from an input image by applying `Smoothing` and `Median` filters.
- [Noise Reduction (discrete Fourier transform)](noise_reduction_2)
  - Removing periodic noise from an input image by applying `Discrete Fourier Transform`.
- [Noise Reduction (Gaussian - Mask)](noise_reduction_3)
  - Removing noise from an input image by applying `Gaussian` and `Mask` filters.
- [Automatic Image Thresholding (Otsu - Adaptive)](otsu-threshold)
  - Comparing global and adaptive thresholding on an input image.
  - Implementing `Otsu` method for Automatic Image Thresholding.
- [Scanner](scanner)
  - Implementing a simple document scanner in Python with OpenCV.
