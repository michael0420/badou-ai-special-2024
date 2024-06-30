## Week 7 Homework Programs

This repository contains implementations for multiple programs as part of the seventh week's homework assignment.

### Programs Included:

1. **RANSAC (Random Sample Consensus) Implementation**

   This program implements the RANSAC algorithm, which is used for fitting a model to data with a significant number of outliers. The RANSAC algorithm iteratively selects a random subset of the data, fits a model to this subset, and evaluates how well the model fits the rest of the data.

   - **Input Parameters**:
     - `data`: The input sample points.
     - `model`: The hypothesized model to be fit.
     - `n`: The minimum number of data points required to fit the model.
     - `k`: The maximum number of iterations allowed in the algorithm.
     - `t`: A threshold value for determining when a data point fits a model.
     - `d`: The minimum number of points required to assert that a model fits well to data.
   
   - **Output**:
     - `bestfit`: The best fitting model after the iterations.

   - **Usage**:
     The program is designed to work with linear least squares models and can be executed with user input for parameters like sample size, number of iterations, and threshold values. It includes functionalities for adding noise and outliers to the data and visualizing the results.

   - **File**: `ransac2.py`

2. **Hashing Algorithms for Image Comparison**

   This program includes implementations of three perceptual hashing algorithms (aHash, dHash, pHash) for comparing images. These algorithms generate hash values based on image content, allowing for efficient similarity comparison.

   - **Functions**:
     - `aHash(img)`: Average Hashing, which generates a hash based on the average pixel value.
     - `dHash(img)`: Difference Hashing, which generates a hash based on the difference in pixel values.
     - `pHash(img_file)`: Perceptual Hashing, which uses the discrete cosine transform to generate a hash.
   
   - **Additional Functions**:
     - `cmp_hash(hash1, hash2)`: Compares two hash values and returns their similarity.
     - `deal(img1_path, img2_path)`: Computes and prints the similarity between two images using the three hashing algorithms.

   - **Usage**:
     The program reads two images, computes their hash values using the three algorithms, and prints the similarity scores. It also includes a main function to test the performance and accuracy of the algorithms with various image transformations.

   - **File**: `Hash_all.py`

### Date: 2024-05-28
