This is the homework for the sixth week.

### Perspective Transformation and Canny Edge Detection
This repository demonstrates how to implement perspective transformation and Canny edge detection using OpenCV.
Perspective transformation is used to convert an image from one viewpoint to another, while Canny edge detection 
is employed to detect edges within an image.

## Usage
### Perspective Transformation
To perform perspective transformation, you need to provide the coordinates of four points in the original image, representing the four 
corners of the region to be transformed, as well as the corresponding positions of these points in the target image. 
Then, you can use OpenCV's `cv2.getPerspectiveTransform()` function to compute the perspective transformation matrix and apply it to the image.

### Canny Edge Detection
Canny edge detection is a popular edge detection algorithm that uses gradient magnitudes to detect edges within an image.
