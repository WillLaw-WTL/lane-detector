# Lane Detection

Using OpenCV to detect lanes.

## Results

###### Canny

![Canny](./img/canny.jpg)
Responsible for taking the input frame and applying these filters:

1. Turning the image to gray
2. Reduce the noise in the image using Gaussian blur

###### Segment

![Segment](./img/segment.jpg)
Forms a triangular mask for the valid lane area.

###### Hough Transformation

![Hough](./img/hough.jpg)
The goal is to identify two straight lines (left + right).

###### Output

![Result](./img/result.jpg)

## Resources

["Tutorial: Build a lane detector"](https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132) by Chuan-en Lin.
