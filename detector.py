# Import libs
import cv2 as cv
import numpy as np


def cannyDetector(frame):
    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # Apply Gaussian blur (5x5)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # Find edges
    canny = cv.Canny(blur, 50, 150)
    return canny


def segmentDetector(frame):
    # Height of image
    height = frame.shape[0]

    # Create triangular polygon mask
    polygons = np.array([
        [(0, height), (800, height), (380, 290)]
    ])

    # Fill mask with 0 or 1s
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, polygons, 255)

    # Segment between the frame and mask
    segment = cv.bitwise_and(frame, mask)
    return segment


def calculateLines(frame, lines):
    left = []
    right = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # Fits a polynomial to the x,y coords extracted
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]

        if slope < 0:
            # Negative slope = left side
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))

    # Average the slope and y-int in both left + right
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)

    # Calculate the coords from the averages
    left_line = calculateCoordinates(frame, left_avg)
    right_line = calculateCoordinates(frame, right_avg)
    return np.array([left_line, right_line])


def calculateCoordinates(frame, parameters):
    slope, intercept = parameters

    y1 = frame.shape[0]
    y2 = int(y1-150)

    x1 = int((y1-intercept) / slope)
    x2 = int((y2-intercept) / slope)

    return np.array([x1, y1, x2, y2])


def visualizeLines(frame, lines):
    linesVisualize = np.zeros_like(frame)

    # Draw line
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(linesVisualize, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return linesVisualize


# Get vid feed
cap = cv.VideoCapture("input.mp4")

while(cap.isOpened()):
    # Get the frame
    ret, frame = cap.read()

    # Canny transform + show window
    canny = cannyDetector(frame)
    cv.imshow("canny", canny)

    # Segment + show window
    segment = segmentDetector(canny)
    cv.imshow("segment", segment)

    # Hough transform
    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100,
                           np.array([]), minLineLength=100, maxLineGap=50)

    # Visualize
    lines = calculateLines(frame, hough)
    linesVisualize = visualizeLines(frame, lines)
    cv.imshow("hough", linesVisualize)

    # Take weighted sum + arbitrary scalar value
    output = cv.addWeighted(frame, 0.9, linesVisualize, 1, 1)
    cv.imshow("output", output)

    # Separate frames in 10 ms intervals
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Close program
cap.release()
cv.destroyAllWindows()
