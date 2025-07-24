import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    """
    Converts the slope and intercept of a line into coordinates.
    """
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    if slope == 0: # Avoid division by zero
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Averages the slope and intercept of detected lines to find the left and right lanes.
    """
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2: # Skip vertical lines
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    left_line = None
    right_line = None
    
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    
    return left_line, right_line

def canny(image):
    """
    Applies Canny edge detection to an image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    return canny_image

def display_lines(image, lines):
    """
    Draws the detected lines on an image.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10) # Green lines
    return line_image

def region_of_interest(image):
    """
    Creates a mask for the region of interest in the image.
    """
    height = image.shape[0]
    width = image.shape[1]
    # Defines a trapezoidal area in the bottom half of the screen
    polygons = np.array([
        [(int(width*0.1), height), (int(width*0.9), height), (int(width*0.55), int(height*0.6)), (int(width*0.45), int(height*0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def process_frame(frame):
    """
    This is the main processing pipeline for a single frame.
    """
    # 1. Apply Canny Edge Detection
    canny_image = canny(frame)
    
    # 2. Define Region of Interest
    cropped_image = region_of_interest(canny_image)
    
    # 3. Detect Lines using Hough Transform
    lines = cv2.HoughLinesP(
        cropped_image, 
        2, # rho
        np.pi / 180, # theta
        100, # threshold
        np.array([]), 
        minLineLength=40, 
        maxLineGap=5
    )
    
    # 4. Average the detected lines to get the final lane lines
    left_line, right_line = average_slope_intercept(frame, lines)
    
    # 5. Create an image with the detected lane lines
    line_image = display_lines(frame, [left_line, right_line])
    
    # 6. Combine the original frame with the detected lanes
    # This creates the final output image with lanes overlaid.
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return combo_image
