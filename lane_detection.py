import cv2
import numpy as np

#isolates pixels that are likely to be lane lines
def _apply_color_threshold(image):
    rgb_threshold = [200, 200, 200]
    thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                 (image[:,:,1] < rgb_threshold[1]) | \
                 (image[:,:,2] < rgb_threshold[2])
    color_selected = image.copy()
    color_selected[thresholds] = [0,0,0]
    return color_selected

def _apply_region_of_interest_mask(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def _draw_hough_lines(image, hough_lines):
    line_image = np.zeros_like(image)
    if hough_lines is not None:
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def process_frame(frame):
    color_selected_frame = _apply_color_threshold(frame)

    #defining dynamic region of interest
    height, width = frame.shape[:2]
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    
    #canny edge detection
    gray_frame = cv2.cvtColor(color_selected_frame, cv2.COLOR_RGB2GRAY)
    canny_edges = cv2.Canny(gray_frame, 50, 150)
    
    #masking canny edges
    masked_edges = _apply_region_of_interest_mask(canny_edges, roi_vertices)

    rho = 1
    theta = np.pi/180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    line_image = _draw_hough_lines(frame, lines)
    
    #combining line with orignal feed
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return final_image
