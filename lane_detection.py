import cv2
import numpy as np
from collections import deque

class Line():
    def __init__(self, n=10):
        self.n = n  
        self.detected = False  
        self.recent_fits = deque(maxlen=self.n) 
        self.best_fit = None  
        self.current_fit = None 

    def update(self, fit):
        """
        Updates the line with a new fit, and calculates the smoothed best_fit.
        """
        if fit is not None:
            self.detected = True
            self.current_fit = fit
            self.recent_fits.append(fit)
            self.best_fit = np.mean(self.recent_fits, axis=0)
        else:
            self.detected = False
            if len(self.recent_fits) > 0:
                self.recent_fits.popleft()
                if len(self.recent_fits) > 0:
                    self.best_fit = np.mean(self.recent_fits, axis=0)
                else:
                    self.best_fit = None
            else:
                self.best_fit = None

def _warp_image(img):
    """(Internal) Transforms perspective to a bird's-eye view."""
    height, width = img.shape[:2]
    src = np.float32([
        [int(width * 0.45), int(height * 0.65)],
        [int(width * 0.55), int(height * 0.65)],
        [int(width * 0.15), int(height * 0.95)],
        [int(width * 0.85), int(height * 0.95)]
    ])
    dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped, Minv

def _threshold_image(img):
    """(Internal) Isolates lane pixels using color and gradient thresholds."""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def _find_lanes_from_scratch(binary_warped):
    """(Internal) Finds lane pixels from scratch using a histogram and sliding windows."""
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int64(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    margin, minpix = 100, 50
    window_height = np.int64(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix: leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError: pass
    
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def _find_lanes_from_prior(binary_warped, left_fit, right_fit):
    """
    (Internal) Finds lane pixels by searching in a narrow margin around the
    previously detected lane lines. This is much faster and more stable.
    """
    margin = 100
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def _fit_polynomial(leftx, lefty, rightx, righty):
    """(Internal) Fits a curve to the lane pixels."""
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit
    except TypeError:
        return None, None

def _draw_lane_area(undist, warped, Minv, left_fit, right_fit):
    """(Internal) Draws the detected lane area back onto the original image."""
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

#main function that will be called
def process_frame(frame, left_line, right_line):
    warped_img, Minv = _warp_image(frame)
    binary_img = _threshold_image(warped_img)

    if left_line.detected and right_line.detected:
        leftx, lefty, rightx, righty = _find_lanes_from_prior(binary_img, left_line.current_fit, right_line.current_fit)
    else:
        leftx, lefty, rightx, righty = _find_lanes_from_scratch(binary_img)

    left_fit, right_fit = _fit_polynomial(leftx, lefty, rightx, righty)

    left_line.update(left_fit)
    right_line.update(right_fit)


    if left_line.best_fit is not None and right_line.best_fit is not None:
        result = _draw_lane_area(frame, binary_img, Minv, left_line.best_fit, right_line.best_fit)
        cv2.putText(result, "Lane Lock Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        result = frame
        cv2.putText(result, "Searching for lanes...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return result