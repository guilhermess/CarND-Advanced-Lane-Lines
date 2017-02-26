
from collections import namedtuple
import matplotlib.pyplot as plt
import camera_calibration as cc
import frame_processing as fp
import numpy as np
from LoggerCV import LoggerCV
from LoggerCV import CVRecord
from LaneLine import LaneLine

import cv2

class LaneLinePipeline:
  '''
  The LaneLinePipeline class implements the framework for processing video images, detecting lane lines and
  marking them on the road image.
  '''

  def __init__(self, calibration_images,
               calibration_nx, calibration_ny,
               logger):
    self.camera_matrix, self.distortion_coeff = cc.calibrate(calibration_images,
                                                             calibration_nx,
                                                             calibration_ny)
    self.logger = logger
    self.frame_count = 0
    self.left_fit = None
    self.right_fit = None
    self.margin = 100
    self.nwindows = 9
    self.left_lane_lines = []
    self.right_lane_lines = []
    self.previous_lane_count = 0


  def process_frame(self, image):
    '''
    This method received a frame image from a video and marks the road lane. The following steps are involved in the
    lane detection pipeline:
    1) Undistort camera image.
    2) Compute threshold images using sobel, saturation and color (white and yellow) filters/masks.
    3) Perform a perspective transform to analyze the lanes in the warped birds eye view.
    4) Detect lane lines in the warped image.
    5) Mark lane lines.
    :param image: road image to detect lane lines
    :return: output image containing detected lane lines
    '''
    # step 1: undistort image
    undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeff, None, self.camera_matrix)

    if self.logger.enabled(self.frame_count):
      self.logger.log(CVRecord("undistorted", self.frame_count, [undistorted_image], 'opencv'))

    # step 2: compute threshold image
    threshold_images = self.threshold_combinations(undistorted_image)

    found_left_lane = False
    found_right_lane = False

    for threshold_image in threshold_images:
      # step 3: perspective transform
      M = fp.birds_eye_matrix_road(shape=threshold_image.shape, inverse=False)
      Minv = fp.birds_eye_matrix_road(shape=threshold_image.shape, inverse=True)
      warped = cv2.warpPerspective(threshold_image, M, (threshold_image.shape[1], threshold_image.shape[0]))

      if self.logger.enabled(self.frame_count):
        self.logger.log(CVRecord("warped_threshold", self.frame_count, [warped], 'opencv'))

      # step 4: find lanes in warped image
      if not found_left_lane:
        if (len(self.left_lane_lines) > 0):
          previous_lane_line = self.left_lane_lines[-1]
          left_xy = self.get_iterative_lanes_xy(warped, previous_lane_line.lane_fit(), 'left' )
        else:
          left_xy = self.get_histogram_lanes_xy(warped, 'left')

      if not found_right_lane:
        if (len(self.right_lane_lines) > 0):
          previous_lane_line = self.right_lane_lines[-1]
          right_xy = self.get_iterative_lanes_xy(warped, previous_lane_line.lane_fit(), 'right')
        else:
          right_xy = self.get_histogram_lanes_xy(warped, 'right')

      left_x = left_xy[0]
      left_y = left_xy[1]
      if len(left_x) > 0:
        candidate_lane_left = LaneLine(left_x, left_y, threshold_image.shape[0], threshold_image.shape[1])
        if self.accept(candidate_lane_left, self.left_lane_lines, threshold_image.shape):
          found_left_lane = True

      right_x = right_xy[0]
      right_y = right_xy[1]
      if len(right_x) > 0:
        candidate_lane_right = LaneLine(right_x, right_y, threshold_image.shape[0], threshold_image.shape[1])
        if self.accept(candidate_lane_right, self.right_lane_lines, threshold_image.shape):
          found_right_lane = True


      if found_left_lane and found_right_lane:
        if self.accept_left_right(candidate_lane_left, candidate_lane_right, undistorted_image.shape):
          color_warped = cv2.warpPerspective(undistorted_image, M, (undistorted_image.shape[1], undistorted_image.shape[0]))
          output_img = self.mark_lanes(undistorted_image, Minv, color_warped, candidate_lane_left, candidate_lane_right)
          self.left_lane_lines.append(candidate_lane_left)
          self.right_lane_lines.append(candidate_lane_right)
          self.frame_count += 1
          self.previous_lane_count = 0
          return output_img
        else:
          found_left_lane = False
          found_right_lane = False

      if not found_left_lane:
          left_xy = self.get_histogram_lanes_xy(warped, 'left')
          left_x = left_xy[0]
          left_y = left_xy[1]
          if len(left_x) > 0:
            candidate_lane_left = LaneLine(left_x, left_y, threshold_image.shape[0], threshold_image.shape[1])
            if self.accept(candidate_lane_left, self.left_lane_lines, threshold_image.shape):
              found_left_lane = True

      if not found_right_lane:
        right_xy = self.get_histogram_lanes_xy(warped, 'right')
        right_x = right_xy[0]
        right_y = right_xy[1]
        if len(right_x) > 0:
          candidate_lane_right = LaneLine(right_x, right_y, threshold_image.shape[0], threshold_image.shape[1])
          if self.accept(candidate_lane_right, self.right_lane_lines, threshold_image.shape):
            found_right_lane = True

      if found_left_lane and found_right_lane:
        if self.accept_left_right(candidate_lane_left, candidate_lane_right, undistorted_image.shape):
          color_warped = cv2.warpPerspective(undistorted_image, M, (undistorted_image.shape[1], undistorted_image.shape[0]))
          output_img = self.mark_lanes(undistorted_image, Minv, color_warped, candidate_lane_left, candidate_lane_right)
          self.left_lane_lines.append(candidate_lane_left)
          self.right_lane_lines.append(candidate_lane_right)
          self.frame_count += 1
          self.previous_lane_count = 0
          return output_img
        else:
          found_left_lane = False
          found_right_lane = False

    if ( len(self.left_lane_lines) and len(self.right_lane_lines) ):
      self.previous_lane_count += 1
      color_warped = cv2.warpPerspective(undistorted_image, M,
                                        (undistorted_image.shape[1], undistorted_image.shape[0]))
      candidate_lane_left = self.left_lane_lines[-1]
      candidate_lane_right = self.right_lane_lines[-1]
      output_img = self.mark_lanes(undistorted_image, Minv, color_warped, candidate_lane_left, candidate_lane_right)
      self.frame_count += 1
      return output_img

    self.frame_count += 1
    self.previous_lane_count += 1
    return undistorted_image


  def mark_lanes(self, image, Minv, warped_image, left_lane, right_lane ):
    '''
    Mark lane lines and warp it back from birds eye view to initial view.
    :param image: image to apply the lane mark
    :param Minv: inverse warp matrix to transform back from birds eye view.
    :param warped_image: warped color image to debug the mark lanes
    :param left_lane: left lane fit
    :param right_lane: right lane fit
    :return: image with marked lane lines
    '''
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fit = left_lane.lane_fit()
    right_fit = right_lane.lane_fit()
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    color_warp= np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    left_radius = left_lane.radius_of_curvature()
    right_radius = right_lane.radius_of_curvature()
    offset = self.lane_offset(left_lane, right_lane, image.shape[0], image.shape[1]/2)

    cv2.putText(result, "Left radius of curvature: {0:.2f}m".format(left_radius), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 250, 200))
    cv2.putText(result, "Right radius of curvature: {0:.2f}m".format(right_radius), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 250, 200))
    cv2.putText(result, "Offset from lane center: {0:.2f}m".format(offset), (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 250, 200))

    if self.logger.enabled(self.frame_count):
      warped_lane = cv2.addWeighted(warped_image, 1, color_warp, 0.3, 0)
      self.logger.log(CVRecord("warped", self.frame_count, [warped_lane], 'opencv'))

    return result


  def lane_offset(self, left_lane, right_lane, ymax, xcenter):
    '''
    Given a left lane and right lane this method returns the offset of the camera from the center
    of the detected lanes
    :param left_lane: left lane fit
    :param right_lane: right lane fit
    :param ymax: max y coordinate (position of the camera in the car)
    :param xcenter: center of the image
    :return: car offset from lane center
    '''
    left_poly = left_lane.lane_fit()
    right_poly = right_lane.lane_fit()

    left_x_ymax = left_poly[0] * ymax ** 2 + left_poly[1] * ymax + left_poly[2]
    right_x_ymax = right_poly[0] * ymax ** 2 + right_poly[1] * ymax + right_poly[2]

    x_lane_center = (left_x_ymax + right_x_ymax) / 2
    offset = (xcenter - x_lane_center) * LaneLine.xm_per_pix
    return offset


  def threshold_combinations(self, image):
    '''
    Return threshold combination images candidates for detecting lane lines.
    Uses white and yellow masks, saturation, Sobel-X, Sobel-Y, Sobel-Magnitude and Sobel-Direction combined to
    create the threshold image.
    :param image: original image
    :return: threshold images
    '''
    white_yellow_image = fp.region_of_interest(fp.white_and_yellow(image))
    sobel_x = fp.region_of_interest(fp.absolute_sobel_threshold(image, 'x', 11, (30, 100)))
    sobel_y = fp.region_of_interest(fp.absolute_sobel_threshold(image, 'y', 11, (30, 100)))
    sobel_magnitude = fp.region_of_interest(fp.magnitude_sobel_threshold(image, 11, (30, 100)))
    sobel_direction = fp.region_of_interest(fp.direction_sobel_threshold(image, 11, (0.7, 1.2)))
    saturation = fp.region_of_interest(fp.saturation_threshold(image, (120, 255)))

    combined1 = np.zeros_like(sobel_magnitude)
    combined2 = np.zeros_like(sobel_magnitude)
    combined3 = np.zeros_like(sobel_magnitude)

    combined1[(white_yellow_image == 1) & ( (saturation==1) | (sobel_x==1) | (sobel_magnitude==1))] = 1
    combined2[(sobel_magnitude==1) & (sobel_direction==1)] = 1
    combined3[(sobel_magnitude == 1) & (sobel_direction == 1) & (saturation==1)] = 1

    self.debug_threshold_combination(image, sobel_x, sobel_y, sobel_magnitude,
                                     sobel_direction, saturation, white_yellow_image, combined1)

    return combined1, combined2, combined3


  def get_histogram_lanes_xy(self, binary_warped, side):
    '''
    Get the X and Y points in the warped image that are part of a lane line for a given lane line with
     side 'left' or 'right'. This method uses histogram and windows to detect points belonging to the lanes
    :param binary_warped: input warped image to detect lane points
    :param side: 'left' or 'right' lane
    :return: X and Y points that are detected as part of the lane line.
    '''
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)

    if side == "left":
      x_base = np.argmax(histogram[:midpoint])
    else:
      x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = self.nwindows

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    x_current = x_base

    # Set the width of the windows +/- margin
    margin = self.margin

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    if self.logger.enabled(self.frame_count):
      out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Step through the windows one by one
    for window in range(nwindows):
      win_y_low = binary_warped.shape[0] - (window + 1) * window_height
      win_y_high = binary_warped.shape[0] - window * window_height
      win_x_low = x_current - margin
      win_x_high = x_current + margin

      if self.logger.enabled(self.frame_count):
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

      # Identify the nonzero pixels in x and y within the window
      good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                   (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

      # Append these indices to the lists
      lane_inds.append(good_inds)

      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_inds) > minpix:
        x_current = np.int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    if self.logger.enabled(self.frame_count):
      if ( len(x) > 0 and len(y) > 0 ):
        poly_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        fitx = poly_fit[0] * ploty ** 2 + poly_fit[1] * ploty + poly_fit[2]

        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
        figure = plt.figure()
        plt.imshow(out_img)
        plt.plot(fitx, ploty, color='yellow')
        plt.close()
        self.logger.log(CVRecord("histogram_lane_fit_" + side, self.frame_count, [figure], 'figure'))

    return x, y


  def get_iterative_lanes_xy(self, binary_warped, lane_fit, lane_side):
    '''
    Get the X and Y points in the warped image that are part of a lane line for a given lane line with
    side 'left' or 'right'. This method uses a previously detected lane as input to detecting lane lines
    in the current frame.
    :param binary_warped: input warped image to detect lane points
    :param lane_fit: previous frame lane fit
    :param lane_side: 'left' or 'right' lane side
    :return: X and Y points that are detected as part of the lane line.
    '''
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = self.margin
    
    lane_inds = ((nonzerox > (lane_fit[0] * (nonzeroy ** 2) +
                              lane_fit[1] * nonzeroy +
                              lane_fit[2] - margin)) &
                 (nonzerox < (lane_fit[0] * (nonzeroy ** 2) +
                              lane_fit[1] * nonzeroy +
                              lane_fit[2] + margin)))

    lane_x = nonzerox[lane_inds]
    lane_y = nonzeroy[lane_inds]

    self.debug_iterative_lanes_xy(binary_warped,
                                 nonzerox, nonzeroy,
                                 lane_inds, lane_x, lane_y, lane_side)

    return (lane_x, lane_y)

  def accept(self, candidate, previous, shape):
    '''
    This method returns True if the candidate lane line fit is reasonable close to the previous detected lane.
    :param candidate: candidate lane line
    :param previous: last detected lane line
    :param shape: image shape
    :return: True if the candidate lane is accept, False otherwise.
    '''
    if (len(previous) == 0 or self.previous_lane_count > 10):
      return True;

    candidate_fit = candidate.lane_fit()
    previous_fit = previous[-1].lane_fit()

    ysize = shape[0]
    ymin = 0.63 * ysize
    ymax = 0.98 * ysize
    yc = (ymin + ymax) / 2

    xmin_candidate = candidate_fit[0] * ymin ** 2 + candidate_fit[1] * ymin + candidate_fit[2]
    xc_candidate = candidate_fit[0] * yc ** 2 + candidate_fit[1] * yc + candidate_fit[2]
    xmax_candidate = candidate_fit[0] * ymax ** 2 + candidate_fit[1] * ymax + candidate_fit[2]

    xmin_previous = previous_fit[0] * ymin ** 2 + previous_fit[1] * ymin + previous_fit[2]
    xc_previous = previous_fit[0] * yc ** 2 + previous_fit[1] * yc + previous_fit[2]
    xmax_previous = previous_fit[0] * ymax ** 2 + previous_fit[1] * ymax + previous_fit[2]

    diff_min = np.abs(xmin_candidate - xmin_previous)
    diff_center = np.abs(xc_candidate - xc_previous)
    diff_max = np.abs(xmax_candidate - xmax_previous)

    if diff_min > 30 or diff_center > 30 or diff_max > 30:
      return False

    return True


  def accept_left_right(self, candidate_left, candidate_right, shape):
    '''
    This method returns True if the candidates for left and right lanes have close parameters, like radius of curvature
     and also if the candidate lines don't cross. Otherwise it returns False.
    :param candidate_left: candidate to left lane
    :param candidate_right: candidate to right lane
    :param shape: image shape
    :return: True if lanes are accepted, False otherwise.
    '''

    left_fit = candidate_left.lane_fit()
    right_fit = candidate_right.lane_fit()

    ymin = 0
    ymax = shape[0]
    yc = (ymin + ymax) / 2
    offset = self.lane_offset(candidate_left, candidate_right, ymax, shape[1] / 2)

    xmin_left = left_fit[0] * ymin ** 2 + left_fit[1] * ymin + left_fit[2]
    xc_left = left_fit[0] * yc ** 2 + left_fit[1] * yc + left_fit[2]
    xmax_left = left_fit[0] * ymax ** 2 + left_fit[1] * ymax + left_fit[2]

    xmin_right = right_fit[0] * ymin ** 2 + right_fit[1] * ymin + right_fit[2]
    xc_right = right_fit[0] * yc ** 2 + right_fit[1] * yc + right_fit[2]
    xmax_right = right_fit[0] * ymax ** 2 + right_fit[1] * ymax + right_fit[2]

    radius_left = candidate_left.radius_of_curvature()
    radius_right = candidate_right.radius_of_curvature();

    if xmin_left > xmin_right or xc_left > xc_right or xmax_left > xmax_right or \
            np.abs(radius_left - radius_right) > 2 * min(radius_left, radius_right) or\
            np.abs(offset) > 1:
      return False
    return True


  def debug_threshold_combination(self, image, sobel_x, sobel_y,
                                  sobel_magnitude, sobel_direction,
                                  saturation, white_yellow_image,
                                  combined):
    '''
    Debug method to log threshold images.
    '''
    if self.logger.enabled(self.frame_count):
      figure = plt.figure(figsize=(18,14))

      plt.subplot('421')
      plt.imshow(image)
      plt.title("Original")

      plt.subplot('422')
      plt.imshow(sobel_x, cmap='gray')
      plt.title("Sobel X")

      plt.subplot('423')
      plt.imshow(sobel_y, cmap='gray')
      plt.title("Sobel Y")

      plt.subplot('424')
      plt.imshow(sobel_magnitude, cmap='gray')
      plt.title("Sobel Magnitude")

      plt.subplot('425')
      plt.imshow(sobel_direction, cmap='gray')
      plt.title("Sobel Direction")

      plt.subplot('426')
      plt.imshow(saturation, cmap='gray')
      plt.title("Saturation")

      plt.subplot('427')
      plt.imshow(white_yellow_image, cmap='gray')
      plt.title("White and Yellow")

      plt.subplot('428')
      plt.imshow(combined, cmap='gray')
      plt.title("Combined")

      plt.close()

      self.logger.log(CVRecord("threshold", self.frame_count, [figure], 'figure'))


  def debug_iterative_lanes_xy(self, binary_warped, nonzerox, nonzeroy,
                               lane_inds, x, y, lane_side):
    '''
    Debug method to log iterative lane detection images.
    '''
    if self.logger.enabled(self.frame_count):
      if ( len(x) > 0 and len(y) > 0 and len(x) > 0 and len(x) > 0 ):
        fit = np.polyfit(y, x, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]

        margin = self.margin

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([fitx - margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        figure = plt.figure()
        plt.imshow(result)
        plt.plot(fitx, ploty, color='yellow')
        plt.close()
        self.logger.log(CVRecord("iterative_lane_fit_" + lane_side, self.frame_count, [figure], 'figure'))



    