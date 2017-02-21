
import numpy as np
import cv2

def absolute_sobel_threshold(image, orient='x', sobel_kernel=3, threshold=(0, 255)):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  binary_orient = (1, 0)
  if (orient == 'y'):
    binary_orient = (0, 1)
  sobel = cv2.Sobel(gray, cv2.CV_64F, *binary_orient, ksize=sobel_kernel)
  abs_sobel = np.absolute(sobel)
  scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
  return binary_output


def magnitude_sobel_threshold(image, sobel_kernel=3, threshold=(0, 255)):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  mag = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
  scaled_mag = np.uint8(255 * mag / np.max(mag))
  binary_output = np.zeros_like(scaled_mag)
  binary_output[(scaled_mag >= threshold[0]) & (scaled_mag <= threshold[1])] = 1
  return binary_output


def direction_sobel_threshold(image, sobel_kernel=3, threshold=(0, np.pi / 2)):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  abs_sobel_x = np.absolute(sobel_x)
  abs_sobel_y = np.absolute(sobel_y)
  direction = np.arctan2(abs_sobel_y, abs_sobel_x)
  binary_output = np.zeros_like(direction)
  binary_output[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
  return binary_output

def saturation_threshold(image, threshold=(0, 255)):
  img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  S = img_hls[:, :, 2]
  binary_output = np.zeros_like(S)
  binary_output[(S > threshold[0]) & (S <= threshold[1])] = 1
  return binary_output

def white_and_yellow(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

  white_image = np.ones((image.shape[0], image.shape[1], 1), np.uint8)
  yellow_mask = cv2.inRange(hsv, np.array([20,90,90]), np.array([30,255,255]))
  white_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([255,30,255]))

  w_image = cv2.bitwise_and(white_image, white_image, mask=white_mask)
  y_image = cv2.bitwise_and(white_image, white_image, mask=yellow_mask)

  result = cv2.bitwise_or(y_image, w_image)
  return result


def birds_eye_matrix_road(shape, inverse):
  width = shape[1]
  height = shape[0]

  sll = [width * 0.4, height * 0.63]
  slr = [width * 0.6, height * 0.63]
  sul = [width * 0.15, height * 0.95]
  sur = [width * 0.85, height * 0.95]

  dll = [width * 0.15, 0]
  dlr = [width * 0.85, 0]
  dul = [width * 0.15, height]
  dur = [width * 0.85, height]

  src = np.float32([sll, slr, sur, sul])
  dst = np.float32([dll, dlr, dur, dul])

  if inverse == True:
    M = cv2.getPerspectiveTransform(dst, src)
  else:
    M = cv2.getPerspectiveTransform(src, dst)
  return M
