
import numpy as np
import cv2
import glob

def calibrate(image_files, nx, ny, debug=False):
  objp = np.zeros((ny * nx, 3), np.float32)
  objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

  # Arrays to store object points and image points from all the images.
  objpoints = []  # 3d points in real world space
  imgpoints = []  # 2d points in image plane.

  first_img = True
  img_size = (0,0)

  for _, img_file in enumerate(image_files):
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if first_img:
      img_size = (img.shape[1], img.shape[0])
      first_img = False

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

    if debug:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        cv2.imshow(img_file, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

  _, camera_matrix, distortion_coeff, _, _= cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
  return camera_matrix, distortion_coeff


if __name__ == '__main__':
  images = glob.glob('camera_cal/calibration*.jpg')
  camera_matrix, distortion_coeff = calibrate(images, 9, 6, False)

  test_images = ['camera_cal/calibration1.jpg',
                 'camera_cal/calibration4.jpg',
                 'camera_cal/calibration5.jpg']
  for test_img in test_images:
    img = cv2.imread(test_img)
    undistorted_img = cv2.undistort(img,camera_matrix, distortion_coeff, None, camera_matrix)
    cv2.imshow(test_img, undistorted_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
