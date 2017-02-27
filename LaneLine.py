
import numpy as np

class LaneLine:
  '''
  This class stores the x,y points of a lane line and provides methods to get the lane fit and radius of curvature
  '''

  ym_per_pix = 30 / 720  # meters per pixel in y dimension
  xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

  def __init__(self, x, y, height, width):
    self.x = x
    self.y = y
    self.height = height
    self.width = width
    self.__lane_fit = None
    self.__radius_of_curvature = None


  def lane_fit(self):
    if self.__lane_fit is None:
      self.__lane_fit = np.polyfit(self.y, self.x, 2)
    return self.__lane_fit


  def radius_of_curvature(self):
    if self.__radius_of_curvature is None:
      lane_fit = np.polyfit(self.y * LaneLine.ym_per_pix, self.x * LaneLine.xm_per_pix, 2)
      self.__radius_of_curvature = ((1 + (2 * lane_fit[0] * self.height * LaneLine.ym_per_pix + lane_fit[1]) ** 2) ** 1.5) / \
                                   np.absolute(2*lane_fit[0])
    return self.__radius_of_curvature

