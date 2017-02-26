
import os.path
from collections import namedtuple
import cv2

CVRecord = namedtuple('CVRecord', ['name', 'frame', 'figures', 'imtype'])

class LoggerCV:
  def __init__(self, enabled, log_rate):
    self.__enabled = enabled
    self.log_rate = log_rate
    self.records = []

  def enabled(self, frame_count):
    return self.__enabled and frame_count % self.log_rate == 0


  def log_rate(self):
    return self.log_rate

  def log(self, record):
    self.records.append(record)


  def write_records(self, output_dir):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    for record in self.records:
      basename = output_dir + '/' + record.name + 'f' + str(record.frame)
      count = 0
      for image in record.figures:
        filename = basename + 'c' + str(count) + '.png'
        if ( record.imtype == 'figure'):
          image.savefig(filename)
        elif (record.imtype == 'opencv'):
          if ( len(image.shape) == 3):
            cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
          else:
            cv2.imwrite(filename, image*255)






