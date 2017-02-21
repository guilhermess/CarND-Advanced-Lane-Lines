
from LaneLinePipeline import LaneLinePipeline
from moviepy.editor import VideoFileClip
import numpy as np
import argparse
import glob
import cv2
import imageio
from LoggerCV import LoggerCV

def video_processing(input_video_file, start, end, output_video_file, lane_line_pipeline):
  input_video = VideoFileClip(input_video_file).subclip(start, end)
  output_video = input_video.fl_image(lane_line_pipeline.process_frame)
  output_video.write_videofile(output_video_file, audio=False)


def advanced_lane_lines(input_video_file,
                        start,
                        end,
                        output_video_file,
                        calibration_images,
                        calibration_nx,
                        calibration_ny,
                        logger):
  lane_line_pipeline = LaneLinePipeline(calibration_images, calibration_nx, calibration_ny, logger)
  video_processing(input_video_file, start, end, output_video_file, lane_line_pipeline)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Self-Driving Car Advanced Lane Lines')
  parser.add_argument('-input_video', help='Path to video file to process', action='store')
  parser.add_argument('-start', help='Process subclip of input video', action='store', default=0/25.0)
  parser.add_argument('-end', help='Process subclip of input video', action='store', default=None)
  parser.add_argument('-output_video', help='Output video file', action='store', default='output.mp4')
  parser.add_argument('-calibration_images', help='Directory containing camera calibration images', action='store', default='./camera_cal/*.jpg')
  parser.add_argument('-calibration_nx', help='Number of x vertices in calibration images chess board', action='store', default=9)
  parser.add_argument('-calibration_ny', help='Number of y vertices in calibration images chess board', action='store', default=6)
  parser.add_argument('-log_enabled', help='Path to file to store the log', action='store', default=True)
  parser.add_argument('-log_dir', help='Path to file to store the log', action='store', default='./log')
  parser.add_argument('-log_rate', help='Every % frames store image in log. Valid only if log is enabled', action='store', default=25)
  args = parser.parse_args()

  logger = LoggerCV(args.log_enabled, args.log_rate)

  calibration_images = glob.glob(args.calibration_images)

  advanced_lane_lines(args.input_video, args.start, args.end,
                      args.output_video, calibration_images,
                      args.calibration_nx, args.calibration_ny, logger)

  if ( args.log_enabled == True ):
    logger.write_records(args.log_dir)
