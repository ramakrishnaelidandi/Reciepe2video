# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 01:00:17 2019

@author: fame
"""

import os
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import cv2

def convert_video(filename, targetname):
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
      "-i", filename,
      "-vcodec", "copy", "-acodec", "copy", targetname]
    subprocess_call(cmd)

if __name__ == '__main__':

    recipe_names_fid = open("ALL_RECIPES.txt", "r")
    recipe_names = recipe_names_fid.readlines()
    recipe_names_fid.close()

    path_2recipes = 'ALL_RECIPES_without_videos/'

    for indu in  range(len(recipe_names)) :
        curr_recipe = recipe_names[indu].replace('\n', '')
        print('video ', curr_recipe)

        curr_file_path = path_2recipes + curr_recipe
        video_file_path = curr_file_path + '/recipe_video.mp4'

        frames_path = curr_file_path + '/frames/'
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)

        vidcap = cv2.VideoCapture(video_file_path)
        success,image = vidcap.read()
        count = 0
        while success:
          image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
          cv2.imwrite(frames_path + "{:05d}".format(count) + '.jpg', image)
          success,image = vidcap.read()
          count += 1
