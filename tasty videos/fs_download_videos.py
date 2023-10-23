# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:59:37 2019

@author: fame
"""

import os
import wget
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import xml.etree.cElementTree as etree
import subprocess


def convert_video(filename, targetname):
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
      "-i", filename,
      "-vcodec", "copy", "-acodec", "copy", targetname]
    subprocess_call(cmd)

def ffmpeg_extract_subclip_ex(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [get_setting("FFMPEG_BINARY"),"-y",
          "-i", filename,
           "-ss", "%0.2f"%t1,
           "-t", "%0.2f"%(t2-t1),
           "-vcodec", "copy", "-acodec", "copy", targetname]

    subprocess_call(cmd)

def get_video_info(fileloc) :
    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries', 'stream=width,height,r_frame_rate,duration',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               fileloc, '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE,
                              stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = str(out.decode()).split('\n')
    return float(out[4].split('/')[0])/float(out[4].split('/')[1])



if __name__ == '__main__':

    recipe_names_fid = open("ALL_RECIPES.txt", "r")
    recipe_names = recipe_names_fid.readlines()
    recipe_names_fid.close()

    path_2recipes = 'ALL_RECIPES_without_videos/'

    for indu in  range(len(recipe_names)) :
        curr_recipe = recipe_names[indu].replace('\n', '')
        print('video ', curr_recipe)

        curr_file_path = path_2recipes + curr_recipe
        xml_file_path = curr_file_path + '/recipe.xml'
        xmlDoc = open(xml_file_path, 'r')
        xmlDocData = xmlDoc.read()
        xmlDoc.close()

        xmlDocTree = etree.XML(xmlDocData)

        type_video = xmlDocTree.find('type').text

        if type_video == 'single':

            url = xmlDocTree.find('url').text

            videos_high = xmlDocTree.find('video')
            video_link_src = videos_high[1].text
            file_name = wget.download(video_link_src, curr_file_path + "/")

            convert_video(file_name, targetname = (curr_file_path +\
                                                   "/recipe_video.mp4") )
            os.remove(file_name)

        else:
            url = xmlDocTree.find('url').text

            videos_high = xmlDocTree.find('video')
            video_link_src = videos_high[1].text

            file_name = wget.download(video_link_src, curr_file_path + "/")

            duration_human = xmlDocTree.find('duration_human')
            if not duration_human is None:
                start_time =  abs( float( duration_human[0].text) )
                end_time = abs( float( duration_human[1].text) )
                fps_vid = get_video_info(file_name)
                start_time =  float(start_time)/float(fps_vid)
                end_time =  float(end_time)/float(fps_vid)
            else:
                duration_aut = xmlDocTree.find('duration')
                start_time =  abs( float( duration_aut[0].text) )
                end_time   =  abs( float( duration_aut[1].text) )


            ffmpeg_extract_subclip_ex(file_name, start_time, end_time, \
                                      targetname = (curr_file_path +\
                                                    "/recipe_video.mp4") )
            os.remove(file_name)

