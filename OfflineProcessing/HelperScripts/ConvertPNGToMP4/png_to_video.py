import cv2 
from tqdm import tqdm
import numpy as np 
import os
import subprocess

class PNGTOVideo():
    @staticmethod
    def frames_to_video(frames_dir, output_dir, fps):
        frame_array = []
        files = [file for file in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, file))]

        # pre-sort files
        files.sort(key = lambda x: int(x.split("-")[1][0:-4]))

        size = None

        for file in tqdm(files, desc="reading frames", ncols=100):
            filename = os.path.join(frames_dir, file)
            #read in each frame
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            #insert frames into image array
            frame_array.append(img)
        print("Frame size: ", size)
        out = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc(*'H264'), fps, size)
        
        for frame in tqdm(frame_array, desc="writing video file", ncols=100):
            out.write(frame)
        out.release()

    @staticmethod
    def multiplex_video_audio(video_file, audio_file, output_file):
        if not os.path.isfile(video_file):
            print("Invalid video path")
            exit(-1)
        
        if not os.path.isfile(audio_file):
            print("Invalid audio path")
            exit(-1)

        cmd = ["ffmpeg", "-i", video_file, "-i", audio_file, "-c:v", "copy", "-c:a", "aac", output_file]
        subprocess.call(cmd)