import cv2
from tqdm import tqdm
import numpy as np
import os
import subprocess

class MergeJPGVideo():
    @staticmethod
    def frames_to_video(frames_dir, output_dir, fps):
        frame_array = []
        files = [file for file in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, file))]

        # pre-sort files
        files.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))

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
