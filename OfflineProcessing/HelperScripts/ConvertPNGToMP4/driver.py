from png_to_video import PNGTOVideo
import sys
import os
import shutil

def main(frames_dir,audio_file,output_file):
     if os.path.isdir("./temp"):
     	shutil.rmtree("./temp")
     os.mkdir("./temp")
     out_file = "./temp/video.mp4"
     fps = 29.96
     PNGTOVideo.frames_to_video(frames_dir, out_file, fps)

     PNGTOVideo.multiplex_video_audio(out_file, audio_file, output_file)

if __name__ == "__main__":
	main(sys.argv[1],sys.argv[2],sys.argv[3])