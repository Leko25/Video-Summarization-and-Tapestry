import cv2
import os

class ExtractFrames():
    def __init__(self, video_path, extracted_frame_dir):
        self.video_path_dir = video_path
        self.extracted_frame_dir = extracted_frame_dir

    def extractor(self):
        '''
        Extract individual frames from mp4 video and write to specified directory
        '''
        video_array = []
        video_array = [file for file in os.listdir(self.video_path_dir) if os.path.isfile(os.path.join(self.video_path_dir, file))]
        video_array.sort(key = lambda x: int(x.split("_")[1][0:-4]))
        for x in video_array:
            print(x)
        # for video_path,video_no in enumerate(video_array,len(video_array)):
        video_no=1
        for video_path in video_array:
            video=os.path.join(self.video_path_dir, video_path)
            assert os.path.isfile(video), "Invalid video path"

            if not os.path.isdir(self.extracted_frame_dir):
                os.mkdir(self.extracted_frame_dir)
                print("Created directory " + self.extracted_frame_dir)

            cap = cv2.VideoCapture(video)
            count = 0
            while(cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break
                cv2.imwrite(os.path.join(self.extracted_frame_dir, "frame_"+str(video_no)+"_" + str(count) + ".jpg"), frame)
                print("extracting frame " + str(video_no)+"_"+str(count) + "...")
                count += 1
            cap.release()
            video_no=video_no+1
