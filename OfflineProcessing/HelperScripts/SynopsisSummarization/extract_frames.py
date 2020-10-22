import cv2
import os

class ExtractFrames():
    def __init__(self, video_path, extracted_frame_dir):
        self.video_path = video_path
        self.extracted_frame_dir = extracted_frame_dir

    def extractor(self):
        '''
        Extract individual frames from mp4 video and write to specified directory
        '''
        assert os.path.isfile(self.video_path), "Invalid video path"

        if not os.path.isdir(self.extracted_frame_dir):
            os.mkdir(self.extracted_frame_dir)
            print("Created directory " + self.extracted_frame_dir)

        cap = cv2.VideoCapture(self.video_path)
        count = 0
        while(cap.isOpened()):
            success, frame = cap.read()
            if not success:
                break
            cv2.imwrite(os.path.join(self.extracted_frame_dir, "frame_" + str(count) + ".jpg"), frame)
            print("extracting frame " + str(count) + "...")
            count += 1
        cap.release()
