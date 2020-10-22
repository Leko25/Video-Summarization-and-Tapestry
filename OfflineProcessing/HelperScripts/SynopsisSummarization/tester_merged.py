from extract_frames_from_dir import ExtractFrames
from frame_to_embeddings import FrameToEmbeddings
from cluster_embeddings import ClusterEmbedding
from cluster_window import ClusterWindow
from merge_jpg_video import MergeJPGVideo
import os
import sys

def driver(video_file_dir, extraction_dir, dir_prefix, merge_file_dir):
    '''
    @param video_file: file path to video
    @param extraction_dir: name of directory to save extracted frames
    @param dir_prefix: prefix to concat further directories
    @param merge_file_dir: directory to merged file
    '''
    extract = ExtractFrames(video_file_dir, extraction_dir)
    extract.extractor()

    embeddings = FrameToEmbeddings(extraction_dir, dir_prefix + "_embedding")
    embeddings.extract_embeddings()

    cluster_embedding = ClusterEmbedding(
                    dir_prefix + "_cluster_image",
                    dir_prefix + "_embedding",
                    "HDBSCAN",
                    dir_prefix + "_cluster_file",
                    n_clusters=10
                )
    cluster_embedding.cluster()

    window = ClusterWindow(dir_prefix + "_synopsis_file",
            dir_prefix + "_cluster_file", 1, dir_prefix + "_embedding")
    window.get_sequence()

    MergeJPGVideo.frames_to_video(dir_prefix + "_synopsis_file", os.path.join(merge_file_dir, "merge_vid.mp4"), 15)

if __name__ == "__main__":
    driver(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
