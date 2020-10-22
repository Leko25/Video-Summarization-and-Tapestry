import os
import pickle
import numpy as np
import itertools
import shutil

class ClusterWindow():
    def __init__(self, synopsis_dir, cluster_dir, window_size, embedding_dir):
        self.synopsis_dir = synopsis_dir
        self.cluster_dir = cluster_dir
        self.window_size = window_size
        self.embedding_dir = embedding_dir

    def map_label_to_frame(self, cluster_dict):
        unique_labels = list(set(label for label in cluster_dict.values()))
        label_to_frame = {}
        for unique_label in unique_labels:
            label_to_frame[unique_label] = []

        for frame_name, label in cluster_dict.items():
            label_to_frame[label].append(frame_name)

        return label_to_frame

    def get_embeddings(self):
        embed_dict = pickle.load(open(os.path.join(self.embedding_dir, 'frame_embeddings.pkl'), 'rb'))
        return embed_dict

    def compute_cluster_centroid(self, label_to_frame, embed_dict):
        label_to_centroid = {}
        for label, frame_name_list in label_to_frame.items():
            if label == -1:
                continue
            label_embedding = np.vstack([embed_dict[frame_name] for frame_name in frame_name_list])
            label_to_centroid[label] = np.mean(label_embedding, axis=0)
        return label_to_centroid

    def compute_error(self, label_to_centroid, label_to_frame, embed_dict):
        label_to_error = {}

        for label, frame_name_list in label_to_frame.items():
            if label == -1:
                continue
            centroid = label_to_centroid[label]
            label_to_error[label] = [np.linalg.norm(np.array(embed_dict[frame_name]) - centroid) for frame_name in frame_name_list]
        return label_to_error

    def get_cluster_sequence(self, cluster_dict):
        label_to_frame = self.map_label_to_frame(cluster_dict)
        embed_dict = self.get_embeddings()
        label_to_centroid = self.compute_cluster_centroid(label_to_frame, embed_dict)
        label_to_error = self.compute_error(label_to_centroid, label_to_frame, embed_dict)

        # Get k smallest errors from centroid
        min_samples = float('inf')
        for frame_name_list in label_to_frame.values():
            if len(frame_name_list) < min_samples:
                min_samples = len(frame_name_list)

        min_samples = min(self.window_size, min_samples)

        label_to_k_smallest = {}
        for label, error_list in label_to_error.items():
            if label == -1:
                continue
            idx = np.argpartition(error_list, min_samples)
            k_smallest_idx = idx[:min_samples]
            label_to_k_smallest[label] = [label_to_frame[label][i] for i in k_smallest_idx]

        seq = [v for v in label_to_k_smallest.values()]
        seq = list(itertools.chain.from_iterable(seq))

        seq.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))

        return seq


    def prLightPurple(self, text): print("\033[94m {}\033[00m".format(text))

    def prRedIntro(self, text): print("Extracting frames \033[91m {}\033[00m".format(text))

    def get_sequence(self):
        assert os.path.isdir(self.cluster_dir), "Cluster directory is invalid"
        cluster_dict =  pickle.load(open(os.path.join(self.cluster_dir, "frame_to_cluster.pkl"), 'rb'))

        seq = self.get_cluster_sequence(cluster_dict)

        directory = os.path.dirname(seq[0])
        file_names = [os.path.basename(file) for file in seq]

        self.prRedIntro("Extracting Synopsis frames:")
        for file_name in file_names:
            self.prLightPurple(file_name)

        if not os.path.isdir(directory):
            self.prRedIntro("You have deleted Extracted Vidoe director, Please re-run PIPLINE")
            return

        if not os.path.isdir(self.synopsis_dir):
            os.mkdir(self.synopsis_dir)

        for file in file_names:
            shutil.copy2(os.path.join(directory, file), self.synopsis_dir)
        return directory, file_names
