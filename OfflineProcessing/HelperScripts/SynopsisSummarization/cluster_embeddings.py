import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import hdbscan
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
from random import sample


class ClusterEmbedding():
    def __init__(self, cluster_img_dir, embeddings_dir, cluster_algorithm,
                    cluster_dir, eps=0.5, min_samples=5, n_clusters=15, fps=30, sample_rate=3, dim=30):
        self.cluster_img_dir = cluster_img_dir
        self.embeddings_dir = embeddings_dir
        self.cluster_algorithm = cluster_algorithm
        self.cluster_dir = cluster_dir
        self.eps = eps
        self.min_samples= min_samples
        self.n_clusters = n_clusters
        self.fps = fps
        self.sample_rate = sample_rate
        self.dim = dim

    def cluster_helper(self, embeddings):
        labels = None
        if self.cluster_algorithm.upper() == "DBSCAN":
            cluster = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1).fit(embeddings)
            labels = cluster.labels_
        if self.cluster_algorithm.upper() == "KMEANS":
            cluster = KMeans(n_clusters=self.n_clusters).fit(embeddings)
            labels = cluster.labels_
        if self.cluster_algorithm.upper() == "HDBSCAN":
            cluster = hdbscan.HDBSCAN(min_cluster_size=self.n_clusters)
            labels = cluster.fit_predict(embeddings)
        self.plottr(labels, embeddings)
        print(len(set(labels)))
        return labels

    def plottr(self, labels, embeddings):
        if not os.path.isdir(self.cluster_img_dir):
            os.mkdir(self.cluster_img_dir)

        embeddings_arr = np.vstack(embeddings)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="plasma")
        # plt.scatter(embeddings_arr[:, 0], embeddings_arr[:, 1], c=labels, cmap="plasma")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Unique Events")
        plt.savefig(os.path.join(self.cluster_img_dir, self.cluster_algorithm + ".png"))
        plt.show()

    def subsample(self, length, embed_dict):
        window = self.fps
        rate = self.fps//self.sample_rate
        samples = []
        for i in range(0, length, window):
            if i + window < length:
                value = [*range(i, i + window)]
                samples.append(sample(value, rate))
            else:
                value = [*range(i, length)]
                samples.append(value)
        samples = list(itertools.chain.from_iterable(samples))
        samples.sort()

        frame_keys = list(embed_dict.keys())
        frame_keys_ = [frame_keys[i] for i in samples]
        frame_values_ = [embed_dict[i] for i in frame_keys_]
        return frame_keys_, frame_values_

    def reduce_dim(self, frame_values_):
        frames_arr = np.vstack(frame_values_)
        reduced = PCA(n_components=self.dim).fit(frames_arr).transform(frames_arr)
        print(reduced[0])
        return reduced

    def cluster(self):
        if not os.path.isdir(self.cluster_dir):
            os.mkdir(self.cluster_dir)
        embeddings = []
        frame_names = []

        cluster_dict = {}

        # get embeddings
        embed_dict = pickle.load(open(os.path.join(self.embeddings_dir, "frame_embeddings.pkl"), 'rb'))

        sample_frames, sample_embeddings = self.subsample(len(embed_dict), embed_dict)
        reduced_embeddings = self.reduce_dim(sample_embeddings)

        labels = self.cluster_helper(reduced_embeddings)
        for i in tqdm(range(len(sample_frames)), desc="Mapping frames to clusters", ncols=100):
            cluster_dict[sample_frames[i]] = labels[i]

        with open(os.path.join(self.cluster_dir, "frame_to_cluster.pkl"), 'wb') as file:
            pickle.dump(cluster_dict, file)
