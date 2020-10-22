"""
Generate embeddings from each frame using Inception net.
Output is single pickl file containing individual embeddings
%% tensorflow --version==1.13.0
"""
import os
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pickle
import urllib.request
import re
import tarfile
import sys, traceback

class FrameToEmbeddings():
    def __init__(self, extracted_frame_dir, embeddings_dir):
        self.extracted_frame_dir = extracted_frame_dir
        self.MODEL_DIR = 'inception-2015-12-05'
        self.embeddings_dir = embeddings_dir

        def retrieveModel():
            '''
            Retrieve Inception model
            '''
            filename="inception-2015-12-05"
            if filename not in os.listdir('.'):
                try:
                    url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

                    temp = urllib.request.urlretrieve(url, filename=None)[0]
                    base_name = os.path.basename(url)

                    file_name, file_extension = os.path.splitext(base_name)
                    tar = tarfile.open(temp)
                    tar.extractall(file_name)
                except:
                    traceback.print_exc(file=sys.stdout)
                    print("Error could not retrieve Inception Model!")

        retrieveModel()

    def create_graph(self):
        '''
        Create tensorflow graph
        '''
        with tf.gfile.GFile(os.path.join(self.MODEL_DIR, "classify_image_graph_def.pb"), 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            _ = tf.import_graph_def(graph_def, name='')

    def extract_embeddings_helper(self, files):
        '''
        @return: dictionary mapping each frame to its respective embeddings
        '''
        self.create_graph()
        embed_dict = {}
        config = tf.ConfigProto(device_count = {'GPU': 0})
        with tf.Session(config=config) as sess:
            count = 0
            for file in tqdm(files, desc="Embedding frames", ncols=100):
                if not tf.gfile.Exists(file):
                    tf.logging.fatal('File does not exist %s', file)
                frame = tf.gfile.GFile(file, 'rb').read()

                softmax_layer = sess.graph.get_tensor_by_name('softmax:0')
                embedding_layer = sess.graph.get_tensor_by_name('pool_3:0')
                embeddings = sess.run(embedding_layer, {'DecodeJpeg/contents:0': frame})

                embed_dict[file] = embeddings.reshape(2048)
                count += 1
        return embed_dict

    def extract_embeddings(self):
        assert os.path.isdir(self.extracted_frame_dir), "Not a valid directory [Frames]"

        if not os.path.isdir(self.embeddings_dir):
            os.mkdir(self.embeddings_dir)
            print("Creating directory ", self.embeddings_dir)

        image_files = [os.path.join(self.extracted_frame_dir, file) for file in os.listdir(self.extracted_frame_dir) if re.match('(jpg|jpeg)', file.split('.')[1])]
        embedding_dict = self.extract_embeddings_helper(image_files)

        #pickle embeddings
        with open(os.path.join(self.embeddings_dir, 'frame_embeddings.pkl'), 'wb') as file:
            pickle.dump(embedding_dict, file)

        print("Saving embeddings!")
