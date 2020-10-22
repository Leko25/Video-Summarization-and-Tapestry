from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import sys
import numpy as np
from config import *
from utilities import preprocess_image, postprocess_predictions
from models import sam_vgg
from scipy.misc import imread, imsave
import random

def get_test(data):
    Xims = np.zeros((1, im_res, im_res, 3))
    original_img = imread(data['image'])
    img_name = os.path.basename(data['image'])
    if original_img.ndim == 2:
        copy = np.zeros((original_img.shape[0], original_img.shape[1], 3))
        copy[:, :, 0] = original_img
        copy[:, :, 1] = original_img
        copy[:, :, 2] = original_img
        original_img = copy
    r_img = preprocess_image(original_img, im_res, im_res)
    Xims[0, :] = np.copy(r_img)
    return [Xims], original_img, img_name

if __name__ == '__main__':
    seed = 7
    random.seed(seed)
    test_data = []
    mypath=sys.argv[1]
    testing_images = [sys.argv[1]+"/"+f for f in os.listdir(mypath)]
    testing_images.sort()

    for image in testing_images:
        data = {'image': image}
        test_data.append(data)

    x = Input(batch_shape=(1, im_res, im_res, 3))
    m = Model(inputs=x, outputs=sam_vgg(x))

    print("Loading weights")
    m.load_weights('PAGE-Net.h5')
    print("Making prediction")
    # Output Folder Path
    saliency_output = sys.argv[2]
    print "Saving saliency files in "+ saliency_output

    if not os.path.exists(saliency_output):
        os.makedirs(saliency_output)

    for data in test_data:
        Ximg, original_image, img_name = get_test(data)
        predictions = m.predict(Ximg, batch_size=1)
        res_saliency = postprocess_predictions(predictions[9][0, :, :, 0], original_image.shape[0],
                                                   original_image.shape[1])
        imsave(saliency_output + '/%s.png' % img_name[0:-4], res_saliency.astype(int))