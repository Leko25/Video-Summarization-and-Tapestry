from __future__ import division
from keras.layers import Softmax, Average, Add, Reshape, MaxPooling2D, Flatten, RepeatVector, Concatenate, Permute, Multiply
from keras.layers.convolutional import Conv2D
import tensorflow as tf
import keras

class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = target.shape
        return tf.image.resize_images(source, [target_shape[1], target_shape[2]])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

def sam_vgg(data):
    # conv_1
    trainable = True
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(data)
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(conv_1_out)

    ds_conv_1_out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_out)

    # conv_2
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(ds_conv_1_out)
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(conv_2_out)

    ds_conv_2_out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_out)

    # conv_3
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(ds_conv_2_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(conv_3_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(conv_3_out)

    ds_conv_3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(conv_3_out)

    # conv_4
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(ds_conv_3_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(conv_4_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(conv_4_out)

    ds_conv_4_out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(conv_4_out)

    # conv_5 #
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(ds_conv_4_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(conv_5_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(conv_5_out)

    conv_6_out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', padding='same')(conv_5_out)

    # salconv_6#
    salconv_6_out = Conv2D(512, (5, 5), padding='same', activation='relu', trainable=trainable)(conv_6_out)
    salconv_6_out = Conv2D(512, (5, 5), padding='same', activation='sigmoid', trainable=trainable)(salconv_6_out)

    salconv_5_out = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_5_out)
    salconv_5_out = Conv2D(512, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_5_out)

    edgeconv_5_out = Conv2D(512, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_5_out)
    edgeconv_5_out = Conv2D(512, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_5_out)

    salconv_4_out = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_4_out)
    salconv_4_out = Conv2D(256, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_4_out)

    edgeconv_4_out = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_4_out)
    edgeconv_4_out = Conv2D(256, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_4_out)

    salconv_3_out = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_3_out)
    salconv_3_out = Conv2D(256, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_3_out)

    edgeconv_3_out = Conv2D(256, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_3_out)
    edgeconv_3_out = Conv2D(256, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_3_out)

    salconv_2_out = Conv2D(128, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_2_out)
    salconv_2_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_2_out)

    edgeconv_2_out = Conv2D(128, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_2_out)
    edgeconv_2_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_2_out)

    salconv_1_out = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_1_out)
    salconv_1_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_1_out)

    edgeconv_1_out = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=trainable)(conv_1_out)
    edgeconv_1_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_1_out)

    # saliency from conv_6 #
    saliency6 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(salconv_6_out)
    saliency6_up = UpsampleLike()([saliency6, conv_1_out])

    # saliency from conv_5 #
    edge5 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_5_out)
    edge5_up = UpsampleLike()([edge5, conv_1_out])

    salconv_5_out = Concatenate()([salconv_5_out, UpsampleLike()([saliency6, salconv_5_out])])
    salconv_5_out = Conv2D(256, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_5_out)

    attention_5a = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(salconv_5_out)
    attention_5a = Flatten()(attention_5a)
    attention_5a = Softmax()(attention_5a)

    attention_5b = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3))\
        (MaxPooling2D((2, 2), strides=(2, 2), padding='same')(salconv_5_out))
    attention_5b = UpsampleLike()([attention_5b, salconv_5_out])
    attention_5b = Flatten()(attention_5b)
    attention_5b = Softmax()(attention_5b)

    attention_5 = Average()([attention_5a, attention_5b])

    attention_5 = RepeatVector(256)(attention_5)
    attention_5 = Permute((2, 1))(attention_5)
    attention_5 = Reshape((14, 14, 256), name='att_conv5')(attention_5)
    attention_5 = Multiply()([attention_5, salconv_5_out])
    salconv_5_out = Add()([attention_5, salconv_5_out])

    salconv_5_out = Concatenate()([salconv_5_out, edge5])
    salconv_5_out = Conv2D(256, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_5_out)
    saliency5 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(salconv_5_out)
    saliency5_up = UpsampleLike()([saliency5, conv_1_out])

    # saliency from conv_4 #
    edgeconv_4_out = Concatenate()([edgeconv_4_out, UpsampleLike()([edge5, salconv_4_out])])
    edgeconv_4_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_4_out)
    edge4 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_4_out)
    edge4_up = UpsampleLike()([edge4, conv_1_out])

    salconv_4_out = Concatenate()([salconv_4_out, UpsampleLike()([saliency6, salconv_4_out]), UpsampleLike()([saliency5, salconv_4_out])])
    salconv_4_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_4_out)

    attention_4a = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(salconv_4_out)
    attention_4a = Flatten()(attention_4a)
    attention_4a = Softmax()(attention_4a)

    attention_4b = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((2, 2), strides=(2, 2), padding='same')(salconv_4_out))
    attention_4b = UpsampleLike()([attention_4b, salconv_4_out])
    attention_4b = Flatten()(attention_4b)
    attention_4b = Softmax()(attention_4b)

    attention_4c = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((4, 4), strides=(4, 4), padding='same')(salconv_4_out))
    attention_4c = UpsampleLike()([attention_4c, salconv_4_out])
    attention_4c = Flatten()(attention_4c)
    attention_4c = Softmax()(attention_4c)

    attention_4 = Average()([attention_4a, attention_4b, attention_4c])

    attention_4 = RepeatVector(128)(attention_4)
    attention_4 = Permute((2, 1))(attention_4)
    attention_4 = Reshape((28, 28, 128), name='att_conv4')(attention_4)
    attention_4 = Multiply()([attention_4, salconv_4_out])
    salconv_4_out = Add()([attention_4, salconv_4_out])

    salconv_4_out = Concatenate()([salconv_4_out, edge4])
    salconv_4_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_4_out)
    saliency4 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(salconv_4_out)
    saliency4_up = UpsampleLike()([saliency4, conv_1_out])

    # saliency from conv_3 #
    edgeconv_3_out = Concatenate()([edgeconv_3_out, UpsampleLike()([edge4, salconv_3_out]), UpsampleLike()([edge5, salconv_3_out])])
    edgeconv_3_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_3_out)
    edge3 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_3_out)
    edge3_up = UpsampleLike()([edge3, conv_1_out])

    salconv_3_out = Concatenate()(
        [salconv_3_out, UpsampleLike()([saliency6, salconv_3_out]), UpsampleLike()([saliency5, salconv_3_out]), UpsampleLike()([saliency4, salconv_3_out])])
    salconv_3_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_3_out)

    attention_3a = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(salconv_3_out)
    attention_3a = Flatten()(attention_3a)
    attention_3a = Softmax()(attention_3a)

    attention_3b = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((2, 2), strides=(2, 2), padding='same')(salconv_3_out))
    attention_3b = UpsampleLike()([attention_3b, salconv_3_out])
    attention_3b = Flatten()(attention_3b)
    attention_3b = Softmax()(attention_3b)

    attention_3c = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((4, 4), strides=(4, 4), padding='same')(salconv_3_out))
    attention_3c = UpsampleLike()([attention_3c, salconv_3_out])
    attention_3c = Flatten()(attention_3c)
    attention_3c = Softmax()(attention_3c)

    attention_3d = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((8, 8), strides=(8, 8), padding='same')(salconv_3_out))
    attention_3d = UpsampleLike()([attention_3d, salconv_3_out])
    attention_3d = Flatten()(attention_3d)
    attention_3d = Softmax()(attention_3d)

    attention_3 = Average()([attention_3a, attention_3b, attention_3c, attention_3d])

    attention_3 = RepeatVector(128)(attention_3)
    attention_3 = Permute((2, 1))(attention_3)
    attention_3 = Reshape((56, 56, 128), name='att_conv3')(attention_3)
    attention_3 = Multiply()([attention_3, salconv_3_out])
    salconv_3_out = Add()([attention_3, salconv_3_out])

    salconv_3_out = Concatenate()([salconv_3_out, edge3])
    salconv_3_out = Conv2D(128, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_3_out)
    saliency3 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(salconv_3_out)
    saliency3_up = UpsampleLike()([saliency3, conv_1_out])

    # saliency from conv_2 #
    edgeconv_2_out = Concatenate()(
        [edgeconv_2_out, UpsampleLike()([edge5, salconv_2_out]), UpsampleLike()([edge4, salconv_2_out]), UpsampleLike()([edge3, salconv_2_out])])
    edgeconv_2_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_2_out)
    edge2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_2_out)
    edge2_up = UpsampleLike()([edge2, conv_1_out])

    salconv_2_out = Concatenate()(
        [salconv_2_out, UpsampleLike()([saliency6, salconv_2_out]), UpsampleLike()([saliency5, salconv_2_out]),
         UpsampleLike()([saliency4, salconv_2_out]), UpsampleLike()([saliency3, salconv_2_out])])
    salconv_2_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_2_out)

    attention_2a = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(salconv_2_out)
    attention_2a = Flatten()(attention_2a)
    attention_2a = Softmax()(attention_2a)

    attention_2b = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((2, 2), strides=(2, 2), padding='same')(salconv_2_out))
    attention_2b = UpsampleLike()([attention_2b, salconv_2_out])
    attention_2b = Flatten()(attention_2b)
    attention_2b = Softmax()(attention_2b)

    attention_2c = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((4, 4), strides=(4, 4), padding='same')(salconv_2_out))
    attention_2c = UpsampleLike()([attention_2c, salconv_2_out])
    attention_2c = Flatten()(attention_2c)
    attention_2c = Softmax()(attention_2c)

    attention_2d = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((8, 8), strides=(8, 8), padding='same')(salconv_2_out))
    attention_2d = UpsampleLike()([attention_2d, salconv_2_out])
    attention_2d = Flatten()(attention_2d)
    attention_2d = Softmax()(attention_2d)

    attention_2e = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((16, 16), strides=(16, 16), padding='same')(salconv_2_out))
    attention_2e = UpsampleLike()([attention_2e, salconv_2_out])
    attention_2e = Flatten()(attention_2e)
    attention_2e = Softmax()(attention_2e)

    attention_2 = Average()([attention_2a, attention_2b, attention_2c, attention_2d, attention_2e])

    attention_2 = RepeatVector(64)(attention_2)
    attention_2 = Permute((2, 1))(attention_2)
    attention_2 = Reshape((112, 112, 64), name='att_conv2')(attention_2)
    attention_2 = Multiply()([attention_2, salconv_2_out])
    salconv_2_out = Add()([attention_2, salconv_2_out])

    salconv_2_out = Concatenate()([salconv_2_out, edge2])
    salconv_2_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_2_out)
    saliency2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(salconv_2_out)
    saliency2_up = UpsampleLike()([saliency2, conv_1_out])

    # saliency from conv_1 #
    edgeconv_1_out = Concatenate()(
        [edgeconv_1_out, UpsampleLike()([edge5, salconv_1_out]), UpsampleLike()([edge4, salconv_1_out]),
         UpsampleLike()([edge3, salconv_1_out]), UpsampleLike()([edge2, salconv_1_out])])
    edgeconv_1_out = Conv2D(32, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_1_out)
    edge1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(edgeconv_1_out)

    salconv_1_out = Concatenate()(
        [salconv_1_out, UpsampleLike()([saliency6, salconv_1_out]), UpsampleLike()([saliency5, salconv_1_out]),
         UpsampleLike()([saliency4, salconv_1_out]), UpsampleLike()([saliency3, salconv_1_out]), UpsampleLike()([saliency2, salconv_1_out])])
    salconv_1_out = Conv2D(32, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_1_out)

    attention_1a = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(salconv_1_out)
    attention_1a = Flatten()(attention_1a)
    attention_1a = Softmax()(attention_1a)

    attention_1b = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((2, 2), strides=(2, 2), padding='same')(salconv_1_out))
    attention_1b = UpsampleLike()([attention_1b, salconv_1_out])
    attention_1b = Flatten()(attention_1b)
    attention_1b = Softmax()(attention_1b)

    attention_1c = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((4, 4), strides=(4, 4), padding='same')(salconv_1_out))
    attention_1c = UpsampleLike()([attention_1c, salconv_1_out])
    attention_1c = Flatten()(attention_1c)
    attention_1c = Softmax()(attention_1c)

    attention_1d = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((8, 8), strides=(8, 8), padding='same')(salconv_1_out))
    attention_1d = UpsampleLike()([attention_1d, salconv_1_out])
    attention_1d = Flatten()(attention_1d)
    attention_1d = Softmax()(attention_1d)

    attention_1e = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((16, 16), strides=(16, 16), padding='same')(salconv_1_out))
    attention_1e = UpsampleLike()([attention_1e, salconv_1_out])
    attention_1e = Flatten()(attention_1e)
    attention_1e = Softmax()(attention_1e)

    attention_1f = Conv2D(1, (1, 1), padding='same', activation='sigmoid', dilation_rate=(3, 3)) \
        (MaxPooling2D((32, 32), strides=(32, 32), padding='same')(salconv_1_out))
    attention_1f = UpsampleLike()([attention_1f, salconv_1_out])
    attention_1f = Flatten()(attention_1f)
    attention_1f = Softmax()(attention_1f)

    attention_1 = Average()([attention_1a, attention_1b, attention_1c, attention_1d, attention_1e, attention_1f])

    attention_1 = RepeatVector(32)(attention_1)
    attention_1 = Permute((2, 1))(attention_1)
    attention_1 = Reshape((224, 224, 32), name='att_conv1')(attention_1)
    attention_1 = Multiply()([attention_1, salconv_1_out])
    salconv_1_out = Add()([attention_1, salconv_1_out])

    salconv_1_out = Concatenate()([salconv_1_out, edge1])
    salconv_1_out = Conv2D(32, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(salconv_1_out)
    saliency1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', trainable=trainable)(salconv_1_out)

    return [saliency6_up, saliency5_up, edge5_up,
            saliency4_up, edge4_up, saliency3_up, edge3_up,
            saliency2_up, edge2_up, saliency1, edge1]