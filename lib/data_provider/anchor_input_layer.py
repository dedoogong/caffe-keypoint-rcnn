# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
AnchorDataLayer implements a Caffe Python layer.
"""

import caffe 
import numpy as np 
import pickle
import cv2
from blob import im_list_to_blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[:2])
    im_size_max = np.max(im_shape[:2])
    im_scale = float(target_size) / float(im_size_min)
    print('im_scale :', im_scale)
    print('target_size :', target_size)
    print('im_shape :', im_shape)
    print('im_size_min :', im_size_min)
    print('im_size_max :', im_size_max)
    print('np.round(im_scale * im_size_max) :', np.round(im_scale * im_size_max))
    print('max_size :', max_size)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    print('im_scale :', im_scale)
    im = cv2.resize(
          im,
          None,
          None,
          fx=im_scale,
          fy=im_scale,
          interpolation=cv2.INTER_LINEAR
    )
    return im, im_scale

def get_image_blob(im, target_scale=800, target_max_size=1333):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    print('im shape', im.shape)
    processed_im, im_scale = prep_im_for_blob(
        im, [102.9801,115.9465, 122.7717], target_scale, target_max_size
    )

    max_shape = np.array([processed_im.shape]).max(axis=0)
    blob = np.zeros(
        (1, max_shape[0], max_shape[1], 3), dtype=np.float32
    )
   
    blob[0, 0:processed_im.shape[0], 0:processed_im.shape[1], :] = processed_im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    print('before swap : ', blob.shape)
    blob = blob.transpose(channel_swap)
    print('after swap : ', blob.shape)
    # NOTE: this height and width may be larger than actual scaled input image
    # due to the FPN.COARSEST_STRIDE related padding in im_list_to_blob. We are
    # maintaining this behavior for now to make existing results exactly
    # reproducible (in practice using the true input image height and width
    # yields nearly the same results, but they are sometimes slightly different
    # because predictions near the edge of the image will be pruned more
    # aggressively).
    print('final im_scale', im_scale)
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.asarray([[height, width, im_scale]])
    return blob, im_info.reshape(1,3)

class AnchorDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        print('==================================================anchor_data_layer_setup==================================================') 

        top[0].reshape(3,4)
        top[1].reshape(3,4)
        top[2].reshape(3,4)
        top[3].reshape(3,4)
        top[4].reshape(3,4)
        top[5].reshape(1,3)
        top[6].reshape(1,3,800,800)
    def forward(self, bottom, top):
        print('==================================================anchor_data_layer_forward==================================================') 
        im=cv2.imread('/home/lee/caffe_vistool/deep-visualization-toolbox/input_images/test.jpg')
        print(im.shape)
        
        blob=np.array([[-22., -10.,  25.,  13.], [-14., -14.,  17.,  17.], [-10., -22.,  13.,  25.]])        
        top[0].data[...] = blob.astype(np.float32, copy=False)
        
        blob=np.array([[-40., -20.,  47.,  27.], [-28., -28.,  35.,  35.], [-20., -44.,  27.,  51.]])
        top[1].data[...] = blob.astype(np.float32, copy=False)

        blob=np.array([[-84., -40.,  99.,  55.], [-56., -56.,  71.,  71.], [-36., -80.,  51.,  95.]])
        top[2].data[...] = blob.astype(np.float32, copy=False)

        blob=np.array([[-164.,  -72.,  195.,  103.], [-112., -112.,  143.,  143.], [ -76., -168.,  107.,  199.]])
        top[3].data[...] = blob.astype(np.float32, copy=False)

        blob=np.array([[-332., -152.,  395.,  215.], [-224., -224.,  287.,  287.], [-148., -328.,  211.,  391.]])
        top[4].data[...] = blob.astype(np.float32, copy=False) 
        
        img_blob, im_info = get_image_blob(im)
        print(im_info)
        print(im_info.shape)
        img_ch1 = img_blob[0, 0, :, :]
        print(img_ch1.shape)
        print(np.min(img_ch1 ))
        print(np.max(img_ch1 ))
        top[5].data[...] = im_info
        
        top[6].data[...] = img_blob        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

