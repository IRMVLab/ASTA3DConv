"""
    Compared with model_baseline, do not use correlation output for skip link
    Compared to model_baseline_fixed, added return values to test whether nsample is set reasonably.
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../..'))
sys.path.append(os.path.join(BASE_DIR, '../../tf_ops/sampling'))
import tf_util
from net_utils import *

def placeholder_inputs(batch_size, num_point, num_frames):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames, 3 + 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point * num_frames))
    labelweights_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames))
    return pointclouds_pl, labels_pl, labelweights_pl, masks_pl

def get_model(point_cloud, num_frames, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // num_frames

    l0_xyz = point_cloud[:, :, 0:3]
    l0_time = tf.concat([tf.ones([batch_size, num_point, 1]) * i for i in range(num_frames)], \
            axis=-2)
    l0_points = tf.concat([point_cloud[:, :, 3:], l0_time], axis=-1)

    RADIUS1 = np.array([0.98, 0.99, 1.0], dtype='float32')
    RADIUS2 = RADIUS1 * 2
    RADIUS3 = RADIUS1 * 4
    RADIUS4 = RADIUS1 * 8

    l1_xyz, l1_time, l1_points, l1_indices = meteor_direct_module(l0_xyz, l0_time, l0_points, npoint=2048, radius=RADIUS1, nsample=32, mlp=[32,32,128], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_time, l2_points, l2_indices = meteor_direct_module8(l1_xyz, l1_time, l1_points, npoint=512, radius=RADIUS2, nsample=32, mlp=[64,64], mlp2=[256], group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_time, l3_points, l3_indices = meteor_direct_module8(l2_xyz, l2_time, l2_points, npoint=128, radius=RADIUS3, nsample=32, mlp=[128,128], mlp2=[512], group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_time, l4_points, l4_indices = meteor_direct_module8(l3_xyz, l3_time, l3_points, npoint=64, radius=RADIUS4, nsample=32, mlp=[256,256], mlp2=[1024], group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points = fp_model7(l3_xyz, l4_xyz, l3_points, l4_points, l3_time, l4_time, npoint=128, radius=RADIUS4, mlp=[256], mlp2=[256], group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='fa_layer1')
    l2_points = fp_model7(l2_xyz, l3_xyz, l2_points, l3_points, l2_time, l3_time, npoint=512, radius=RADIUS3, mlp=[256], mlp2=[256], group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2')
    l1_points = fp_model7(l1_xyz, l2_xyz, l1_points, l2_points, l1_time, l2_time, npoint=2048, radius=RADIUS2, mlp=[256], mlp2=[128], group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3')
    l0_points = fp_model7(l0_xyz, l1_xyz, l0_points, l1_points, l0_time, l1_time, npoint=8192*2, radius=RADIUS1, mlp=[128], mlp2=[128], group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='fa_layer4')


    ##### debug
    net = tf_util.conv1d(l0_points, 12, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points

def get_loss(pred, label, mask, end_points, label_weights):
    """ pred: BxNx3,
        label: BxN,
        mask: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy( labels=label, \
                                                            logits=pred, \
                                                            weights=label_weights, \
                                                            reduction=tf.losses.Reduction.NONE)
    classify_loss = tf.reduce_sum(classify_loss * mask) / (tf.reduce_sum(mask) + 1)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)

    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
