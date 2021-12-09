""" PointNet++ Layers

Author: Charles R. Qi
Modified by Xingyu Liu
Date: November 2019
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, query_ball_point_var_rad, query_ball_point_var_rad_var_seed, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    sample_idx = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, sample_idx)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
    if points is not None:
        grouped_points = group_point(points, idx)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, sample_idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz



def meteor_direct_module(xyz, time, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, module_type='ind', fps=True, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    '''
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            time: (batch_size, ndataset, 1) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radiuses in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            module_type: 'ind' or 'rel' -- the type of meteor module
            fps: whether to do farthest point sampling; Requires npoint == xyz.get_shape()[1].value, when fps=False
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    sample_idx = None
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:

        if fps:
            ##### sample and group with variable radius
            sample_idx = farthest_point_sample(npoint, xyz)
        else:
            ##### no sampling at all
            sample_idx = tf.tile(tf.expand_dims(tf.range(npoint, dtype=tf.int32), 0), [batch_size, 1])

        new_xyz = gather_point(xyz, sample_idx)
        new_time = gather_point(time, sample_idx)
        new_points = gather_point(points, sample_idx)
        time_ = tf.reshape(time, [batch_size, 1, -1])
        new_time_ = tf.abs(new_time - time_)
        radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32))

        idx, pts_cnt = query_ball_point_var_rad(radius_, nsample, xyz, new_xyz)

        grouped_xyz = group_point(xyz, idx)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
        if points is not None:
            grouped_points = group_point(points, idx)
            grouped_time = group_point(time, idx)
            if use_xyz:
                if module_type == 'ind':
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points], axis=-1)
                else:
                    new_points_expand = tf.tile(tf.expand_dims(new_points, 2), [1,1,nsample,1])
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points, new_points_expand], axis=-1)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)

        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
        return new_xyz, new_time, new_points, idx


def meteor_direct_module8(xyz, time, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,
                         bn=True,side=1,side_scale=1.1, radius_scale=1.1*1.00001, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    '''
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            time: (batch_size, ndataset, 1) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radiuses in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    print(xyz)
    sample_idx = None
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        ##### sample and group with variable radius

        # <------My Code: Find 2048 * 27
        delta_xyz = [[0.4714, -0.8165, -0.3333], [0.4714, 0.8165, -0.3333], [-0.9428, 0.0, -0.3333], [0.0, 0.0, 1.0]]
        
        standard=radius[0]
        side=(standard*side_scale).astype(np.float32)
        radius=(radius*radius_scale)
        
        delta_xyz = tf.reshape(tf.convert_to_tensor(delta_xyz), [1, 1, 4, 3])  # inside_covert [27,3] outside_reshape
        sample_idx = farthest_point_sample(npoint, xyz)  # index of 2048 from current frame

        new_xyz_core = gather_point(xyz, sample_idx)
        new_xyz = tf.expand_dims(new_xyz_core, 2)

        # side=1: side of cube

        new_anchor_xyz = tf.reshape(new_xyz + delta_xyz*side, [batch_size, npoint * 4, 3])


        new_time_core = gather_point(time, sample_idx)
        new_time = tf.tile(new_time_core, [1, 4, 1])
        time_a = tf.tile(new_time, [1, 1, 8])
        time_b = tf.reshape(time_a, [batch_size, npoint * 4, 8, 1])

        time_ = tf.reshape(time, [batch_size, 1, -1])

        new_time_ = tf.abs(new_time - time_)
        radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32))
        # End My Code: Find 2048 * 27 ------>

        # <-----My query_ball
        # 9 represents another sample
        idx, pts_cnt = query_ball_point_var_rad(radius_, 8, xyz, new_anchor_xyz)
        mask = tf.reshape(tf.stop_gradient(tf.cast(pts_cnt > 0, float)), [batch_size, npoint * 4, 1, 1])

        grouped_xyz = group_point(xyz, idx)
        grouped_xyz -= tf.tile(tf.expand_dims(new_anchor_xyz, 2),[1,1,8,1])
        if points is not None:
            grouped_points = group_point(points, idx)
            grouped_time = group_point(time, idx)
            temp_time = tf.abs(grouped_time - time_b)
            if use_xyz:
                new_points = tf.concat([grouped_xyz, grouped_points, temp_time],
                                       axis=-1)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        new_points = new_points * mask
        # My query_ball------>

        # <-----My Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        # After mlp : 2 layers
        local_feature = tf.concat([new_points, grouped_points, grouped_xyz, temp_time], axis=-1)
        local_weight = tf_util.conv2d(local_feature, new_points.get_shape()[-1], kernel_size=[1, 1],
                                      padding='VALID', stride=[1, 1],
                                      bn=bn, is_training=is_training, activation_fn=None,#
                                      scope='convw', bn_decay=bn_decay, data_format=data_format)
        local_weight = tf.nn.softmax(local_weight, dim=2)
        new_points = tf.multiply(new_points, local_weight)
        new_points = tf.reduce_sum(new_points, axis=2)


        # After pooling : 1 layer

        new_points = tf.reshape(new_points, [batch_size, npoint, 4, -1])

        for i, num_out_channel in enumerate(mlp2):
            i+=len(mlp)
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 4],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conva%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        # After mlp : 1 layers

        new_points = tf.squeeze(new_points, [2])


        return new_xyz_core, new_time_core, new_points, idx
        # My Point Feature Embedding------>

def fp_model7(xyz1, xyz2, points1, points2, time1, time2, npoint, radius, mlp, mlp2, group_all, is_training, bn_decay, scope,
                         bn=True,side=1,side_scale=1.1, radius_scale=1.1*1.00001, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    '''
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            time: (batch_size, ndataset, 1) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radiuses in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    print(xyz1)
    #sample_idx = None
    batch_size = xyz1.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        ##### sample and group with variable radius

        # <------My Code: Find 2048 * 27
        delta_xyz = [[0.4714, -0.8165, -0.3333], [0.4714, 0.8165, -0.3333], [-0.9428, 0.0, -0.3333], [0.0, 0.0, 1.0]]
        
        #radius=radius
        standard=radius[0]
        side=(standard*side_scale).astype(np.float32)
        radius=(radius*radius_scale)
        
        delta_xyz = tf.reshape(tf.convert_to_tensor(delta_xyz), [1, 1, 4, 3])  # inside_covert [27,3] outside_reshape
        #sample_idx = farthest_point_sample(npoint, xyz)  # index of 2048 from current frame

        new_xyz = tf.expand_dims(xyz1, 2)

        # side=1: side of cube
        new_anchor_xyz = tf.reshape(new_xyz + delta_xyz*side, [batch_size, npoint * 4, 3])
        new_time = tf.tile(time1, [1, 4, 1])
        time_a = tf.tile(new_time, [1, 1, 8])
        time_b = tf.reshape(time_a, [batch_size, npoint * 4, 8, 1])
        time_ = tf.reshape(time2, [batch_size, 1, -1])
        new_time_ = tf.abs(new_time - time_)
        radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32))
        # End My Code: Find 2048 * 27 ------>


        # <-----My query_ball
        # 9 represents another sample
        idx, pts_cnt = query_ball_point_var_rad(radius_, 8, xyz2, new_anchor_xyz)
        mask = tf.reshape(tf.stop_gradient(tf.cast(pts_cnt > 0, float)), [batch_size, npoint * 4, 1, 1])


        grouped_xyz = group_point(xyz2, idx)
        grouped_xyz -= tf.tile(tf.expand_dims(new_anchor_xyz, 2),[1,1,8,1])
        if points2 is not None:
            grouped_points = group_point(points2, idx)
            grouped_time = group_point(time2, idx)
            temp_time = tf.abs(grouped_time - time_b)
            if use_xyz:
                new_points = tf.concat([grouped_xyz, grouped_points, temp_time],
                                       axis=-1)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        new_points = new_points * mask
        #new_points = tf.to_float(new_points, name='ToFloat')
        # My query_ball------>

        # <-----My Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        # After mlp : 2 layers
        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')

        # After pooling : 1 layer
        new_points = tf.reshape(new_points, [batch_size, npoint, 4, -1])

        for i, num_out_channel in enumerate(mlp2):
            i+=len(mlp)
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 4],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conva%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        # After mlp : 1 layers

        new_points = tf.squeeze(new_points, [2])
        # After pooling : 2 layers
        # Output of layer 1: [12,2048,mlp1[-1]]
        if points1 is not None:
            new_points = tf.concat(axis=2, values=[new_points, points1]) # B,ndataset1,nchannel1+nchannel2
        new_points = tf.expand_dims(new_points, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2])
        return new_points


def set_upconv_module(xyz1, xyz2, feat1, feat2, nsample, mlp, mlp2, is_training, scope, bn_decay=None, bn=True, pooling='max', radius=None, knn=True):
    """
        Feature propagation from xyz2 (less points) to xyz1 (more points)
    Inputs:
        xyz1: (batch_size, npoint1, 3)
        xyz2: (batch_size, npoint2, 3)
        feat1: (batch_size, npoint1, channel1) features for xyz1 points (earlier layers)
        feat2: (batch_size, npoint2, channel2) features for xyz2 points
    Output:
        feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)
        TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
    """
    with tf.variable_scope(scope) as sc:
        if knn:
            l2_dist, idx = knn_point(nsample, xyz2, xyz1)
        else:
            idx, pts_cnt = query_ball_point(radius, nsample, xyz2, xyz1)
        xyz2_grouped = group_point(xyz2, idx)
        xyz1_expanded = tf.expand_dims(xyz1, 2)
        xyz_diff = xyz2_grouped - xyz1_expanded

        feat2_grouped = group_point(feat2, idx)
        net = tf.concat([feat2_grouped, xyz_diff], axis=3)

        if mlp is None: mlp=[]
        for i, num_out_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_out_channel, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv%d'%(i), bn_decay=bn_decay)
        if pooling=='max':
            feat1_new = tf.reduce_max(net, axis=[2], keep_dims=False, name='maxpool')
        elif pooling=='avg':
            feat1_new = tf.reduce_mean(net, axis=[2], keep_dims=False, name='avgpool')

        if feat1 is not None:
            feat1_new = tf.concat([feat1_new, feat1], axis=2)

        feat1_new = tf.expand_dims(feat1_new, 2)
        if mlp2 is None: mlp2=[]
        for i, num_out_channel in enumerate(mlp2):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='post-conv%d'%(i), bn_decay=bn_decay)
        feat1_new = tf.squeeze(feat1_new, [2]) # batch_size, npoint1, mlp2[-1]
        return feat1_new

