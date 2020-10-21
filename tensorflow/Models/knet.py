import numpy as np
import tensorflow as tf
from model_utils import batch_norm_relu
from Layers import layers as custom_layers
from collections import OrderedDict
from Metrics import observables_config as ocfg

def knet(inputs,
        num_classes,
        train=True,
        weight_decay=1e-5,
        norm=True,
        use_bn=False,
        add_conv4=False,
        alignment=None,
        gpu_mode=True,
        seed=None,
        drop_final_fc=False,
        **kwargs):

    '''
    Krizhevsky's network used in the standard CIFAR10 Caffe Tutorial: https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full.prototxt
    More modern implementation here: https://mochajl.readthedocs.io/en/latest/tutorial/cifar10.html
    Added batch norm layers (to be potentially used in place of LRN layers) for more robustness to learning hyperparameters
    '''
    params = OrderedDict()
    print("Krizhevsky Net Model func: Seeding the model initializers with weight decay {} and seed: {}".format(weight_decay, seed))
    if gpu_mode:
        im = inputs['images']
    else:
        im = inputs
    print('Input shape', im.shape)
    if use_bn:
        assert norm is False, "Batch norm has been set to True so LRN has to be unused!"
    # get obs kwargs
    obs_kwargs = {}
    for k in ocfg.KWARGS:
        obs_kwargs[k] = kwargs[k]

    with tf.variable_scope('conv1'):
        im = custom_layers.Conv2D(input=im,
                             filters=32,
                             kernel_size=5,
                             strides=1,
                             padding='SAME',
                             activation=None,
                             alignment_relu=True,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.0001, seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)

    with tf.variable_scope('pool1'):
        im = tf.nn.max_pool(value=im,
                            ksize=[1,3,3,1],
                            strides=[1,2,2,1],
                            padding='SAME')

    if use_bn:
        with tf.variable_scope('bn1'):
            im = batch_norm_relu(inputs=im, is_training=train, relu=True)
    elif norm:
        im = tf.nn.relu(im)
        with tf.variable_scope('norm1'):
            im = tf.nn.local_response_normalization(input=im,
                                                    depth_radius=3,
                                                    bias=1,
                                                    alpha=5e-5,
                                                    beta=.75)

    with tf.variable_scope('conv2'):
        im = custom_layers.Conv2D(input=im,
                             filters=32,
                             kernel_size=5,
                             strides=1,
                             padding='SAME',
                             activation='relu',
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)

    with tf.variable_scope('pool2'):
        im = tf.nn.avg_pool(value=im,
                            ksize=[1,3,3,1],
                            strides=[1,2,2,1],
                            padding='SAME')

    if use_bn:
        with tf.variable_scope('bn2'):
            im = batch_norm_relu(inputs=im, is_training=train, relu=False)
    elif norm:
        with tf.variable_scope('norm2'):
            im = tf.nn.local_response_normalization(input=im,
                                                    depth_radius=3,
                                                    bias=1,
                                                    alpha=5e-5,
                                                    beta=.75)

    with tf.variable_scope('conv3'):
        im = custom_layers.Conv2D(input=im,
                             filters=64,
                             kernel_size=5,
                             strides=1,
                             padding='SAME',
                             activation='relu',
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)

    with tf.variable_scope('pool3'):
        im = tf.nn.avg_pool(value=im,
                            ksize=[1,3,3,1],
                            strides=[1,2,2,1],
                            padding='SAME')

    if use_bn: # lrn not used beyond second layer
        with tf.variable_scope('bn3'):
            im = batch_norm_relu(inputs=im, is_training=train, relu=False)

    if add_conv4:
        print('Adding conv4 layer')
        with tf.variable_scope('conv4'):
            im = custom_layers.Conv2D(input=im,
                                 filters=128,
                                 kernel_size=5,
                                 strides=1,
                                 padding='SAME',
                                 activation='relu',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 alignment=alignment,
                                 **obs_kwargs)

        with tf.variable_scope('pool4'):
            im = tf.nn.avg_pool(value=im,
                                ksize=[1,3,3,1],
                                strides=[1,2,2,1],
                                padding='SAME')

        if use_bn: # lrn not used beyond second layer
            with tf.variable_scope('bn4'):
                im = batch_norm_relu(inputs=im, is_training=train, relu=False)

    total_conv_feats = np.prod(im.get_shape().as_list()[1:])
    im = tf.reshape(im, [-1, total_conv_feats])

    if not drop_final_fc:
        fc_nm = 'fc5' if add_conv4 else 'fc4'
        with tf.variable_scope(fc_nm):
            im = custom_layers.Dense(input=im,
                                     units=num_classes,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                     bias_initializer=tf.zeros_initializer(),
                                     activation=None,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     alignment=alignment,
                                     **obs_kwargs)

        print('Logits shape', im.shape)

    if gpu_mode:
        return im, params
    else:
        return im
