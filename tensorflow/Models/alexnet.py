import numpy as np
import tensorflow as tf
from Layers import layers as custom_layers
from collections import OrderedDict
from Metrics import observables_config as ocfg

def alexnet(inputs,
            num_classes,
            train=True,
            weight_decay=0.0005,
            norm=False,
            alignment=None,
            dropout_seed=0,
            gpu_mode=True,
            seed=None,
            drop_final_fc=False,
            **kwargs):

    params = OrderedDict()
    print("Alexnet Model func: Seeding the model initializers with weight decay {} and seed: {}".format(weight_decay, seed))
    if gpu_mode:
        im = inputs['images']
    else:
        im = inputs
    # get obs kwargs
    obs_kwargs = {}
    for k in ocfg.KWARGS:
        obs_kwargs[k] = kwargs[k]

    dropout = 0.5 if train else None

    with tf.variable_scope('conv1'):
        im = custom_layers.Conv2D(input=im,
                             filters=96,
                             kernel_size=11,
                             strides=4,
                             padding='VALID',
                             activation='relu',
                             kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)
    if norm:
        with tf.variable_scope('lrn1'):
            im = tf.nn.local_response_normalization(input=im,
                                                    depth_radius=5,
                                                    bias=1,
                                                    alpha=.0001,
                                                    beta=.75)

    with tf.variable_scope('pool1'):
        im = tf.nn.max_pool(value=im,
                            ksize=[1,3,3,1],
                            strides=[1,2,2,1],
                            padding='SAME')

    with tf.variable_scope('conv2'):
        im = custom_layers.Conv2D(input=im,
                             filters=256,
                             kernel_size=5,
                             strides=1,
                             padding='SAME',
                             activation='relu',
                             kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)

    if norm:
        with tf.variable_scope('lrn2'):
            im = tf.nn.local_response_normalization(input=im,
                                                    depth_radius=5,
                                                    bias=1,
                                                    alpha=.0001,
                                                    beta=.75)

    with tf.variable_scope('pool2'):
        im = tf.nn.max_pool(value=im,
                            ksize=[1,3,3,1],
                            strides=[1,2,2,1],
                            padding='SAME')

    with tf.variable_scope('conv3'):
        im = custom_layers.Conv2D(input=im,
                             filters=384,
                             kernel_size=3,
                             strides=1,
                             padding='SAME',
                             activation='relu',
                             kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)

    with tf.variable_scope('conv4'):
        im = custom_layers.Conv2D(input=im,
                             filters=384,
                             kernel_size=4,
                             strides=1,
                             padding='SAME',
                             activation='relu',
                             kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)

    with tf.variable_scope('conv5'):
        im = custom_layers.Conv2D(input=im,
                             filters=256,
                             kernel_size=3,
                             strides=1,
                             padding='SAME',
                             activation='relu',
                             kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                             bias_initializer='zeros',
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             alignment=alignment,
                             **obs_kwargs)

    with tf.variable_scope('pool5'):
        im = tf.nn.max_pool(value=im,
                            ksize=[1,3,3,1],
                            strides=[1,2,2,1],
                            padding='SAME')

    total_pool5_feats = np.prod(im.get_shape().as_list()[1:])
    im = tf.reshape(im, [-1, total_pool5_feats])

    with tf.variable_scope('fc6'):
        im = custom_layers.Dense(input=im,
                                 units=4096,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=seed),
                                 bias_initializer=tf.constant_initializer(value=0.1),
                                 activation='relu',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 alignment=alignment,
                                 **obs_kwargs)

    if dropout:
        im = tf.nn.dropout(im, keep_prob=dropout, seed=dropout_seed, name='fc6_dropout')

    with tf.variable_scope('fc7'):
        im = custom_layers.Dense(input=im,
                                 units=4096,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=seed),
                                 bias_initializer=tf.constant_initializer(value=0.1),
                                 activation='relu',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 alignment=alignment,
                                 **obs_kwargs)

    if dropout:
        im = tf.nn.dropout(im, keep_prob=dropout, seed=dropout_seed, name='fc7_dropout')

    if not drop_final_fc:
        with tf.variable_scope('fc8'):
            im = custom_layers.Dense(input=im,
                                     units=num_classes,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=seed),
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
