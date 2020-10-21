import timeit
import numpy as np
import tensorflow as tf
import operations as op
import math
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
from Metrics import functions as m
from Metrics import observables_config as ocfg

# Custom fully connected layer using feedback alignment
def Dense(input,
          units,
          activation=None,
          use_bias=True,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=None,
          bias_regularizer=None,
          alignment=None,
          obs_frac=None,
          obs_stddev=None,
          obs_seed=0):

    input_dim = int(input.shape[-1])
    units = int(units)
    activation = activations.get(activation)
    bias = None
    if isinstance(kernel_initializer, str):
        kernel_initializer = initializers.get(kernel_initializer)
    if isinstance(bias_initializer, str):
        bias_initializer = initializers.get(bias_initializer)
    if isinstance(kernel_regularizer, str):
        kernel_regularizer = regularizers.get(kernel_regularizer)
    if isinstance(bias_regularizer, str):
        bias_regularizer = regularizers.get(bias_regularizer)

    forward_kernel = tf.get_variable(
        name='forward_kernel',
        shape=[input_dim, units],
        initializer=kernel_initializer,
        regularizer=kernel_regularizer,
        dtype=tf.float32,
        trainable=True)

    backward_kernel = tf.get_variable(
        name='backward_kernel',
        shape=[units, input_dim],
        initializer=kernel_initializer,
        dtype=tf.float32,
        trainable=True)

    if use_bias:
        bias = tf.get_variable(
            name='bias',
            shape=[units,],
            initializer=bias_initializer,
            regularizer=bias_regularizer,
            dtype=tf.float32,
            trainable=True)

    if alignment is None:
        outputs = tf.matmul(input, forward_kernel)
    else:
        alignment(input,
                  forward_kernel=forward_kernel,
                  backward_kernel=backward_kernel,
                  activation=activation,
                  layer_type='fc',
                  bias=bias)
        matmul = op.custom_matmul(tie_grad=alignment.tie_grad)
        outputs = matmul(input, forward_kernel, backward_kernel)

    if use_bias:
        outputs = tf.nn.bias_add(outputs, bias)

    if activation is not None:
        outputs = activation(outputs)

    act_gradient = m.compute_act_gradient(layer_output=outputs,
                                          layer_input=input,
                                          subsample_frac=obs_frac,
                                          noise_stddev=obs_stddev,
                                          seed=obs_seed)

    # Compute observables
    for measure in ocfg.MEASURES:
        if measure == 'weight':
            mtensor = forward_kernel
            obs_ignore_first_dim = False
        elif measure == 'act':
            mtensor = outputs
            obs_ignore_first_dim = True
        else:
            raise NotImplementedError('{} is not implemented'.format(measure))
        mtensor = m.obfuscate_tensor(mtensor,
                                     subsample_frac=obs_frac,
                                     noise_stddev=obs_stddev,
                                     seed=obs_seed,
                                     ignore_first_dim=obs_ignore_first_dim)

        m.stats(mtensor, name=measure)
    m.stats(act_gradient, name='gradavg')

    tf.identity(0.0, name='hebb_angle')

    return outputs

# Custom 2D convolution layer using feedback alignment
# assumes strides=[1,1], padding='valid', data_format='NHWC'
def Conv2D(input,
           filters,
           kernel_size,
           strides = [1,1,1,1],
           padding = "VALID",
           activation=None,
           use_bias=True,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros',
           kernel_regularizer=None,
           bias_regularizer=None,
           alignment=None,
           alignment_relu=False,
           alignment_batchnorm=False,
           obs_frac=None,
           obs_stddev=None,
           obs_seed=0):

    input_dim = int(input.shape[-1])
    kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    kernel_shape = kernel_size + (input_dim, filters)
    activation = activations.get(activation)
    bias = None
    if isinstance(strides, int):
        strides = [1,strides,strides,1]
    if len(strides) == 2:
        strides = [1] + list(strides) + [1]
    if isinstance(kernel_initializer, str):
        kernel_initializer = initializers.get(kernel_initializer)
    if isinstance(bias_initializer, str):
        bias_initializer = initializers.get(bias_initializer)
    if isinstance(kernel_regularizer, str):
        kernel_regularizer = regularizers.get(kernel_regularizer)
    if isinstance(bias_regularizer, str):
        bias_regularizer = regularizers.get(bias_regularizer)

    forward_kernel = tf.get_variable(
            name='forward_kernel',
            shape=kernel_shape,
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            trainable=True)

    backward_kernel = tf.get_variable(
            name='backward_kernel',
            shape=kernel_shape,
            initializer=kernel_initializer,
            trainable=True)

    if use_bias:
        bias = tf.get_variable(
                name='bias',
                shape=[filters,],
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                trainable=True)

    if alignment is None:
        outputs = tf.nn.conv2d(input=input,
                               filter=forward_kernel,
                               strides=strides,
                               padding=padding)
    else:
        if alignment_relu:
            alignment_activation = activations.get("relu")
        else:
            alignment_activation = activation

        alignment(input,
                  forward_kernel=forward_kernel,
                  backward_kernel=backward_kernel,
                  activation=alignment_activation,
                  layer_type='conv',
                  bias=bias,
                  strides=strides,
                  padding=padding)
        conv2d = op.custom_conv2d(strides=strides,
                                  padding=padding,
                                  tie_grad=alignment.tie_grad)
        outputs = conv2d(input, forward_kernel, backward_kernel)

    if use_bias:
        outputs = tf.nn.bias_add(outputs, bias, data_format='NHWC')

    if activation is not None:
        outputs = activation(outputs)

    act_gradient = m.compute_act_gradient(layer_output=outputs,
                                          layer_input=input,
                                          subsample_frac=obs_frac,
                                          noise_stddev=obs_stddev,
                                          seed=obs_seed)

    # Compute observables
    for measure in ocfg.MEASURES:
        if measure == 'weight':
            mtensor = forward_kernel
            obs_ignore_first_dim = False
        elif measure == 'act':
            mtensor = outputs
            obs_ignore_first_dim = True
        else:
            raise NotImplementedError('{} is not implemented'.format(measure))
        mtensor = m.obfuscate_tensor(mtensor,
                                     subsample_frac=obs_frac,
                                     noise_stddev=obs_stddev,
                                     seed=obs_seed,
                                     ignore_first_dim=obs_ignore_first_dim)

        m.stats(mtensor, name=measure)
    m.stats(act_gradient, name='gradavg')

    # Note currently hebb angle is not defined for conv layers
    tf.identity(0.0, name='hebb_angle')
    return outputs
