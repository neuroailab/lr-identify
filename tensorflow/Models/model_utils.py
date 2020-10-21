import tensorflow as tf

def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    center=True, scale=True, batch_norm_decay=0.9):
    """Performs a batch normalization followed by a ReLU.
    Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    center: `bool` whether to add learnable bias factor.
    scale: `bool` whether to add learnable scaling factor.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    Returns:
    A normalized `Tensor` with the same `data_format`.
    """
    # NOTE: we do not do cross tpu batch norm as it requires fused=False which is much slower
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    axis = 1

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=batch_norm_decay,
        epsilon=1e-5,
        center=center,
        scale=scale,
        training=is_training,
        trainable=True,
        fused=True,
        gamma_initializer=gamma_initializer)

    if relu:
        inputs = tf.nn.relu(inputs)
    return inputs