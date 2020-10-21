import tensorflow as tf
from model_utils import batch_norm_relu
from Layers import layers as custom_layers
from contrastive_loss import add_contrastive_loss
from dataproviders.simclr.data_util import batch_random_blur
from Metrics import observables_config as ocfg

def linear_layer(x,
                 is_training,
                 num_classes,
                 weight_decay=1e-6,
                 alignment=None,
                 use_bias=True,
                 use_bn=False,
                 batch_norm_decay=0.9,
                 seed=None,
                 name='linear_layer',
                 **obs_kwargs):
    """Linear head for linear evaluation.
    Args:
    x: hidden state tensor of shape (bsz, dim).
    is_training: boolean indicator for training or test.
    num_classes: number of classes.
    use_bias: whether or not to use bias.
    use_bn: whether or not to use BN for output units.
    name: the name for variable scope.
    Returns:
    logits of shape (bsz, num_classes)
    """
    assert x.shape.ndims == 2, x.shape
    with tf.variable_scope(name):
        x = custom_layers.Dense(
                input=x,
                units=num_classes,
                use_bias=use_bias and not use_bn,
                kernel_initializer=tf.random_normal_initializer(stddev=.01, seed=seed),
                bias_initializer='zeros',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                alignment=alignment,
                activation=None,
                **obs_kwargs)
        if use_bn:
            x = batch_norm_relu(x, is_training,
                relu=False,
                center=use_bias,
                batch_norm_decay=batch_norm_decay)
        x = tf.identity(x, '%s_out' % name)
    return x

def head_proj(hiddens,
              is_training,
              weight_decay=1e-6,
              alignment=None,
              use_bn=True,
              batch_norm_decay=0.9,
              num_nlh_layers=1,
              head_proj_dim=128,
              seed=None,
              name='head_contrastive',
              **obs_kwargs):

    print('Using alignment strategy {} inside projection head'.format(alignment))
    with tf.variable_scope(name):
        hiddens = linear_layer(x=hiddens,
                               is_training=is_training,
                               num_classes=hiddens.shape[-1],
                               weight_decay=weight_decay,
                               alignment=alignment,
                               use_bias=True,
                               use_bn=use_bn,
                               batch_norm_decay=batch_norm_decay,
                               seed=seed,
                               name='nl_0',
                               **obs_kwargs)
        for j in range(1, num_nlh_layers+1):
            hiddens = tf.nn.relu(hiddens)
            hiddens = linear_layer(x=hiddens,
                                   is_training=is_training,
                                   num_classes=head_proj_dim,
                                   weight_decay=weight_decay,
                                   alignment=alignment,
                                   use_bias=False,
                                   use_bn=use_bn,
                                   batch_norm_decay=batch_norm_decay,
                                   seed=seed,
                                   name='nl_%d'%j,
                                   **obs_kwargs)
    return hiddens

def simclr_func(inputs,
                model,
                num_transforms=2,
                weight_decay=1e-6,
                head_proj_kwargs={},
                con_loss_kwargs={},
                use_blur=True,
                image_height=224,
                image_width=224,
                data_seed=None,
                **model_kwargs):
    '''Wraps a standard imagenet model, drops the final fc and add the 2 layer MLP'''

    gpu_mode = model_kwargs.get('gpu_mode', True)
    if gpu_mode:
        features = inputs['images']
    else:
        features = inputs

    # Split channels, and optionally apply extra batched augmentation.
    features_list = tf.split(features, num_or_size_splits=num_transforms, axis=-1)
    if use_blur and model_kwargs['train']:
        print('Using Gaussian Blur')
        features_list = batch_random_blur(features_list, image_height, image_width, seed=data_seed)
    features = tf.concat(features_list, axis=0)  # (num_transforms * bsz, h, w, c)
    print('SimCLR input feature shape', features.shape)

    if gpu_mode:
        inputs = {'images': features}
    else:
        inputs = features

    model_out = model(inputs, drop_final_fc=True, weight_decay=weight_decay, **model_kwargs)

    if gpu_mode:
        # the other entry is the params
        assert(len(model_out) == 2)
        hiddens = model_out[0]
        params = model_out[1]
    else:
        hiddens = model_out

    # add obs kwargs to head_proj_kwargs
    for k in ocfg.KWARGS:
        head_proj_kwargs[k] = model_kwargs[k]
    hiddens = head_proj(hiddens,
                        is_training=model_kwargs['train'],
                        weight_decay=weight_decay,
                        alignment=model_kwargs.get('alignment', None),
                        seed=model_kwargs['seed'],
                        **head_proj_kwargs)

    loss, logits_con, labels_con = add_contrastive_loss(hiddens, **con_loss_kwargs)
    tf.add_to_collection('SIMCLR_LOSSES', loss)

    if gpu_mode:
        return logits_con, params
    else:
        return logits_con
