from resnet_model_v1 import MODEL_PARAMS as modelparamsv1
from resnet_model_v2 import MODEL_PARAMS as modelparamsv2


# MODEL LAYER LISTS

def make_resnet_layer_list(num, v2=False):

    layers = ['conv0']
    block_nums = modelparamsv2[num]['layers'] if v2 else modelparamsv1[num]['layers']
    residual_block = num in [18, 34]

    for blockgroup in range(1, 5):
        for block in range(block_nums[blockgroup - 1]):
            for convlayer in range(0 if block == 0 else 1, 3 if residual_block else 4):
                layername = 'block_{}{}_{}/conv{}'.format('layer' if v2 else 'group', blockgroup, block, convlayer)
                layers.append(layername)
    layers.append('dense')

    return layers


# Resnets
MODEL_LAYERS = {'resnet{}{}'.format(num, 'v2' if v2 else ''): make_resnet_layer_list(num, v2=v2) for num in [18, 34, 50, 101, 152] for v2 in (True, False)}

# Alexnets
ALEXNET_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
MODEL_LAYERS['alexnet'] = ALEXNET_LAYERS
MODEL_LAYERS['alexnetlrn'] = ALEXNET_LAYERS

# Knets
KNET_LAYERS = ['conv1', 'conv2', 'conv3', 'fc4']
KNETC4_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'fc5']
MODEL_LAYERS['knet'] = KNET_LAYERS
MODEL_LAYERS['knetlrn'] = KNET_LAYERS
MODEL_LAYERS['knetc4'] = KNETC4_LAYERS
MODEL_LAYERS['knetc4lrn'] = KNETC4_LAYERS


def group_layers(model_nm,
                 grp_keys=['early', 'middle', 'deep']):

    """For each model, group its layers by grp_keys designation.
    This is to construct the ternary indicator of layer position.
    """

    layer_mapping = {}
    model_layers = MODEL_LAYERS[model_nm.lower()]

    num_layers = len(model_layers)
    num_grps = len(grp_keys)
    grp_size = num_layers // num_grps

    grp_iter = np.arange(0, num_layers, grp_size)
    grp_start_idxs = grp_iter[:num_grps]
    for idx, i in enumerate(grp_start_idxs):
        start_idx = i
        end_idx = (i + grp_size)
        if idx < num_grps - 1:
            curr_layers = model_layers[start_idx:end_idx]
        else:
            curr_layers = model_layers[start_idx:]

        for l in curr_layers:
            layer_mapping[l] = grp_keys[idx]

    """for simclr
    we keep the grouping of base layers the exact same
    we just make the remaining two layers highest,
    in order to keep consistent across multiple tasks"""
    layer_mapping['head_contrastive/nl_0'] = grp_keys[-1]
    layer_mapping['head_contrastive/nl_1'] = grp_keys[-1]
    return layer_mapping
