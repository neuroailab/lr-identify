from resnet_model_v1 import resnet_v1
from resnet_model_v2 import resnet_v2
from Metrics import observables_config as ocfg

from collections import OrderedDict

#### This is the tfutils wrapper aroung the google resnet
def google_resnet_func(inputs,
                       train=True,
                       resnet_size=50,
                       num_classes=1000,
                       alignment=None,
                       tf_layers=False,
                       use_v2=False,
                       gpu_mode=True,
                       return_endpoints=False,
                       bn_trainable=True,
                       seed=None,
                       weight_decay=1e-4,
                       drop_final_fc=False,
                       **kwargs):

    # get obs kwargs
    obs_kwargs = {}
    for k in ocfg.KWARGS:
        obs_kwargs[k] = kwargs[k]

    params = OrderedDict()
    print("Google Model func: Seeding the model initializers with seed: {}".format(seed))
    if tf_layers:
        print("Using tf.layers")
    else:
        print("Using custom layers")
    if use_v2:
        print("using v2 resnet-", resnet_size, " as model")
        network = resnet_v2(
                resnet_size=resnet_size, num_classes=num_classes,
                alignment=alignment, tf_layers=tf_layers,
                bn_trainable=bn_trainable, seed=seed,
                weight_decay=weight_decay, drop_final_fc=drop_final_fc, **obs_kwargs)
    else:
        print("using v1 resnet-", resnet_size, " as model")
        network = resnet_v1(
                resnet_depth=resnet_size, num_classes=num_classes,
                alignment=alignment, tf_layers=tf_layers,
                bn_trainable=bn_trainable, seed=seed,
                weight_decay=weight_decay, drop_final_fc=drop_final_fc, **obs_kwargs)

    print("gpu_mode set to: {}".format(gpu_mode))
    if gpu_mode:
        print('GPU MODE')
        print('inputs shape', inputs['images'].shape)
        logits = network(
                inputs=inputs['images'], is_training=train,
                return_endpoints=return_endpoints)
        if return_endpoints:
            logits, endpoints = logits
            return logits, endpoints, params
        else:
            print('Logits shape', logits.shape)
            return logits, params
    else:
        print('TPU MODE')
        # TPU mode
        print('inputs shape', inputs.shape)
        logits = network(
                inputs=inputs, is_training=train)
        print('Logits shape', logits.shape)
        return logits
