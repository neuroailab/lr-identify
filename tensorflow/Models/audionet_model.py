from audionet.preprocess.tfcochleagram import cochleagram_graph

def cochleagram(audio_signal, gpu_mode=True):
    """
    Using tf cochlear graph for preprocessing the input.
    The first layer.
    """
    COCH_PARAMS = {
         "SR": 20000,
         "ENV_SR":200,
         "HIGH_LIM":8000,
         "LOW_LIM":20,
         "N":50,
         "SAMPLE_FACTOR":4,
         "compression":"clipped_point3",
         "rFFT":True,
         "reshape_kell2018":False,
         "erb_filter_kwargs":{'no_lowpass':True, 'no_highpass':True}
     }

    # use the syntax of cochlegram
    nets = {'input_signal': audio_signal}
    coch_container = cochleagram_graph(nets, **COCH_PARAMS)
    coch_out = nets['cochleagram']

    if gpu_mode:
        output = {'images': coch_out}
    else:
        output = coch_out

    return output

def audionet_func(inputs,
                  model,
                  coch_online=True,
                  **model_kwargs):
    '''Wraps a standard imagenet model and feeds in cochleagram'''

    if coch_online:
        gpu_mode = model_kwargs.get('gpu_mode', True)
        if gpu_mode:
            im = inputs['images']
        else:
            im = inputs

        im = cochleagram(im, gpu_mode=gpu_mode)
    else:
        im = inputs

    model_out = model(im, **model_kwargs)
    return model_out
