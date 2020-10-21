import math
import tensorflow as tf
import numpy as np

from observables_config import FUNCTIONS, STATISTICS, make_obs_name

# Calculates various moment statistics of a tensor
def stats(tensor, name='weight'):
    '''
    Statistics are generated from a cross of MEASURES, FUNCTIONS, and STATISTICS
    found in observables_config.py.

    This function receives the measure as input and generates observables from all
    (function, statistic) pairs.
    Deprecated: sem
    '''

    # Apply function to tensor
    for function in FUNCTIONS:
        if function == 'sq':
            ftensor = tf.square(tensor)
        elif function == 'abs':
            ftensor = tf.abs(tensor)
        elif function == 'raw':
            ftensor = tf.identity(tensor)
        else:
            raise NotImplementedError('{} is not implemented'.format(function))

        # If receiving average gradient, save as identity statistic. Else, calculate distribution statistics.
        if name == 'gradavg':
            tf.identity(ftensor, name=make_obs_name(name, function, 'id'))

        else:
            if name == 'weight':            # because weight doesn't have batch dimension
                reduction_axes = list(range(len(ftensor.get_shape().as_list())))
            else:
                reduction_axes = list(range(len(ftensor.get_shape().as_list())))[1:]

            # Calculate and create named variables for each statistic
            for statistic in STATISTICS:

                mean, var = tf.nn.moments(ftensor, axes=reduction_axes, keep_dims=True)
                median = tf.contrib.distributions.percentile(ftensor, 50., axis=reduction_axes, keep_dims=True)

                if statistic == 'norm':
                    stensor = tf.norm(tf.reshape(ftensor, [-1, np.prod(ftensor.get_shape().as_list()[1:])]), ord=2, axis=1)

                elif statistic == 'mean':
                    stensor = tf.squeeze(mean)

                elif statistic == 'median':
                    stensor = tf.squeeze(median)

                elif statistic == 'thirdquartile':
                    stensor = tf.contrib.distributions.percentile(ftensor, 75., axis=reduction_axes)

                elif statistic == 'var':
                    stensor = tf.squeeze(var)

                elif statistic == 'skew':    # Pearson's second coefficient
                    sub = mean - median
                    submul = tf.multiply(sub, 3.)
                    submul = tf.squeeze(submul)
                    reducestd = tf.math.reduce_std(ftensor, axis=reduction_axes)
                    div = tf.math.divide(submul, reducestd)
                    stensor = div

                elif statistic == 'kurtosis':    # coefficient of kurtosis
                    sub = ftensor - mean
                    power = tf.pow(sub, 4)
                    reducemean = tf.math.reduce_mean(power, axis=reduction_axes)
                    varsq = tf.squeeze(tf.square(var))
                    div = tf.math.divide(reducemean, varsq)
                    stensor = div

                else:
                    raise NotImplementedError('{} is not implemented'.format(statistic))

                # name_str = make_obs_name(name, function, statistic)
                # stensor = tf.Print(stensor, [stensor], name_str)
                stensor = tf.identity(stensor, name=make_obs_name(name, function, statistic))

def obfuscate_tensor(tensor,
                     subsample_frac=None,
                     noise_stddev=None,
                     seed=0,
                     ignore_first_dim=True):

    '''Given a tensor as input, will return a subsampled version with added noise.
    ignore_first_dim: For most tensors, the first dimension is the batch dimension, so we do not subsample here.
    Set this to False for weight tensors.'''

    obf_tensor = tf.identity(tensor)
    if noise_stddev is not None:
        obf_tensor = obf_tensor + tf.random_normal(shape=tf.shape(obf_tensor),
                                                   mean=0.0,
                                                   stddev=noise_stddev,
                                                   seed=seed,
                                                   dtype=tf.float32)
        print('Added noise with standard deviation {}, using a seed of {}'.format(noise_stddev, seed))

    if subsample_frac is not None:
        if ignore_first_dim:
            num_units = np.prod(obf_tensor.get_shape().as_list()[1:])
            flat_obf_tensor = tf.reshape(obf_tensor, [-1, num_units])
        else:
            print('Including first dimension in subsampling')
            # we flatten the entire tensor in this case
            num_units = np.prod(obf_tensor.get_shape().as_list())
            flat_obf_tensor = tf.reshape(obf_tensor, [num_units])
        perm_inds = np.random.RandomState(seed=seed).permutation(num_units)
        # we np.ceil so that the minimum number of units is 1
        num_subsample_units = (int)(np.ceil(subsample_frac * num_units))
        subsample_indices = tf.constant(perm_inds[:num_subsample_units], dtype=tf.int32)
        if ignore_first_dim:
            # we now do the tensorflow equivalent of numpy's flat_obf_tensor[:, subsample_indices]
            selected = tf.gather(tf.transpose(flat_obf_tensor, [1, 0]), subsample_indices)
            obf_tensor = tf.transpose(selected, [1, 0])
        else:
            obf_tensor = tf.gather(flat_obf_tensor, subsample_indices)
            # add extra dimension of 1 to batch dimension to be able to compute norm in the statistics
            obf_tensor = tf.expand_dims(obf_tensor, axis=0)
        print('Subsampled fraction {} of {} total units, resulting in {} units, using a seed of {}'.format(subsample_frac, num_units, num_subsample_units, seed))

    return obf_tensor


def compute_act_gradient(layer_output,
                         layer_input,
                         subsample_frac=None,
                         noise_stddev=None,
                         seed=0):
    '''Computes (sum_{layer_input} dlayer_output/dlayer_input) averaged over layer_output units.'''

    inputs_shape, outputs_shape = layer_input.get_shape().as_list(), layer_output.get_shape().as_list()
    flattened_output_shape = np.prod(outputs_shape[1:])
    y = tf.reshape(layer_output, [-1, flattened_output_shape])
    # we use a different seed for the postsynaptic activity than presynaptic,
    # since realistically we don't expect to sample the same units for inputs and outputs with the same number of units
    y = obfuscate_tensor(y,
                         subsample_frac=subsample_frac,
                         noise_stddev=noise_stddev,
                         seed=(seed+1))
    J = tf.gradients(y, layer_input)[0]
    J = tf.reshape(J, [-1, np.prod(J.get_shape().as_list()[1:])])
    J = obfuscate_tensor(J,
                         subsample_frac=subsample_frac,
                         noise_stddev=noise_stddev,
                         seed=seed)
    act_gradient_unnorm = tf.reduce_sum(J, axis=1)
    num_output_units = flattened_output_shape
    if subsample_frac is not None:
        # we np.ceil so that the minimum number of units is 1
        num_output_units = (int)(np.ceil(subsample_frac*flattened_output_shape))
    act_gradient = tf.math.divide(act_gradient_unnorm, tf.dtypes.cast(num_output_units, tf.float32))
    return act_gradient
