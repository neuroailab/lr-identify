import numpy as np
import copy
import scipy.stats as stats
from observable_names import ADMISSIBLE_OBS

def cls_dataset_transform(data,
                          num_dims=2,
                          obs_keys=None,
                          obs_measure='all'):
    """Routine for converting the full dataset into one amenable for classifier training (no infinite values).

       Arguments:
           data: The full dataset (3D) with 'obs_keys', 'meta', 'X', 'Y', and if included, 'X_cat', keys
           num_dims: the dimensions of the returned dataset, either 2D (for SVM and Random Forest classifiers) or 3D (for Conv1D MLP)
           obs_keys: The specific sets of observable statistics you want to pass in
           obs_measure: The specific observable measure ('all', 'weights', 'act', 'grad') from which you want all finite statistics from (recommended).

       Returns:
           Classifier amenable training.

    """

    # basic sanity checks
    assert len(data['X'].shape) == 3, 'The dataset passed into this function must be 3D!'
    assert ((num_dims == 2) or (num_dims == 3)), 'The dimensions you have specified {} is not valid!'.format(num_dims)
    if obs_measure is not None:
        assert obs_measure in ADMISSIBLE_OBS.keys(), 'The obs_measure {} you specified does not exist!'.format(obs_measure)
    if obs_keys is None:
        assert obs_measure is not None, 'You must pass in obs_measure if obs_keys is not specified!'
        obs_keys = ADMISSIBLE_OBS[obs_measure]
    if isinstance(obs_keys, str):
        obs_keys = [obs_keys]

    cls_data = {}
    cls_data['obs_keys'] = np.array(obs_keys).astype('U')
    cls_data['meta'] = copy.deepcopy(data['meta'])
    cls_data['Y'] = copy.deepcopy(data['Y'])
    cls_data['X'] = copy.deepcopy(data['X'])
    # subselect the obs keys we want
    sel_obs_idx = [np.where(data['obs_keys'] == k)[0][0] for k in cls_data['obs_keys']]
    cls_data['X'] = cls_data['X'][:, :, sel_obs_idx]
    if len(cls_data['X'].shape) == 2:
        cls_data['X'] = np.expand_dims(cls_data['X'], axis=-1)
    if num_dims == 2:
        # when flattened, the first 21 elements are the first observable's trajectory
        # the next 21 elements are the second observable's trajectory, etc
        cls_data['X'] = np.transpose(cls_data['X'], axes=(0, 2, 1))
        cls_data['X'] = np.reshape(cls_data['X'], (cls_data['X'].shape[0], -1))
    if 'X_cat' in data.keys():
        cls_data['X_cat'] = copy.deepcopy(data['X_cat'])
        if num_dims == 2:
            # because the ternary layer depth repeats across the trajectory
            cls_data['X_cat'] = cls_data['X_cat'][:, 0, :]

    assert np.isfinite(cls_data['X']).all(), 'One or more observables you specified {} should be removed'.format(obs_keys)

    return cls_data

def featurewise_norm(data, fmean=None, fvar=None):
    """perform a whitening-like normalization operation on the data, feature-wise
       Assumes data = (K, M) matrix where K = number of stimuli and M = number of features
    """
    fdata = copy.deepcopy(data)
    if fmean is None:
        fmean = fdata.mean(0)
    if fvar is None:
        fvar = fdata.std(0)
    fdata = fdata - fmean  #subtract the feature-wise mean of the data
    fdata = fdata / np.maximum(fvar, 1e-5)  #divide by the feature-wise std of the data

    return fdata, fmean, fvar


def get_off_diagonal(mat):
    n = mat.shape[0]
    i0, i1 = np.triu_indices(n, 1)
    i2, i3 = np.tril_indices(n, -1)
    return np.concatenate([mat[i0, i1], mat[i2, i3]])


def spearman_brown(uncorrected, multiple):
    numerator = multiple * uncorrected
    denominator = 1 + (multiple - 1) * uncorrected
    return numerator / denominator


def idfunc(x):
    return x


def pearsonr(a, b):
    return stats.pearsonr(a, b)[0]


def spearmanr(a, b):
    return stats.spearmanr(a, b)[0]


def split_half_correlation(datas_by_trial,
                           num_splits,
                           aggfunc=idfunc,
                           statfunc=pearsonr):

    """arguments:
              data_by_trial -- list of (numpy arrays)
                        assumes each is a tensor with structure is (trials, stimuli)
              num_splits (nonnegative integer) how many splits of the data to make
    """

    random_number_generator = np.random.RandomState(seed=0)

    corrvals = []
    for split_index in range(num_splits):
        stats1 = []
        stats2 = []
        for data in datas_by_trial:
            #get total number of trials
            num_trials = data.shape[0]

            #construct a new permutation of the trial indices
            perm = random_number_generator.permutation(num_trials)

            #take the first num_trials/2 and second num_trials/2 pieces of the data
            first_half_of_trial_indices = perm[:num_trials / 2]
            second_half_of_trial_indices = perm[num_trials / 2: num_trials]

            #mean over trial dimension
            s1 = aggfunc(data[first_half_of_trial_indices].mean(axis=0))
            s2 = aggfunc(data[second_half_of_trial_indices].mean(axis=0))
            stats1.extend(s1)
            stats2.extend(s2)

        #compute the correlation between the means
        corrval = statfunc(np.array(stats1),
                           np.array(stats2))
        #add to the list
        corrvals.append(corrval)

    return spearman_brown(np.array(corrvals), 2)
