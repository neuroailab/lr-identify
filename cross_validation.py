"""
library of routines for cross validation
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from itertools import product
import operator

import metrics
from utils import featurewise_norm

def iterate_dicts(inp):
    '''Computes cartesian product of parameters
    From: https://stackoverflow.com/questions/10472907/how-to-convert-dictionary-into-string'''
    return list((dict(zip(inp.keys(), values)) for values in product(*inp.values())))

def dict_to_str(adict):
    '''Converts a dictionary (e.g. hyperparameter configuration) into a string'''
    return ''.join('{}{}'.format(key, val) for key, val in sorted(adict.items()))

def internal_model_call(model_class, cls_kwargs, override_internal_lbl=False):
    if override_internal_lbl:
        try:
            curr_model = model_class(use_internal_lbl_map=False, **cls_kwargs)
        except:
            print('This model does not create an internal label map, so nothing to override')
            curr_model = model_class(**cls_kwargs)
    else:
        curr_model = model_class(**cls_kwargs)

    return curr_model

def get_model_idx(key):
    model_prefix = key.split('_')[0]
    model_idx = (int)(model_prefix.split('model')[-1])
    return model_idx

def run_cv(model_class,
           X,
           Y,
           gridcv_params,
           X_cat=None,
           override_internal_lbl=False,
           n_splits=5,
           verbose=False,
           **gridcv_args):

    # generate stratified splits
    skf = StratifiedKFold(n_splits=n_splits, **gridcv_args)
    skf.get_n_splits(X, Y)

    if not isinstance(model_class, list):
        model_class = [model_class]

    if not isinstance(gridcv_params, list):
        gridcv_params = [gridcv_params]

    assert(len(model_class) == len(gridcv_params))

    cv_res = {}
    cv_res_kwargs = {}
    test_eval_bs_arr = []
    for model_class_idx, curr_model_class in enumerate(model_class):
        iter_cls_kwargs = iterate_dicts(gridcv_params[model_class_idx])

        # set eval batch sizes to None temporarily to avoid problem with val set not evenly dividing what was originally set for the test set
        # this will use all val set examples which is most efficient and likely will fit into gpu memory since the val set is a small subset of the train set
        eval_batch_size_keys = [k for k in iter_cls_kwargs[0].keys() if 'eval_batch_size' in k]
        test_eval_bs_dict = None
        if len(eval_batch_size_keys) > 0: # only tensorflow classifiers will have this, either as 'eval_batch_size' or 'cls__eval_batch_size'
            assert(len(eval_batch_size_keys) == 1)
            eval_bs_k = eval_batch_size_keys[0]
            test_eval_bs = iter_cls_kwargs[0][eval_bs_k] # eval batch size will be the same for a given model class across all cv param dicts
            if test_eval_bs is not None:
                test_eval_bs_dict = {eval_bs_k: test_eval_bs}
        test_eval_bs_arr.append(test_eval_bs_dict)

        for curr_cls_kwargs in iter_cls_kwargs:
            # override current cls kwargs eval batch size if it exists
            if test_eval_bs_dict is not None: # the original model kwargs have an eval batch size
                curr_cls_kwargs[test_eval_bs_dict.keys()[0]] = None
                if verbose:
                    print('Setting val set batch size to use all val examples')

            curr_model = internal_model_call(model_class=curr_model_class,
                                             cls_kwargs=curr_cls_kwargs,
                                             override_internal_lbl=override_internal_lbl)
            curr_key = 'model{}_'.format(model_class_idx) + dict_to_str(curr_cls_kwargs)
            cv_res_kwargs[curr_key] = curr_cls_kwargs
            curr_cv_results = []
            for train_index, val_index in skf.split(X, Y):
                X_train, X_val = X[train_index], X[val_index]
                Y_train, Y_val = Y[train_index], Y[val_index]
                X_cat_train, X_cat_val = None, None
                if X_cat is not None:
                    X_cat_train, X_cat_val = X_cat[train_index], X_cat[val_index]

                try:
                    if X_cat is None:
                        curr_model.fit(X_train, Y_train)
                        curr_acc_score = curr_model.score(X_val, Y_val)
                    else:
                        curr_model.fit(X_train, Y_train, X_cat=X_cat_train)
                        curr_acc_score = curr_model.score(X_val, Y_val, X_cat=X_cat_val)
                    if verbose:
                        print('Score: {}. Num train examples: {}. Num val examples: {}'.format(curr_acc_score, X_train.shape[0], X_val.shape[0]))
                    curr_cv_results.append(curr_acc_score)
                except:
                    if verbose:
                        print('{} failed to train'.format(curr_key))

            # average across cross val test splits
            if len(curr_cv_results) > 0:
                curr_cv_mean_acc = np.mean(curr_cv_results)
                cv_res[curr_key] = curr_cv_mean_acc

    # pick the hyperparameter with the highest validation accuracy
    max_key = max(cv_res.items(), key=operator.itemgetter(1))[0]
    if verbose:
        print('Best hyperparameter configuration: {}'.format(max_key))
    cv_res_best_params = cv_res_kwargs[max_key]
    cv_res_best_model_idx = get_model_idx(max_key)

    # restore original test batch size if we indeed overwrote it during validation
    assert(len(model_class) == len(test_eval_bs_arr))
    original_test_eval_bs_params = test_eval_bs_arr[cv_res_best_model_idx]
    if original_test_eval_bs_params is not None:
        cv_res_best_params.update(original_test_eval_bs_params)

    # instantiate the best model
    cv_res_best_model = internal_model_call(model_class=model_class[cv_res_best_model_idx],
                                         cls_kwargs=cv_res_best_params,
                                         override_internal_lbl=override_internal_lbl)
    return cv_res_best_model, cv_res_best_params


def get_possible_inds(metadata, filter):
    inds = np.arange(len(metadata))
    if filter is not None:
        subset = np.array(list(map(filter, metadata))).astype(np.bool)
        inds = inds[subset]
    return inds


def get_splits(metadata,
               split_by_func,
               num_splits,
               num_per_class_test,
               num_per_class_train,
               train_filter=None,
               test_filter=None,
               seed=0):
    """
    construct a consistent set of splits for cross validation

    arguments:
        metadata: numpy.rec.array of metadata
        split_by_func: callable, returns label for spliting data into balanced categories
                       when applied to metadata
        num_per_class_test: number of testing examples for each unique
                            split_by category
        num_per_class_train: number of train examples for each unique
                            split_by category
        train_filter: callable (or None): specifying which subset of the data
                 to use in training applied on a per-element basis to metadata
        test_filter: callable (or None): specifying which subset of the data
                 to use in testing applied on a per-element basis to metadata
        seed: seed for random number generator
    """

    #filter the data by train and test filters
    train_inds = get_possible_inds(metadata, train_filter)
    test_inds = get_possible_inds(metadata, test_filter)

    #construct possibly category labels for balancing data
    labels = split_by_func(metadata)
    #for later convenience, get unique values of splitting labels in train and test data
    unique_train_labels = np.unique(labels[train_inds])
    unique_test_labels = np.unique(labels[test_inds])

    #seed the random number generator
    rng = np.random.RandomState(seed=seed)

    #construct the splits one by one
    splits = []
    for _split_ind in range(num_splits):
        #first construct the testing data
        actual_test_inds = []
        #for each possible test label
        for label in unique_test_labels:
            #look at all possible stimuli with this label
            possible_test_inds_this_label = test_inds[labels[test_inds] == label]
            #count how many there are
            num_possible_test_inds_this_label = len(possible_test_inds_this_label)
            #make sure there are enough
            err_msg = 'You requested %s per test class but there are only %d available' % (
                    num_per_class_test, num_possible_test_inds_this_label)
            assert num_possible_test_inds_this_label >= num_per_class_test, err_msg
            #select num_per_class_test random examples
            perm = rng.permutation(num_possible_test_inds_this_label)
            actual_test_inds_this_label = possible_test_inds_this_label[
                                                      perm[ :num_per_class_test]]
            actual_test_inds.extend(actual_test_inds_this_label)
        actual_test_inds = np.sort(actual_test_inds)

        #now, since the pools of possible train and test data overlap,
        #but since we don't want the actual train and data examples to overlap at all,
        #remove the chosen test examples for this split from the pool of possible
        #train examples for this split
        remaining_available_train_inds = np.unique(list(set(
                           train_inds).difference(actual_test_inds)))

        #now contruct the train portion of the split
        #basically the same way as for the testing examples
        actual_train_inds = []
        for label in unique_train_labels:
            _this_label = labels[remaining_available_train_inds] == label
            possible_train_inds_this_label = remaining_available_train_inds[_this_label]
            num_possible_train_inds_this_label = len(possible_train_inds_this_label)
            err_msg = 'You requested %s per train class but there are only %d available' % (
                  num_per_class_train, num_possible_train_inds_this_label)
            assert num_possible_train_inds_this_label >= num_per_class_train, err_msg
            perm = rng.permutation(num_possible_train_inds_this_label)
            actual_train_inds_this_label = possible_train_inds_this_label[
                                                      perm[ :num_per_class_train]]
            actual_train_inds.extend(actual_train_inds_this_label)
        actual_train_inds = np.sort(actual_train_inds)

        split = {'train': actual_train_inds, 'test': actual_test_inds}
        splits.append(split)

    return splits


def validate_splits(splits, labels):
    train_classes = np.unique(labels[splits[0]['train']])
    for split in splits:
        train_inds = split['train']
        test_inds = split['test']
        assert set(train_inds).intersection(test_inds) == set([])
        train_labels = labels[split['train']]
        test_labels = labels[split['test']]
        assert (np.unique(train_labels) == train_classes).all()
        assert set(test_labels) <= set(train_classes)
    return train_classes


def train_and_test_classifier(features,
                             labels,
                             splits,
                             model_class,
                             model_args=None,
                             override_internal_lbl=False,
                             gridcv_params=None,
                             gridcv_args=None,
                             fit_args=None,
                             feature_norm=False,
                             X_cat=None,
                             return_models=False,
                             split_idxs=None,
                             index_to_label=None,
                             verbose=False
                            ):
    """Routine for constructing, training and testing correlation classifier

       Arguments:
           features: (K, M) feature array where K = number of stimuli and M = number of features
           labels: length-K vector of labels to be predicted
           index_to_label: dictionary mapping from label indices to label string names
           splits: splits of data (constructed by calling the get_splits function)
           split_idxs: If not None, runs particular set of splits rather than all of them. Useful for job parallelization.
           model_class: the actual live python object that is the classifier "class" object. This can be a list if doing model selection via cross validation.
           model_args: dictionary of arguments for instantiating the classifier class object
           gridcv_params: dictionary of params for applying gridSearch cross-validation to
           gridcv_args: additional arguments to the GridSearcCV construction function
           fit_args: additional arguments to send to the model's fit method during fitting
           feature_norm: apply featurewise_norm
           X_cat: (K, M) feature array consisting of only the categorical features. Use this only if you want to separate them from the continuous ones (for e.g. feature normalization and PCA).
           return_models: return actual trained models for each split

       Returns:
           dictionary summary of training and testing results

    """
    train_confmats = []
    test_confmats = []

    if model_args is None:
        model_args = {}
    if fit_args is None:
        fit_args = {}
    if override_internal_lbl:
        print('CAUTION: Overriding internal label, this may cause mismatches if subselecting learning rules')

    training_sidedata = []
    train_classes = validate_splits(splits, labels)

    models = []
    cv_params = []

    if split_idxs is not None:
        if not isinstance(split_idxs, list):
            split_idxs = [split_idxs]

        splits_torun = []
        for _split_idx in split_idxs:
            splits_torun.append(splits[_split_idx])
    else:
        splits_torun = splits

    for split in splits_torun:
        train_inds = split['train']
        test_inds = split['test']
        train_features = features[train_inds]
        train_labels = labels[train_inds]
        test_features = features[test_inds]
        test_labels = labels[test_inds]
        train_X_cat = None
        test_X_cat = None
        if X_cat is not None:
            train_X_cat = X_cat[train_inds]
            test_X_cat = X_cat[test_inds]

        if feature_norm:
            train_features, fmean, fvar = featurewise_norm(train_features)
            sidedata = {'fmean': fmean, 'fvar': fvar}
            training_sidedata.append(sidedata)

        if gridcv_params is not None:
            if gridcv_args is None:
                gridcv_args = {}

            model, model_cv_best_params = run_cv(model_class=model_class,
                                                 X=train_features,
                                                 Y=train_labels,
                                                 X_cat=train_X_cat,
                                                 gridcv_params=gridcv_params,
                                                 override_internal_lbl=override_internal_lbl,
                                                 verbose=verbose,
                                                 **gridcv_args)
        else:
            # here we instantiate the general classifier, whatever it is
            # we do not accept lists of classifiers unless we are cross validating
            assert(not isinstance(model_class, list))
            model = internal_model_call(model_class=model_class,
                                         cls_kwargs=model_args,
                                         override_internal_lbl=override_internal_lbl)

        if X_cat is None:
            model.fit(train_features, train_labels, **fit_args)
        else:
            model.fit(train_features, train_labels, X_cat=train_X_cat, **fit_args)
        if gridcv_params is not None:
            cv_params.append(model_cv_best_params)

        if X_cat is None:
            train_predictions = model.predict(train_features)
        else:
            train_predictions = model.predict(train_features, X_cat=train_X_cat)

        # convert the internal labels (0,...,num_classes-1) back to the labels
        if (hasattr(model, '_convert_to_labels')) and (not override_internal_lbl):
            train_predictions = model._convert_to_labels(train_predictions)

        train_confmat = metrics.get_confusion_matrix(train_predictions,
                                                     train_labels,
                                                     train_classes)
        train_confmats.append(train_confmat)

        if feature_norm:
            test_features, _ignore, _ignore = featurewise_norm(test_features,
                                                               fmean=fmean,
                                                               fvar=fvar)

        if X_cat is None:
            test_predictions = model.predict(test_features)
        else:
            test_predictions = model.predict(test_features, X_cat=test_X_cat)

        # convert the internal labels (0,...,num_classes-1) back to the labels
        if (hasattr(model, '_convert_to_labels')) and (not override_internal_lbl):
            test_predictions = model._convert_to_labels(test_predictions)

        test_confmat = metrics.get_confusion_matrix(test_predictions,
                                                    test_labels,
                                                    train_classes)
        test_confmats.append(test_confmat)

        models.append(model)

    train_confmats = np.array(train_confmats)
    train_results = metrics.evaluate_results(confmats=train_confmats,
                                             labels=train_classes,
                                             index_to_label=index_to_label)
    test_confmats = np.array(test_confmats)
    test_results = metrics.evaluate_results(confmats=test_confmats,
                                            labels=train_classes,
                                            index_to_label=index_to_label)
    results = {'train': train_results,
               'test': test_results,
               'cv_params': cv_params,
               'training_sidedata': training_sidedata}

    results['feature_importances_by_split'] = []
    for m in models:
        try:
            results['feature_importances_by_split'].append(m.feature_importances)
        except:
            if verbose:
                print('This model does not have feature importances, so nothing to save')
            results['feature_importances_by_split'].append(None)

    if return_models:
        results['models'] = models
    return results, train_classes

