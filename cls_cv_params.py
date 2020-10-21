import numpy as np
from functools import partial
from itertools import product
from fit_pipeline import PipelineClassifier

def return_cls(cls_type):
    if cls_type == 'randomforest':
        classifier_type = partial(PipelineClassifier, estimators=[('cls', 'randomforest')])
        cv_param_grid = {'cls__max_features': ['sqrt', 'log2', None], 'cls__n_estimators': [20, 50, 100, 500, 1000]}
    elif cls_type == 'svm':
        # we allow for pca preprocessing in this case
        classifier_type = [partial(PipelineClassifier, estimators=[('reduce_dim', 'passthrough'), ('cls', 'svm')]),
                           partial(PipelineClassifier, estimators=[('reduce_dim', 'pca'), ('cls', 'svm')])]
        pca_n_components = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600]
        svm_C_vals = [1.0, 5e1, 5e2, 5e3, 5e4, 5e5, 5e6]
        nopca_grid = {'cls__C': svm_C_vals}
        wpca_grid = {'reduce_dim__n_components': pca_n_components, 'cls__C': svm_C_vals}
        cv_param_grid = [nopca_grid, wpca_grid]
    elif cls_type == 'conv1d':
        conv1d_num_layers = 1
        ksizes_frac = [3e-3, 7e-3, 5e-2, 0.25, 0.5, 1.0]
        ksizes = [(int)(np.ceil(f*21)) for f in ksizes_frac]
        ksizes = list(np.unique(ksizes)) # in case any ksize frac rounds up to the same integer value
        strides = [1, 2, 4]
        num_filters = [20, 40]
        paddings = ['valid']
        nopca_grid = {'cls__init_lr': [1e-3, 1e-4],
                      'cls__max_epochs': [400],
                      'cls__ksizes': [list(e) for e in product(ksizes, repeat=conv1d_num_layers)],
                      'cls__train_batch_size': [512, 1024],
                      'cls__eval_batch_size': [None], # loads entire batch size into gpu memory during validation
                      'cls__num_filters': [list(e) for e in product(num_filters, repeat=conv1d_num_layers)],
                      'cls__strides': [list(e) for e in product(strides, repeat=conv1d_num_layers)],
                      'cls__paddings': [list(e) for e in product(paddings, repeat=conv1d_num_layers)],
                      'cls__pool_type': [None, 'max', 'avg'],
                      'cls__weight_decay': [1e-4, 0.0]}
        cv_param_grid = [nopca_grid]
        classifier_type = [partial(PipelineClassifier, estimators=[('reduce_dim', 'passthrough'), ('cls', 'conv1d')])]
    else:
        raise ValueError

    return classifier_type, cv_param_grid
