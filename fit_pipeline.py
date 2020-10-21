import sklearn
from sklearn import svm as sklearn_svm
from sklearn import ensemble as sklearn_ensemble
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, make_scorer
from collections import OrderedDict
import numpy as np
import copy
from cross_validation import get_splits, validate_splits, train_and_test_classifier
from observable_names import DEFAULT_GRP_KEYS
from utils import cls_dataset_transform

def estimator_from_str(cls_name):
    if cls_name.lower() == 'randomforest':
        cls_func = sklearn_ensemble.RandomForestClassifier
    elif cls_name.lower() == 'logistic':
        cls_func = sklearn.linear_model.LogisticRegression
    elif cls_name.lower() == 'svm':
        cls_func = sklearn_svm.LinearSVC
    elif cls_name.lower() == 'pca':
        cls_func = PCA
    elif cls_name.lower() == 'conv1d':
        cls_func = Conv1DClassifier
    else:
        raise ValueError

    return cls_func

class ClassifierBase(object):
    '''Abstract object representing a classifier'''
    def build_label_mapping(self, labels):
        self._label2index = OrderedDict()
        for label in labels:
            if label not in self._label2index:
                self._label2index[label] = (max(self._label2index.values()) + 1) if len(self._label2index) > 0 else 0
        self._index2label = OrderedDict((index, label) for label, index in self._label2index.items())

    def _convert_to_indices(self, labels):
        if (not hasattr(self, '_label2index')) or (self._label2index is None): # e.g. index2label was passed in through the classifier rather than through the build_label_mapping function
            self._label2index = OrderedDict((label, index) for index, label in self._index2label.items())
        indices = []
        for label in labels:
            indices.append(self._label2index[label])
        return np.array(indices)

    def _convert_to_labels(self, pred):
        pred_labels = []
        for i in range(pred.shape[0]):
            curr_label = self._index2label[pred[i]]
            pred_labels.append(curr_label)
        return np.array(pred_labels)

    def fit(self, X, Y, X_cat=None):
        raise NotImplementedError("Abstract method")

    def predict_proba(self, X, X_cat=None):
        raise NotImplementedError("Abstract method")

    def predict(self, X, X_cat=None):
        if X_cat is None:
            proba = self.predict_proba(X)
        else:
            proba = self.predict_proba(X, X_cat=X_cat)
        return np.argmax(proba, axis=-1)

    def score(self, X, Y, X_cat=None):
        raise NotImplementedError("Abstract method")

class TFClassifier(ClassifierBase):
    def __init__(self,
                 num_classes=None,
                 init_lr=1e-4,
                 max_epochs=200,
                 train_batch_size=64,
                 eval_batch_size=None,
                 gpu_options=None,
                 index2label=None,
                 use_internal_lbl_map=True,
                 **model_func_kwargs):

        self._num_classes = num_classes
        self._lr = init_lr
        self._max_epochs = max_epochs
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._model_func_kwargs = model_func_kwargs
        self._gpu_options = gpu_options
        self._graph = None
        self._lr_ph = None
        self._opt = None
        self._index2label = index2label
        self._use_internal_lbl_map = use_internal_lbl_map

    def setup(self):
        import tensorflow as tf
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._lr_ph = tf.placeholder(dtype=tf.float32)
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

    def initializer(self, kind='xavier', *args, **kwargs):
        import tensorflow as tf
        if kind == 'xavier':
            init = tf.contrib.layers.xavier_initializer(*args, **kwargs)
        else:
            init = getattr(tf, kind + '_initializer')(*args, **kwargs)
        return init

    def model_func(self, *args, **kwargs):
        raise NotImplementedError

    def _iterate_minibatches(self, inputs, targets=None, batchsize=240, shuffle=False):
        """
        Iterates over inputs with minibatches, useful if you are doing SGD, like in a TensorflowClassifier
        :param inputs: input dataset, first dimension should be examples
        :param targets: [n_examples, ...] response values, first dimension should be examples
        :param batchsize: batch size
        :param shuffle: flag indicating whether to shuffle the data while making minibatches
        :return: minibatch of (X, Y)
        """
        if isinstance(inputs, dict):
            input_len = inputs[list(inputs.keys())[0]].shape[0]
        else:
            input_len = inputs.shape[0]
        if shuffle:
            indices = np.arange(input_len)
            np.random.shuffle(indices)

        # such that every example has the same batch size, in case the batch size needs to be specified for certain ops
        final_input_len = input_len - (input_len % batchsize)
        if shuffle:
            final_indices = indices[:final_input_len]
        for start_idx in range(0, final_input_len, batchsize):
            if shuffle:
                excerpt = final_indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

            if isinstance(inputs, dict):
                input_yield = {k: inputs[k][excerpt] for k in inputs.keys()}
            else:
                input_yield = inputs[excerpt]

            if targets is None:
                yield input_yield
            else:
                yield input_yield, targets[excerpt]

    def _make_graph(self):
        """
        Makes the temporal mapping function computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.variable_scope('cls'):
                self._predictions = self.model_func(input=self._input_placeholder,
                                                    num_classes=self._num_classes,
                                                    **self._model_func_kwargs)

    def _make_loss(self):
        """
        Makes the loss computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.variable_scope('loss'):

                logits = self._predictions

                self.classification_error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._target_placeholder))
                all_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                if len(all_reg_losses) > 0:
                    self.reg_loss = tf.add_n(all_reg_losses)
                else:
                    self.reg_loss = tf.constant(0.0)

                self.total_loss = self.classification_error + self.reg_loss
                self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                                   global_step=tf.train.get_or_create_global_step())

    def _init_graph(self, X, Y):
        """
        Initializes the graph
        :param X: input data
        """
        import tensorflow as tf
        assert len(Y.shape) == 1
        if self._num_classes is None:
            self._num_classes = len(np.unique(Y))

        with self._graph.as_default():
            self._input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
            self._target_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
            # Build the model graph
            self._make_graph()
            self._make_loss()

            # initialize graph
            init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess = tf.Session(
                config=tf.ConfigProto(gpu_options=self._gpu_options) if self._gpu_options is not None else None)
            self._sess.run(init_op)

    def fit(self, X, Y):
        """
        Fits the parameters to the data
        :param X: Source data, first dimension is examples
        :param Y: Target data, first dimension is examples
        """
        from tqdm import tqdm

        cls_input = copy.deepcopy(X)
        cls_labels = copy.deepcopy(Y)
        assert(self._train_batch_size <= cls_input.shape[0])
        if self._use_internal_lbl_map:
            if self._index2label is None:
                self.build_label_mapping(Y)
            cls_labels = self._convert_to_indices(Y)

        self.setup()
        with self._graph.as_default():
            self._init_graph(cls_input, cls_labels)
            lr = self._lr
            for epoch in tqdm(range(self._max_epochs), desc=' epochs'):
                for counter, batch in enumerate(
                        self._iterate_minibatches(cls_input, cls_labels, batchsize=self._train_batch_size, shuffle=True)):

                    feed_dict = {self._input_placeholder: batch[0],
                                 self._target_placeholder: batch[1],
                                 self._lr_ph: lr}

                    _, loss_value, reg_loss_value = self._sess.run([self.train_op, self.classification_error, self.reg_loss],
                                                                   feed_dict=feed_dict)

    def predict_proba(self, X):
        import tensorflow as tf
        cls_input = copy.deepcopy(X)
        # we make copy to use for different test sets but a single trained model
        internal_eval_batch_size = self._eval_batch_size
        if internal_eval_batch_size is None:
            internal_eval_batch_size = cls_input.shape[0]
        assert(cls_input.shape[0] % internal_eval_batch_size == 0)
        with self._graph.as_default():
            preds = []
            for batch in self._iterate_minibatches(cls_input, batchsize=internal_eval_batch_size, shuffle=False):
                feed_dict = {self._input_placeholder: batch}
                preds.append(np.squeeze(self._sess.run([tf.nn.softmax(self._predictions)], feed_dict=feed_dict)))
            proba = np.concatenate(preds, axis=0)
        return proba

    def score(self, X, Y):
        pred = self.predict(X)
        pred_labels = pred
        if self._use_internal_lbl_map:
            pred_labels = self._convert_to_labels(pred)
        return accuracy_score(y_true=Y, y_pred=pred_labels, normalize=True)

    def close(self):
        """
        Closes occupied resources
        """
        import tensorflow as tf
        tf.reset_default_graph()
        self._sess.close()

class Conv1DClassifier(TFClassifier):
    def model_func(self,
                   input,
                   num_classes,
                   ksizes=[],
                   num_filters=[],
                   strides=[],
                   paddings=[],
                   activation_conv='relu',
                   pool_type=None,
                   weight_decay=None):

        import tensorflow as tf

        # batch, time, num_features
        assert(len(input.get_shape().as_list()) == 3)

        if activation_conv is not None:
            activation_conv = getattr(tf.nn, activation_conv)

        if weight_decay is None:
            weight_decay = 0.

        if not isinstance(ksizes, list):
            ksizes = [ksizes]

        if not isinstance(num_filters, list):
            num_filters = [num_filters]

        assert(len(ksizes) == len(num_filters))
        assert(len(num_filters) == len(strides))
        assert(len(strides) == len(paddings))
        num_conv_layers = len(ksizes)

        curr_out = input
        for layer_idx in list(range(num_conv_layers)):
            curr_out = tf.layers.conv1d(curr_out,
                            filters=num_filters[layer_idx],
                            kernel_size=ksizes[layer_idx],
                            strides=strides[layer_idx],
                            padding=paddings[layer_idx],
                            data_format='channels_last',
                            activation=activation_conv,
                            use_bias=True,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            trainable=True,
                            name='conv1dcls/conv_{}'.format(layer_idx))

        # (batch, new_time, num_filters) --> (batch, num_filters)
        if pool_type == 'max':
            curr_pool_out = tf.reduce_max(curr_out, axis=1, keepdims=False)
        elif pool_type == 'avg':
            curr_pool_out = tf.reduce_mean(curr_out, axis=1, keepdims=False)
        else:
            curr_pool_out = tf.reshape(curr_out, [-1, np.prod(curr_out.get_shape().as_list()[1:])])

        output = tf.contrib.layers.fully_connected(
                    curr_pool_out,
                    num_outputs=int(num_classes),
                    activation_fn=None,
                    weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    trainable=True,
                    scope="conv1dcls/fc_final")
        return output

class PipelineClassifier(ClassifierBase):
    '''Allows you to chain multiple estimators together -- similar to Sklearn Pipeline method, but more general as it handles TensorFlow ops and transitions between CPU and GPU.
    Assumes that the estimators all have fit and predict methods.

    estimators: a list of tuples of [('estimator_1name', estimator_1func), ('estimator_2name', estimator_2func)].
    estimators_kwargs is a dictionary of the form {'estimator_1name__kwarg1': value, 'estimator_1name__kwarg2': value, 'estimator_2name__kwarg1': value, etc}'''
    def __init__(self,
                 estimators,
                 index2label=None,
                 use_internal_lbl_map=True,
                 verbose=False,
                 **estimators_kwargs):

        self._index2label = index2label
        self._use_internal_lbl_map = use_internal_lbl_map
        self._verbose = verbose
        self._estimators = self._make_list(estimators)
        self.estimators_kwargs = estimators_kwargs if estimators_kwargs is not None else {}
        self.estimators = []
        for estimator_name, estimator_func in self._estimators:
            if isinstance(estimator_func, str) and (estimator_func.lower() == 'passthrough'):
                estimator_call = estimator_func
            else:
                if isinstance(estimator_func, str):
                    estimator_func = estimator_from_str(cls_name=estimator_func)
                estimator_kwargs = self._gather_estimator_kwargs(estimator_name=estimator_name, overall_kwarg_dict=self.estimators_kwargs)
                # we override the internal label map of the estimator when we can in order to keep things consistent, since we will create it outside in our own fit method
                try:
                    estimator_call = estimator_func(use_internal_lbl_map=False, **estimator_kwargs)
                except:
                    if self._verbose:
                        print('Estimator {} does NOT have an internal label map to override.'.format(estimator_name))
                    estimator_call = estimator_func(**estimator_kwargs)
            self.estimators.append((estimator_name, estimator_call))

        # useful for random forest to visualize feature importances
        self.feature_importances = None

    def _make_list(self, kwarg):
        if not isinstance(kwarg, list):
            kwarg = [kwarg]
        return kwarg

    def _gather_estimator_kwargs(self,
                           estimator_name,
                           overall_kwarg_dict):

        '''Gather the kwargs for a particular named estimator'''
        estimator_kwargs = {}
        for k,v in overall_kwarg_dict.items():
            k_arr = k.split('__')
            assert(len(k_arr) == 2)
            if k_arr[0] == estimator_name:
                estimator_kwargs[k_arr[1]] = v

        return estimator_kwargs

    def _inference(self, X, Y=None, X_cat=None):
        estimator_input = copy.deepcopy(X)
        X_cat = copy.deepcopy(X_cat)
        for estimator_idx, estimator_tuple in enumerate(self.estimators):
            estimator_name, estimator_obj = estimator_tuple
            if isinstance(estimator_obj, str):
                # the only allowed strings are passthroughs
                assert(estimator_obj.lower() == 'passthrough')
            else:
                if hasattr(estimator_obj, 'transform'):
                    if type(estimator_obj).__name__.lower() == 'pca':
                        # PCA accepts 2D input only
                        if len(estimator_input.shape) > 2:
                            estimator_input = np.reshape(estimator_input, (estimator_input.shape[0], np.prod(estimator_input.shape[1:])))
                    else:
                        if len(estimator_input.shape) == 2:
                            # we assume other methods may operate on original 3D shape (e.g. TCA)
                            estimator_input = np.expand_dims(estimator_input, axis=-1)

                    if Y is not None:
                        # we need to fit the actual object, not merely the estimator_obj reference to it
                        self.estimators[estimator_idx][1].fit(estimator_input)
                    estimator_input = self.estimators[estimator_idx][1].transform(estimator_input)
                else:
                    # classifier is final layer
                    assert(estimator_idx == (len(self.estimators) - 1))
                    if (len(estimator_input.shape) > 2) and (type(estimator_obj).__name__.lower() != 'conv1dclassifier'):
                        estimator_input = np.reshape(estimator_input, (estimator_input.shape[0], np.prod(estimator_input.shape[1:])))
                    elif (len(estimator_input.shape) == 2) and (type(estimator_obj).__name__.lower() == 'conv1dclassifier'):
                        # we add a 1 in the feature dimension to have a 3D input shape after PCA
                        estimator_input = np.expand_dims(estimator_input, axis=-1)

                    if X_cat is not None:
                        if len(X_cat.shape) != len(estimator_input.shape):
                            if (len(estimator_input.shape) == 2) and (len(X_cat.shape) > 2):
                                # e.g. estimator_input was passed through TCA then to an SVM so it is no longer originally 3D as X_cat is
                                X_cat = np.reshape(X_cat, (X_cat.shape[0], np.prod(X_cat.shape[1:])))
                            else:
                                # this would happen if the input is of dimension 3 and X_cat is of dimension 2
                                # but this is undetermined since it is not clear which axis to expand dims on
                                # In this case, it means you are using a Conv1D Classifier, but the underlying process which generated X (originally 3D then)
                                # should have also generated X_cat to be 3D
                                raise ValueError

                        # prepend the categorical features as the first ones in the feature axis
                        estimator_input = np.concatenate([X_cat, estimator_input], axis=-1)

                    if Y is not None:
                        # we need to fit the actual object, not merely the estimator_obj reference to it
                        estimator_label = copy.deepcopy(Y)
                        if self._use_internal_lbl_map:
                            if self._index2label is None:
                                self.build_label_mapping(Y)
                            estimator_label = self._convert_to_indices(Y)

                        self.estimators[estimator_idx][1].fit(estimator_input, estimator_label)
                        # save out random forest feature importances
                        if (type(estimator_obj).__name__.lower() == 'randomforestclassifier'):
                            self.feature_importances = [tree.feature_importances_ for tree in self.estimators[estimator_idx][1].estimators_]
                            self.feature_importances = np.array(self.feature_importances)
                    else:
                        if type(estimator_obj).__name__.lower() == 'linearsvc':
                            proba = self.estimators[estimator_idx][1]._predict_proba_lr(estimator_input)
                            assert(len(proba.shape) > 1)
                        else:
                            proba = self.estimators[estimator_idx][1].predict_proba(estimator_input)

        if Y is not None:
            return self
        else:
            return proba

    def fit(self, X, Y, X_cat=None):
        return self._inference(X=X, Y=Y, X_cat=X_cat)

    def predict_proba(self, X, X_cat=None):
        return self._inference(X=X, Y=None, X_cat=X_cat)

    def score(self, X, Y, X_cat=None):
        pred = self.predict(X, X_cat=X_cat)
        pred_labels = pred
        if self._use_internal_lbl_map:
            pred_labels = self._convert_to_labels(pred)
        return accuracy_score(y_true=Y, y_pred=pred_labels, normalize=True)

class ObsClassifier(ClassifierBase):
    """Given an Sklearn classifier (or a PipelineClassifier), we can either fit it or CV and validate on test data to get a list of metrics.

    Arguments:
        data: The data dictionary, either the full dataset to be subselected later via cls_dataset_transform_kwargs or a classifier amenable one
        cls_dataset_transform_kwargs: Used to subselect observables and to flatten to 2D if need be, no need to use this if the dataset has all finie values
        grp_keys: Defaults to using them but set it to None if you want to exclude the ternary categorical variables as features, as supplied by data['X_cat']
        classifier_type: A classifier object (either Sklearn or PipelineClassifier)
        classifier_kwargs: Any kwargs to be passed to the classifier (only relevant if you are NOT using cv_param_grid to select parameters for you instead)
        index2label: A user-specified mapping from named category labels to integers (automatically set by default if not specified)
        num_splits: Number of train/test splits
        split_idxs: If not None, runs particular a set of splits (single list of indices) rather than all num_splits of them. Useful for job parallelization.
        num_per_class_train: Numberof examples per category to train on
        num_per_class_test: Number of examples per category to test on
        train_filter: The subset of data we want to train on (None means all of the data is considered for training)
        test_filter: The subset of data we want to test on (None means all of the data is considered for training)
        cv_param_grid: Cross validation search space, see cls_cv_params.py for examples
        cv_kwargs: Any kwargs to be passed to GridSearchCV
        return_models: Whether to return the actual trained models for each split
        zscore_data: Whether to zscore the continuous data (data['X']). Note that data['X_cat'] is never zscored as it is categorical.
        verbose: Whether to print cross-validation scores and other messages (useful for debugging).
    """
    def __init__(self,
                 data,
                 cls_dataset_transform_kwargs=None,
                 grp_keys=DEFAULT_GRP_KEYS,
                 classifier_type='randomforest',
                 classifier_kwargs={},
                 index2label=None,
                 num_splits=10,
                 split_idxs=None,
                 num_per_class_train=None,
                 num_per_class_test=None,
                 train_filter=None,
                 test_filter=None,
                 cv_param_grid=None,
                 cv_kwargs={},
                 return_models=False,
                 zscore_data=False,
                 verbose=False
                 ):

        self._index2label = index2label
        self._grp_keys = grp_keys
        self._cls_dataset_transform_kwargs = cls_dataset_transform_kwargs
        self._num_splits = num_splits
        self._split_idxs = split_idxs
        self._num_per_class_train = num_per_class_train
        self._num_per_class_test = num_per_class_test
        self._train_filter = train_filter
        self._test_filter = test_filter

        self._cv_param_grid = cv_param_grid
        self._cv_kwargs = cv_kwargs
        self._return_models = return_models

        self._classifier_type = classifier_type
        self._classifier_kwargs = classifier_kwargs
        if isinstance(self._classifier_type, str):
            self._cls_func = estimator_from_str(cls_name=self._classifier_type)
        else:
            self._cls_func = self._classifier_type
        self._zscore_data = zscore_data
        self._verbose = verbose

        if not isinstance(self._cls_func, list):
            self.classifier = self._cls_func(**self._classifier_kwargs)
        else:
            self.classifier = None

        self._splits = None
        self.data = copy.deepcopy(data)
        if self._cls_dataset_transform_kwargs is not None:
            # subselect the specified obs_keys from full (3D) dataset in this case
            self.data = cls_dataset_transform(self.data,
                          **self._cls_dataset_transform_kwargs)

        self._obs_keys = list(self.data['obs_keys'])

        if self._grp_keys is not None:
            self._metric_keys = ['grp_id{}'.format(i) for i in range(len(grp_keys))] + self._obs_keys
        else:
            self._metric_keys = self._obs_keys

        if self._index2label is None:
            self.build_label_mapping(self.data['Y'])
        self.data['Y'] = self._convert_to_indices(self.data['Y'])

        if (self._num_per_class_train is not None) and (self._num_per_class_test is not None):
            train_test_splits = get_splits(metadata=self.data['meta'],
                                      split_by_func=lambda x: x['learning_rule'], # split data by learning rule category
                                      num_splits=self._num_splits,
                                      num_per_class_train=self._num_per_class_train,
                                      num_per_class_test=self._num_per_class_test,
                                      train_filter=self._train_filter,
                                      test_filter=self._test_filter)

            _ = validate_splits(splits=train_test_splits, labels=self.data['Y'])
            self._splits = train_test_splits

        self.results = None
        if self._cv_param_grid is not None:
            assert(self._splits is not None)
            self.classifier = None
            if self._verbose:
                print('Will be running cross val and returning results.')

            # we only consider the categorical portion of the data if grp_keys is specified as features to train on
            if self._grp_keys is not None:
                X_cat = self.data.get('X_cat', None)
                assert(X_cat is not None)
            else:
                X_cat = None

            self.results, _ = train_and_test_classifier(features=self.data['X'],
                                                       labels=self.data['Y'],
                                                       splits=self._splits,
                                                       split_idxs=self._split_idxs,
                                                       model_class=self._cls_func,
                                                       model_args=self._classifier_kwargs,
                                                       gridcv_params=self._cv_param_grid,
                                                       gridcv_args=self._cv_kwargs,
                                                       feature_norm=self._zscore_data,
                                                       X_cat=X_cat,
                                                       return_models=self._return_models,
                                                       index_to_label=self._index2label,
                                                       verbose=self._verbose)

            self.results['feature_names'] = np.array(self._metric_keys).astype('U')

    def fit(self, X=None, Y=None, X_cat=None, split_idx=0):
        assert(self.classifier is not None)
        if (X is None) or (Y is None):
            assert(self._splits is not None)
            self._inds = self._splits[split_idx]['train']
            X = copy.deepcopy(self.data['X'][self._inds])
            Y = copy.deepcopy(self.data['Y'][self._inds])
            if self._grp_keys is not None:
                X_cat = copy.deepcopy(self.data.get('X_cat', None))
                assert(X_cat is not None)
            else:
                X_cat = None
            if X_cat is not None:
                X_cat = X_cat[self._inds]

        if self._verbose:
            print('Input shape {} Label Shape {}'.format(X.shape, Y.shape))

        Y = copy.deepcopy(Y)
        if Y is None: # we use the internal mapping in __init__
            Y = self._convert_to_indices(Y)

        if X_cat is None:
            # to support sklearn cls
            self.classifier.fit(X, Y)
        else:
            self.classifier.fit(X, Y, X_cat=X_cat)
        return self

    def predict_proba(self, X=None, X_cat=None, split_idx=0, name='test'):
        assert(self.classifier is not None)
        if X is None:
            assert(self._splits is not None)
            self._inds = self._splits[split_idx][name]
            X = copy.deepcopy(self.data['X'][self._inds])
            if self._grp_keys is not None:
                X_cat = copy.deepcopy(self.data.get('X_cat', None))
                assert(X_cat is not None)
            else:
                X_cat = None
            if X_cat is not None:
                X_cat = X_cat[self._inds]

        if self._verbose:
            print('Input shape {}'.format(X.shape))

        assert len(X.shape) == 2, "expected 2-dimensional input"
        if type(self.classifier).__name__.lower() == 'linearsvc':
            assert(X_cat is None)
            proba = self.classifier._predict_proba_lr(X)
            assert(len(proba.shape) > 1)
        else:
            if X_cat is None:
                proba = self.classifier.predict_proba(X)
            else:
                proba = self.classifier.predict_proba(X, X_cat=X_cat)
        return proba

    def score(self, X=None, Y=None, X_cat=None, split_idx=0, name='test'):
        assert(self.classifier is not None)
        if (X is None) or (Y is None):
            assert(self._splits is not None)
            self._inds = self._splits[split_idx][name]
            X = copy.deepcopy(self.data['X'][self._inds])
            Y = copy.deepcopy(self.data['Y'][self._inds])
            if self._grp_keys is not None:
                X_cat = copy.deepcopy(self.data.get('X_cat', None))
                assert(X_cat is not None)
            else:
                X_cat = None
            if X_cat is not None:
                X_cat = X_cat[self._inds]

        if self._verbose:
            print('Input shape {} Label Shape {}'.format(X.shape, Y.shape))

        if X_cat is None:
            pred = self.predict(X)
        else:
            pred = self.predict(X, X_cat=X_cat)

        if Y is not None:
            pred_labels = pred
        else: # we are using the dataset in __init__ so we use the internal mapping
            pred_labels = self._convert_to_labels(pred)
        return accuracy_score(y_true=Y, y_pred=pred_labels, normalize=True)
