from collections import OrderedDict

DEFAULT_GRP_KEYS = ['early', 'middle', 'deep']

'''The set of observable statistics that have been finite across all models and tasks tested so far.'''
# preserve observable class ordering
ADMISSIBLE_OBS = OrderedDict()
ADMISSIBLE_OBS['weight'] = ['weightabs_mean',
                            'weightabs_median',
                            'weightabs_norm',
                            'weightabs_thirdquartile',
                            'weightabs_var',
                            'weightraw_mean',
                            'weightraw_median',
                            'weightraw_norm',
                            'weightraw_thirdquartile',
                            'weightraw_var',
                            'weightsq_mean',
                            'weightsq_median',
                            'weightsq_thirdquartile']

ADMISSIBLE_OBS['act'] = ['actabs_mean',
                         'actabs_median',
                         'actabs_norm',
                         'actabs_thirdquartile',
                         'actraw_mean',
                         'actraw_median',
                         'actraw_norm',
                         'actraw_thirdquartile']

ADMISSIBLE_OBS['grad'] = ['gradavgabs_id',
                          'gradavgraw_id',
                          'gradavgsq_id']

ADMISSIBLE_OBS['all'] = ADMISSIBLE_OBS['weight'] + ADMISSIBLE_OBS['act'] + ADMISSIBLE_OBS['grad']

'''The set of all observable statistics (not guaranteed to always be finite).'''
# preserve observable class ordering
ALL_OBS = OrderedDict()
ALL_OBS['weight'] = ['weightabs_kurtosis',
                     'weightabs_mean',
                     'weightabs_median',
                     'weightabs_norm',
                     'weightabs_skew',
                     'weightabs_thirdquartile',
                     'weightabs_var',
                     'weightraw_kurtosis',
                     'weightraw_mean',
                     'weightraw_median',
                     'weightraw_norm',
                     'weightraw_skew',
                     'weightraw_thirdquartile',
                     'weightraw_var',
                     'weightsq_kurtosis',
                     'weightsq_mean',
                     'weightsq_median',
                     'weightsq_norm',
                     'weightsq_skew',
                     'weightsq_thirdquartile',
                     'weightsq_var']

ALL_OBS['act'] = ['actabs_kurtosis',
                  'actabs_mean',
                  'actabs_median',
                  'actabs_norm',
                  'actabs_skew',
                  'actabs_thirdquartile',
                  'actabs_var',
                  'actraw_kurtosis',
                  'actraw_mean',
                  'actraw_median',
                  'actraw_norm',
                  'actraw_skew',
                  'actraw_thirdquartile',
                  'actraw_var',
                  'actsq_kurtosis',
                  'actsq_mean',
                  'actsq_median',
                  'actsq_norm',
                  'actsq_skew',
                  'actsq_thirdquartile',
                  'actsq_var']

ALL_OBS['grad'] = ['gradavgabs_id',
                   'gradavgraw_id',
                   'gradavgsq_id']

ALL_OBS['all'] = ALL_OBS['weight'] + ALL_OBS['act'] + ALL_OBS['grad']
