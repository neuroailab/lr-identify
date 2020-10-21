'''
Central file in which to define observables for use. Each observable consists of a measure (the measured
quantity being transformed), a function of the measure (some transformation to all measurements), and a
statistic calculated from the transformed measurement.
'''
import itertools


# STANDARD CASES
KWARGS = ['obs_frac', 'obs_stddev', 'obs_seed']
MEASURES = ['weight', 'act']
FUNCTIONS = ['raw', 'sq', 'abs']
STATISTICS = ['norm', 'mean', 'median', 'thirdquartile', 'var', 'skew', 'kurtosis']

def make_obs_name(measure, function, statistic):
    return '{}{}_{}'.format(measure, function, statistic)

ALL_OBSERVABLE_NAMES = [make_obs_name(measure, function, statistic) for measure, function, statistic in itertools.product(MEASURES, FUNCTIONS, STATISTICS)]


# SPECIAL CASES
ALL_OBSERVABLE_NAMES.extend([make_obs_name('gradavg', function, 'id') for function in FUNCTIONS])


# INDEPENDENT VARIABLE VALUES
TASKS = ['imagenet', 'simclr', 'audionet', 'cifar10']
ARCHITECTURES = ['resnet34v2', 'resnet34', 'resnet18v2', 'resnet18', 'alexnet', 'alexnetlrn', 'knet', 'knetlrn', 'knetc4', 'knetc4lrn']
BATCH_SIZES = [128, 256, 512]
MODEL_SEEDS = [None, 0]
DATASET_SEEDS = [None, 0]

# DEPENDENT VARIABLE VALUES
LEARNING_RULES = ['adam', 'sgdm', 'information', 'feedback']
