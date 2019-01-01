import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from importlib import import_module
import os

from .non_bayesian_models import non_bayesian_model

abs_path = os.path.abspath(__file__)[:-len('/get_model.py')]

def get_regression_model(name):
    assert name in all_regression_models
    if name == 'neural_kernel_network':
        from .neural_kernel_network.models import RegressionModel as m
        return m
    if name == 'doubly_stochastic_dgp':
        from .deep_gp_doubly_stochastic.models import RegressionModel as m
        return m
    return non_bayesian_model(name, 'regression') or \
           import_module('bayesian_benchmarks.models.{}.models'.format(name)).RegressionModel

def get_classification_model(name):
    assert name in all_classification_models
    return non_bayesian_model(name, 'classification') or \
           import_module('bayesian_benchmarks.models.{}.models'.format(name)).ClassificationModel

all_regression_models = [
      'linear',
      'variationally_sparse_gp',
      'variationally_sparse_gp_minibatch',
      'deep_gp_doubly_stochastic',
      'svm',
      'knn',
      'decision_tree',
      'random_forest',
      'gradient_boosting_machine',
      'adaboost',
      'mlp',
      'neural_kernel_network',
      ]

all_classification_models = [
    'linear',
    'variationally_sparse_gp',
    'variationally_sparse_gp_minibatch',
    'deep_gp_doubly_stochastic',
    'svm',
    'naive_bayes',
    'knn',
    'decision_tree',
    'random_forest',
    'gradient_boosting_machine',
    'adaboost',
    'mlp',
    'neural_kernel_network',
    ]

all_models = list(set(all_regression_models).union(set(all_classification_models)))
