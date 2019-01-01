import numpy as np


models = [
  'linear',
  'variationally_sparse_gp',
  # 'variationally_sparse_gp_minibatch',
  # 'deep_gp_doubly_stochastic',
  'svm',
  'knn',
  'decision_tree',
  'random_forest',
  'gradient_boosting_machine',
  'adaboost',
  'mlp',
  'neural_kernel_network',
  ]

splits = list(range(1, 11))

datasets = ['boston', 'energy', 'yacht']


def parse_log(model, dataset, sp):
    with open('results/%s/%s/split_%d.txt' % (dataset, model, sp), 'r') as file_:
        res = file_.readlines()[-2]
        loglik = float(res.split('test_loglik_unnormalized')[1][3:].split('test_ma')[0][:-3])
        rmse = float(res.split('test_rmse_unnormalized')[1][3:].split('model')[0][:-3])
    return loglik, rmse

def parse_logs(model, dataset):
    logliks, rmses = [], []
    for sp in splits:
        l, r = parse_log(model, dataset, sp)
        logliks.append(l)
        rmses.append(r)
    return logliks, rmses

for dataset in datasets:
    print('---------- dataset %s ---------' % dataset)
    for m in models:
        logliks, rmses = parse_logs(m, dataset)
        print('model %15s -- loglikelihood %.4f, %.4f -- rmse %.4f, %.4f' % (
            m, np.mean(logliks), np.std(logliks), np.mean(rmses), np.std(rmses)
        ))