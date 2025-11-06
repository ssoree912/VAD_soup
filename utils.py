import torch

import random
import os
import numpy as np
import time
import logging
from sklearn import metrics

def set_seeds(seed):
    print('set seed {}'.format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def get_timestamp():
    return time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def log_param(logger, args):
    params = vars(args)
    keys = sorted(params.keys())
    for k in keys:
        logger.info('{}\t{}'.format(k, params[k]))

def cal_pr_auc(scores, labels):
    precision, recall, th = metrics.precision_recall_curve(labels, scores)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

def cal_rec_auc(scores,labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels,scores,pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc

def _to_1d_array(values, dtype):
    """Flatten potentially nested metric inputs into a 1D numpy array."""
    def _collect(item, collector):
        if isinstance(item, np.ndarray):
            if item.dtype == object:
                _collect(item.tolist(), collector)
            else:
                collector.extend(item.reshape(-1).tolist())
        elif isinstance(item, (list, tuple)):
            for element in item:
                _collect(element, collector)
        else:
            collector.append(item)

    if isinstance(values, np.ndarray) and values.dtype != object:
        return values.reshape(-1).astype(dtype, copy=False)

    flat_list = []
    _collect(values, flat_list)
    return np.asarray(flat_list, dtype=dtype)

def calc_metrics(total_scores, total_labels):
    total_scores = _to_1d_array(total_scores, np.float32)
    total_labels = _to_1d_array(total_labels, np.int64)

    pr_auc = cal_pr_auc(total_scores, total_labels)
    rec_auc = cal_rec_auc(total_scores, total_labels)

    return pr_auc*100, rec_auc*100
