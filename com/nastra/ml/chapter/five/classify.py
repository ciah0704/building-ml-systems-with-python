__author__ = 'nastra'
import time
start_time = time.time()

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import KFold
from sklearn import neighbors
from data import chosen, chosen_meta
from utils import plot_roc, plot_pr
from utils import plot_feat_importance
from utils import load_meta
from utils import fetch_posts
from utils import plot_feat_hist
from utils import plot_bias_variance
from utils import plot_k_complexity

# question Id -> {'features'->feature vector, 'answers'->[answer Ids]}, 'scores'->[scores]}
# scores will be added on-the-fly as the are not in meta
meta, id_to_idx, idx_to_id = load_meta(chosen_meta)

import nltk

# splitting questions into train (70%) and test(30%) and then take their
# answers
all_posts = list(meta.keys())
all_questions = [q for q, v in meta.items() if v['ParentId'] == -1]
all_answers = [q for q, v in meta.items() if v['ParentId'] != -1]  # [:500]

feature_names = np.array((
    'NumTextTokens',
    'NumCodeLines',
    'LinkCount',
    'AvgSentLen',
    'AvgWordLen',
    'NumAllCaps',
    'NumExclams',
    'NumImages'
))

# activate the following for reduced feature space
"""
feature_names = np.array((
    'NumTextTokens',
    'LinkCount',
))
"""