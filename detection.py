import os
from os.path import join as pjoin
import glob

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.decomposition import RandomizedPCA
from masker import NeurovaultEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report

# data_dir = pjoin(os.getenv('HOME'), 'neurovault_analysis', 'data')
# cache_dir = pjoin(os.getenv('HOME'), 'neurovault_analysis', 'cache')

data_dir = pjoin('/media', 'ahoyosid', 'Seagate Backup Plus Drive',
                 'neurovault_analysis')
cache_dir = 'cache'


niimgs_dir = pjoin(data_dir, 'original')
images = sorted(glob.glob(pjoin(niimgs_dir, '*.nii.gz')))
images.remove(pjoin(niimgs_dir, '0407.nii.gz')

encoder = NeurovaultEncoder(memory=cache_dir, percentiles=[5, 95], n_jobs=-1)
X = encoder.fit_transform(images)

imputer = Imputer(strategy='median')
X = imputer.fit_transform(X)
imputer = Imputer(np.inf, strategy='median')
X = imputer.fit_transform(X)
imputer = Imputer(-np.inf, strategy='median')
X = imputer.fit_transform(X)

labels = pd.read_csv(pjoin(data_dir,'labels.csv'))[['image_id', 'map_type',
                                                    'is_yannick_brain']]
y = labels['is_yannick_brain'].values

print (y == 0).sum()

cv = ShuffleSplit(y.size, n_iter=50, random_state=10)
clf = LogisticRegression(C=1, penalty='l2')
clf = RandomForestClassifier(n_estimators=20, max_depth=20, n_jobs=-1)

Y_pred = []
Y_true = []
errors = []
truth = []
for train, test in cv:
    y_pred, y_true = clf.fit(X[train], y[train]).predict(X[test]), y[test]

    errors.extend(test[y_true != y_pred])
    truth.extend(y_true[y_true != y_pred])

    Y_pred.append(y_pred)
    Y_true.append(y_true)

Y_true = np.hstack(Y_true)
Y_pred = np.hstack(Y_pred)

print classification_report(Y_true, Y_pred)
print np.unique(np.array(images)[np.array(errors)])

# scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
# print 'scores', np.mean(scores), np.std(scores)
