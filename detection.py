import os
import glob

import numpy as np
import pandas as pd
import nibabel as nb

from matplotlib import pyplot as plt
from sklearn.decomposition import RandomizedPCA
from masker import NeurovaultFeatureExtractor
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import classification_report


# setting up paths and removing 0407 which does not open
data_dir = os.path.join(os.getenv('HOME'), 'neurovault_analysis', 'data')
cache_dir = os.path.join(os.getenv('HOME'), 'neurovault_analysis', 'cache')
images = sorted(glob.glob(os.path.join(data_dir, 'original', '*.nii.gz')))
images.remove(os.path.join(data_dir, 'original', '0407.nii.gz'))
brain_mask_img = nb.load(os.path.join(data_dir, 'MNI152_T1_3mm_brain_mask.nii.gz'))


# extracting features
extractor = NeurovaultFeatureExtractor(memory=cache_dir, percentiles=[1, 5, 10, 40, 50, 60, 80, 90, 95, 99], n_jobs=-1)
X = extractor.fit_transform(images)

# remove nans with Imputers
imputer = Imputer(strategy='median')
X = imputer.fit_transform(X)
imputer = Imputer(np.inf, strategy='median')
X = imputer.fit_transform(X)
imputer = Imputer(-np.inf, strategy='median')
X = imputer.fit_transform(X)

# load labels for is_stat_map classification
labels = pd.read_csv(os.path.join(data_dir,'labels.csv'))[['image_id', 'map_type', 'is_stat_map', 'map_type_v2']]
y = labels['is_stat_map'].values

# cross_validation and prediction
cv = StratifiedShuffleSplit(y, n_iter=50, random_state=10)
clf = RandomForestClassifier(n_estimators=20, max_depth=20, n_jobs=-1)

Y_pred = []
Y_true = []

for train, test in cv:
    y_pred, y_true = clf.fit(X[train], y[train]).predict(X[test]), y[test]

    Y_pred.append(y_pred)
    Y_true.append(y_true)

Y_true = np.hstack(Y_true)
Y_pred = np.hstack(Y_pred)

print classification_report(Y_true, Y_pred)

# now load labels for multi-class predicion (predict map_type)

y_type = labels['map_type_v2'].values

Y_type_pred = []
Y_type_true = []

for train, test in cv:
    y_type_pred, y_type_true = clf.fit(X[train], y_type[train]).predict(X[test]), y_type[test]

    Y_type_pred.append(y_type_pred)
    Y_type_true.append(y_type_true)

Y_type_true = np.hstack(Y_type_true)
Y_type_pred = np.hstack(Y_type_pred)

print classification_report(Y_type_true, Y_type_pred)
