import os

import numpy as np
import nibabel as nb

from scipy.ndimage import binary_dilation, binary_erosion
from nilearn._utils import CacheMixin
from nilearn._utils import check_niimg
from nilearn.image import resample_img
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from joblib import Parallel, delayed, Memory
from nilearn.plotting import plot_img

from matplotlib import pyplot as plt

data_dir = os.path.join(os.getenv('HOME'), 'neurovault_analysis', 'data')
brain_mask_img = nb.load(os.path.join(data_dir, 'MNI152_T1_3mm_brain_mask.nii.gz'))
brain_n_voxels = int(brain_mask_img.get_data().sum())


class NeurovaultFeatureExtractor(TransformerMixin, CacheMixin):
    """Compute masks from Neurovault images, and extract features
    for outlier detection.

    Parameters
    ----------
    percentiles: list or None
        list of percentiles of the image to compute
    n_dilations: int
        number of dilation iterations (and then of erosions) to compute the final mask
    """
    def __init__(self, percentiles=None, n_dilations=1, n_jobs=1, memory=None, memory_level=1):
        self.percentiles = _check_percentiles(percentiles)
        self.n_dilations = n_dilations
        self.n_jobs = n_jobs
        self.memory = memory
        self.memory_level = 1

    def fit(self, imgs, y=None):
        return self

    def transform(self, imgs, y=None):
        X = []

        delayed_extract_neurovault_features = delayed(self._cache(extract_neurovault_features))

        X = Parallel(n_jobs=self.n_jobs)(
            delayed_extract_neurovault_features(img, self.n_dilations, self.percentiles)
            for img in imgs)

        return np.vstack(X)


def compute_neurovault_mask(img, n_dilations):
    # compute mask
    img = check_niimg(img)
    data = img.get_data()
    data[np.isnan(data)] = 0

    # resample reference brain mask from FSL to image size
    try:
        brain_mask = resample_img(brain_mask_img,
                                  target_affine=img.get_affine(),
                                  target_shape=img.shape,
                                  interpolation='nearest').get_data() > 0.
    except Exception, e:
        return None, None, None

    # if there is a least 10 percent of zeros in the image, we consider it the background value
    if (data == 0).sum() / float(data.size) > .1:
        mask = np.abs(data) > 0
        mask = binary_dilation(mask, iterations=n_dilations)
        mask = binary_erosion(mask, iterations=n_dilations)

    # else we compute a kmeans with 2 clusters, and use the kmeans label that gives the
    # best brain coverage (actually the ratio of brain coverage, and number of voxels out of
    # the reference brain mask).
    else:
        best_ratio = 0
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data.ravel()[np.newaxis].T)

        for label in np.unique(kmeans.labels_):
            mask = (kmeans.labels_ == label).reshape(img.shape)
            brain_coverage = brain_mask[mask].sum() / float(brain_mask.sum())
            if mask.sum() == 0:
                out_brain = 0
            else:
                out_brain = mask[~brain_mask].sum() / float(mask.sum())
            
            if best_ratio < brain_coverage / out_brain:
                best_ratio = brain_coverage / out_brain
                best_label = label

        mask = (kmeans.labels_ == label).reshape(img.shape)

    # recompute for everyone
    brain_coverage = brain_mask[mask].sum() / float(brain_mask.sum())
    if mask.sum() == 0:
        out_brain = 0
    else:
        out_brain = mask[~brain_mask].sum() / float(mask.sum())

    mask_img = nb.Nifti1Image(mask.astype(np.float), affine=check_niimg(img).get_affine())
    return mask_img, brain_coverage, out_brain


def extract_neurovault_features(img, n_dilations, percentiles):
    mask_img, brain_coverage, out_brain = compute_neurovault_mask(img, n_dilations)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters)

    n_features = len(percentiles) + 7

    # if mask computation fails, then the image must be weird and we return a vector of zeros
    if mask_img is None:
        return np.zeros(n_features)

    # get data inside the mask
    data = check_niimg(img).get_data()
    x = data[mask_img.get_data() > 0.]

    # if image has no values inside the mask, we return a vector of zeros
    if x.shape[0] == 0:
        return np.zeros(n_features)
    else:
        # extract a bunch of descriptors
        f = np.percentile(x, percentiles)
        f = np.hstack([f,                            # percentiles
                       [brain_coverage,              # percentage of brain coverage
                        out_brain,                   # percentage of voxels out of the brain from computed mask
                        brain_coverage / out_brain,  # ratio
                        brain_coverage + out_brain,  # sum
                        np.std(x),                   # std
                        np.min(x),                   # min
                        np.max(x),                   # max
                       ]
                   ])

    return f


def _check_percentiles(percentiles):
    if percentiles is None:
        percentiles = np.linspace(5, 99, 3).tolist()
    if isinstance(percentiles, np.ndarray):
        percentiles = percentiles.tolist()

    return percentiles
