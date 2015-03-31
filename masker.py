import os
from os.path import join as pjoin
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

# data_dir = pjoin(os.getenv('HOME'), 'neurovault_analysis', 'data')
brain_mask_img = nb.load(pjoin(os.sep, 'usr', 'share', 'fsl', 'data',
                               'standard', 'MNI152_T1_3mm_brain_mask.nii.gz'))
brain_n_voxels = int(brain_mask_img.get_data().sum())


class NeurovaultEncoder(TransformerMixin, CacheMixin):

    def __init__(self, percentiles=None, n_dilations=1, n_jobs=1, memory=None, memory_level=1):
        self.percentiles = _check_percentiles(percentiles)
        self.n_dilations = n_dilations
        self.n_jobs = n_jobs
        self.memory = memory
        self.memory_level = 1

    def fit(self, imgs, y=None):
        # compute_neurovault_mask_proxy = delayed(self._cache(compute_neurovault_mask, func_memory_level=1))
        # Parallel(n_jobs=self.n_jobs)(
        #     compute_neurovault_mask_proxy(img, self.n_dilations)
        #     for img in imgs)

        return self

    def transform(self, imgs, y=None):
        X = []

        delayed_extract_neurovault_features = delayed(self._cache(extract_neurovault_features))

        X = Parallel(n_jobs=self.n_jobs)(
            delayed_extract_neurovault_features(img, self.n_dilations, self.percentiles)
            for img in imgs)

        # for img in imgs:
        #     x = self._cache(extract_neurovault_features, func_memory_level=1)(img, self.n_dilations, self.percentiles)
        #     X.append(x)

        return np.vstack(X)


def compute_neurovault_mask(img, n_dilations):
    # compute mask
    img = check_niimg(img)
    data = img.get_data()
    data[np.isnan(data)] = 0

    try:
        brain_mask = resample_img(brain_mask_img,
                                  target_affine=img.get_affine(),
                                  target_shape=img.shape,
                                  interpolation='nearest').get_data() > 0.
    except Exception, e:
        print e
        return None, None, None

    if (data == 0).sum() / float(data.size) > .1:
        mask = np.abs(data) > 0
        mask = binary_dilation(mask, iterations=n_dilations)
        mask = binary_erosion(mask, iterations=n_dilations)

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

    n_features = len(percentiles) + 5

    if mask_img is None:
        return np.zeros(n_features)

    # mask_img.to_filename('/home/ys218403/neurovault_analysis/data/masks/%s' % img.split('/')[-1])
    # try:
    #     plot_img(mask_img).savefig('/home/ys218403/neurovault_analysis/data/masks/%s.png' % img.split('/')[-1])
    # except:
    #     mask_img.to_filename('/home/ys218403/neurovault_analysis/data/masks/%s' % img.split('/')[-1])

    # plt.close('all')
    data = check_niimg(img).get_data()
    x = data[mask_img.get_data() > 0.]

    if x.shape[0] == 0:
        return np.zeros(n_features)
    else:
        f = np.percentile(x, percentiles)
        f = np.hstack([f, [brain_coverage,
                           out_brain,
                           brain_coverage / out_brain,
                           brain_coverage + out_brain,
                           np.std(x),
                       ]
                   ])

    return f


def _check_percentiles(percentiles):
    if percentiles is None:
        percentiles = np.linspace(5, 99, 3).tolist()
    if isinstance(percentiles, np.ndarray):
        percentiles = percentiles.tolist()

    return percentiles


if __name__ == '__main__':
    import glob

    cache_dir = os.path.join(os.getenv('HOME'), 'neurovault_analysis', 'cache')
    images = glob.glob(os.path.join(data_dir, 'original', '*.nii.gz'))
    images.remove('/home/ys218403/neurovault_analysis/data/original/0407.nii.gz')

    encoder = NeurovaultEncoder(memory=cache_dir, n_jobs=-1)
    X = encoder.fit_transform(images)
