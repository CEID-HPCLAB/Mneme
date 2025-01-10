"""
Generate samples of synthetic data sets.
"""

# Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
#          G. Louppe, J. Nothman
# License: BSD 3 clause

# import numbers
# import array
# from collections.abc import Iterable

import numpy as np
# from scipy import linalg
# import scipy.sparse as sp

# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement
# from sklearn.utils.validation import _deprecate_positional_args


def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions
    """
    if dimensions > 30:
        return np.hstack([rng.randint(2, size=(samples, dimensions - 30)),
                          _generate_hypercube(samples, 30, rng)])
    out = sample_without_replacement(2 ** dimensions, samples,
                                     random_state=rng).astype(dtype='>u4',
                                                              copy=False)
    out = np.unpackbits(out.view('>u1')).reshape((-1, 32))[:, -dimensions:]
    return out


# @_deprecate_positional_args
def make_classification(n_samples=100, n_features=20, *, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None,
                        fixed_random_state=None, shuffle_features=False):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder. Note that the default setting flip_y > 0 might lead
        to less than n_classes in y in some cases.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See also
    --------
    make_blobs: simplified variant
    make_multilabel_classification: unrelated generator for multilabel tasks
    """
    generator = check_random_state(random_state)
    if fixed_random_state:
        fixed_generator = check_random_state(fixed_random_state)
    else:
        fixed_generator = generator

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated "
                         "features must sum to less than the number of total"
                         " features")
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(msg.format(n_classes, n_clusters_per_class,
                                    n_informative, 2**n_informative))

    if weights is not None:
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError("Weights specified but incompatible with number "
                             "of classes.")
        if len(weights) == n_classes - 1:
            if isinstance(weights, list):
                weights = weights + [1.0 - sum(weights)]
            else:
                weights = np.resize(weights, n_classes)
                weights[-1] = 1.0 - sum(weights[:-1])
    else:
        weights = [1.0 / n_classes] * n_classes

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters by weight
    n_samples_per_cluster = [
        int(n_samples * weights[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int32)

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative,
                                  fixed_generator).astype(float, copy=False)
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= fixed_generator.rand(n_clusters, 1)
        centroids *= fixed_generator.rand(1, n_informative)

    # print("centroids=", centroids)
    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.randn(n_samples, n_informative)

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster

        A = 2 * fixed_generator.rand(n_informative, n_informative) - 1
        # print(k, A)
        X_k[...] = np.dot(X_k, A)  # introduce random covariance

        X_k += centroid  # shift the cluster to a vertex

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = generator.randn(n_samples, n_useless)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * fixed_generator.rand(n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * fixed_generator.rand(n_features)
    X *= scale

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

    if shuffle_features:
        # Randomly permute features
        if fixed_random_state:
            fixed_generator = check_random_state(fixed_random_state)
        else:
            fixed_generator = generator

        indices = np.arange(n_features)
        fixed_generator.shuffle(indices)
        X[:, :] = X[:, indices]

    return X, y