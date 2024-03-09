import numpy as np
import os
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import smacof
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
# from pattern_search.mds import MDS

target_dimensions_init = np.array([5, 10, 25, 40, 50, 75, 100])


def ISOMAP_dr(X, target_dimensions=None):
    if target_dimensions is None:
        global target_dimensions_init
        target_dimensions = target_dimensions_init

    for i, l in enumerate(target_dimensions):
        dr = Isomap(n_components=l)
        yield dr.fit_transform(X)


def LLE_dr(X, target_dimensions=None):
    if target_dimensions is None:
        global target_dimensions_init
        target_dimensions = target_dimensions_init

    for i, l in enumerate(target_dimensions):
        dr = LLE(n_components=l)
        yield dr.fit_transform(X)


def LLE_dr2(X, target_dimensions=None):
    if target_dimensions is None:
        global target_dimensions_init
        target_dimensions = target_dimensions_init

    dr = LLE(n_components=l)
    return dr.fit_transform(X)

def MLLE_dr(X, target_dimensions=None):
    if target_dimensions is None:
        global target_dimensions_init
        target_dimensions = target_dimensions_init

    for i, l in enumerate(target_dimensions):
        dr = LLE(n_neighbors=2 * l, n_components=l, method='modified', eigen_solver='dense')
        yield dr.fit_transform(X)


def PCA_dr(X, target_dimensions=None):
    if target_dimensions is None:
        global target_dimensions_init
        target_dimensions = target_dimensions_init

    for i, l in enumerate(target_dimensions):
        dr = PCA(n_components=l)
        yield dr.fit_transform(X)


# def pattern_search_MDS_dr(X, target_dimensions=None):
#     if target_dimensions is None:
#         global target_dimensions_init
#         target_dimensions = target_dimensions_init
#
#     for i, l in enumerate(target_dimensions):
#         dr = MDS(n_components=l)
#         yield dr.fit_transform(X)


def SMACOF_dr(X, target_dimensions=None):
    if target_dimensions is None:
        global target_dimensions_init
        target_dimensions = target_dimensions_init
    dist = DistanceMetric.get_metric('euclidean')
    dist_mat = dist.pairwise(X)

    for i, l in enumerate(target_dimensions):
        X_reduced = smacof(dist_mat, n_components=l)[0]
        yield X_reduced


def spectral_embedding_dr(X, target_dimensions=None):
    if target_dimensions is None:
        global target_dimensions_init
        target_dimensions = target_dimensions_init

    for i, l in enumerate(target_dimensions):
        dr = SpectralEmbedding(n_components=l)
        yield dr.fit_transform(X)


DR = [ISOMAP_dr, LLE_dr, MLLE_dr, PCA_dr, SMACOF_dr, spectral_embedding_dr]
# DR = [ISOMAP_dr, LLE_dr, MLLE_dr, PCA_dr, pattern_search_MDS_dr, SMACOF_dr, spectral_embedding_dr]
