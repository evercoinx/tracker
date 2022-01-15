"""
This type stub file was generated by pyright.
"""

from ._spectral import SpectralClustering, spectral_clustering
from ._mean_shift import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from ._affinity_propagation import AffinityPropagation, affinity_propagation
from ._agglomerative import AgglomerativeClustering, FeatureAgglomeration, linkage_tree, ward_tree
from ._kmeans import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from ._dbscan import DBSCAN, dbscan
from ._optics import OPTICS, cluster_optics_dbscan, cluster_optics_xi, compute_optics_graph
from ._bicluster import SpectralBiclustering, SpectralCoclustering
from ._birch import Birch

"""
The :mod:`sklearn.cluster` module gathers popular unsupervised clustering
algorithms.
"""
__all__ = ['AffinityPropagation', 'AgglomerativeClustering', 'Birch', 'DBSCAN', 'OPTICS', 'cluster_optics_dbscan', 'cluster_optics_xi', 'compute_optics_graph', 'KMeans', 'FeatureAgglomeration', 'MeanShift', 'MiniBatchKMeans', 'SpectralClustering', 'affinity_propagation', 'dbscan', 'estimate_bandwidth', 'get_bin_seeds', 'k_means', 'kmeans_plusplus', 'linkage_tree', 'mean_shift', 'spectral_clustering', 'ward_tree', 'SpectralBiclustering', 'SpectralCoclustering']
