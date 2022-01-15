"""
This type stub file was generated by pyright.
"""

from ._locally_linear import LocallyLinearEmbedding, locally_linear_embedding
from ._isomap import Isomap
from ._mds import MDS, smacof
from ._spectral_embedding import SpectralEmbedding, spectral_embedding
from ._t_sne import TSNE, trustworthiness

"""
The :mod:`sklearn.manifold` module implements data embedding techniques.
"""
__all__ = ['locally_linear_embedding', 'LocallyLinearEmbedding', 'Isomap', 'MDS', 'smacof', 'SpectralEmbedding', 'spectral_embedding', "TSNE", 'trustworthiness']
