"""
This type stub file was generated by pyright.
"""

from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import _deprecate_positional_args

"""
Neighborhood Component Analysis
"""
class NeighborhoodComponentsAnalysis(TransformerMixin, BaseEstimator):
    """Neighborhood Components Analysis

    Neighborhood Component Analysis (NCA) is a machine learning algorithm for
    metric learning. It learns a linear transformation in a supervised fashion
    to improve the classification accuracy of a stochastic nearest neighbors
    rule in the transformed space.

    Read more in the :ref:`User Guide <nca>`.

    Parameters
    ----------
    n_components : int, default=None
        Preferred dimensionality of the projected space.
        If None it will be set to ``n_features``.

    init : {'auto', 'pca', 'lda', 'identity', 'random'} or ndarray of shape \
            (n_features_a, n_features_b), default='auto'
        Initialization of the linear transformation. Possible options are
        'auto', 'pca', 'lda', 'identity', 'random', and a numpy array of shape
        (n_features_a, n_features_b).

        'auto'
            Depending on ``n_components``, the most reasonable initialization
            will be chosen. If ``n_components <= n_classes`` we use 'lda', as
            it uses labels information. If not, but
            ``n_components < min(n_features, n_samples)``, we use 'pca', as
            it projects data in meaningful directions (those of higher
            variance). Otherwise, we just use 'identity'.

        'pca'
            ``n_components`` principal components of the inputs passed
            to :meth:`fit` will be used to initialize the transformation.
            (See :class:`~sklearn.decomposition.PCA`)

        'lda'
            ``min(n_components, n_classes)`` most discriminative
            components of the inputs passed to :meth:`fit` will be used to
            initialize the transformation. (If ``n_components > n_classes``,
            the rest of the components will be zero.) (See
            :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)

        'identity'
            If ``n_components`` is strictly smaller than the
            dimensionality of the inputs passed to :meth:`fit`, the identity
            matrix will be truncated to the first ``n_components`` rows.

        'random'
            The initial transformation will be a random array of shape
            `(n_components, n_features)`. Each value is sampled from the
            standard normal distribution.

        numpy array
            n_features_b must match the dimensionality of the inputs passed to
            :meth:`fit` and n_features_a must be less than or equal to that.
            If ``n_components`` is not None, n_features_a must match it.

    warm_start : bool, default=False
        If True and :meth:`fit` has been called before, the solution of the
        previous call to :meth:`fit` is used as the initial linear
        transformation (``n_components`` and ``init`` will be ignored).

    max_iter : int, default=50
        Maximum number of iterations in the optimization.

    tol : float, default=1e-5
        Convergence tolerance for the optimization.

    callback : callable, default=None
        If not None, this function is called after every iteration of the
        optimizer, taking as arguments the current solution (flattened
        transformation matrix) and the number of iterations. This might be
        useful in case one wants to examine or store the transformation
        found after each iteration.

    verbose : int, default=0
        If 0, no progress messages will be printed.
        If 1, progress messages will be printed to stdout.
        If > 1, progress messages will be printed and the ``disp``
        parameter of :func:`scipy.optimize.minimize` will be set to
        ``verbose - 2``.

    random_state : int or numpy.RandomState, default=None
        A pseudo random number generator object or a seed for it if int. If
        ``init='random'``, ``random_state`` is used to initialize the random
        transformation. If ``init='pca'``, ``random_state`` is passed as an
        argument to PCA when initializing the transformation. Pass an int
        for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The linear transformation learned during fitting.

    n_iter_ : int
        Counts the number of iterations performed by the optimizer.

    random_state_ : numpy.RandomState
        Pseudo random number generator object used during initialization.

    Examples
    --------
    >>> from sklearn.neighbors import NeighborhoodComponentsAnalysis
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ... stratify=y, test_size=0.7, random_state=42)
    >>> nca = NeighborhoodComponentsAnalysis(random_state=42)
    >>> nca.fit(X_train, y_train)
    NeighborhoodComponentsAnalysis(...)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> knn.fit(X_train, y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(X_test, y_test))
    0.933333...
    >>> knn.fit(nca.transform(X_train), y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(nca.transform(X_test), y_test))
    0.961904...

    References
    ----------
    .. [1] J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov.
           "Neighbourhood Components Analysis". Advances in Neural Information
           Processing Systems. 17, 513-520, 2005.
           http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf

    .. [2] Wikipedia entry on Neighborhood Components Analysis
           https://en.wikipedia.org/wiki/Neighbourhood_components_analysis

    """
    @_deprecate_positional_args
    def __init__(self, n_components=..., *, init=..., warm_start=..., max_iter=..., tol=..., callback=..., verbose=..., random_state=...) -> None:
        ...
    
    def fit(self, X, y): # -> Self@NeighborhoodComponentsAnalysis:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.

        y : array-like of shape (n_samples,)
            The corresponding training labels.

        Returns
        -------
        self : object
            returns a trained NeighborhoodComponentsAnalysis model.
        """
        ...
    
    def transform(self, X):
        """Applies the learned transformation to the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        X_embedded: ndarray of shape (n_samples, n_components)
            The data samples transformed.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called before.
        """
        ...
    


