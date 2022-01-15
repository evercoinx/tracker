"""
This type stub file was generated by pyright.
"""

from ..base import BaseEstimator
from ..utils.validation import _deprecate_positional_args

"""
Maximum likelihood covariance estimator.

"""
def log_likelihood(emp_cov, precision):
    """Computes the sample mean of the log_likelihood under a covariance model

    computes the empirical expected log-likelihood (accounting for the
    normalization terms and scaling), allowing for universal comparison (beyond
    this software package)

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        Maximum Likelihood Estimator of covariance.

    precision : ndarray of shape (n_features, n_features)
        The precision matrix of the covariance model to be tested.

    Returns
    -------
    log_likelihood_ : float
        Sample mean of the log-likelihood.
    """
    ...

@_deprecate_positional_args
def empirical_covariance(X, *, assume_centered=...): # -> ndarray[Unknown, Unknown]:
    """Computes the Maximum likelihood covariance estimator


    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).

    Examples
    --------
    >>> from sklearn.covariance import empirical_covariance
    >>> X = [[1,1,1],[1,1,1],[1,1,1],
    ...      [0,0,0],[0,0,0],[0,0,0]]
    >>> empirical_covariance(X)
    array([[0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25]])
    """
    ...

class EmpiricalCovariance(BaseEstimator):
    """Maximum likelihood covariance estimator

    Read more in the :ref:`User Guide <covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import EmpiricalCovariance
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                             cov=real_cov,
    ...                             size=500)
    >>> cov = EmpiricalCovariance().fit(X)
    >>> cov.covariance_
    array([[0.7569..., 0.2818...],
           [0.2818..., 0.3928...]])
    >>> cov.location_
    array([0.0622..., 0.0193...])

    """
    @_deprecate_positional_args
    def __init__(self, *, store_precision=..., assume_centered=...) -> None:
        ...
    
    def get_precision(self): # -> None:
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
        ...
    
    def fit(self, X, y=...): # -> Self@EmpiricalCovariance:
        """Fits the Maximum Likelihood Estimator covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
        """
        ...
    
    def score(self, X_test, y=...):
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.
        """
        ...
    
    def error_norm(self, comp_cov, norm=..., scaling=..., squared=...): # -> Any:
        """Computes the Mean Squared Error between two covariance estimators.
        (In the sense of the Frobenius norm).

        Parameters
        ----------
        comp_cov : array-like of shape (n_features, n_features)
            The covariance to compare with.

        norm : {"frobenius", "spectral"}, default="frobenius"
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool, default=True
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool, default=True
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        result : float
            The Mean Squared Error (in the sense of the Frobenius norm) between
            `self` and `comp_cov` covariance estimators.
        """
        ...
    
    def mahalanobis(self, X): # -> NDArray[signedinteger[Any]]:
        """Computes the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations, the Mahalanobis distances of the which we
            compute. Observations are assumed to be drawn from the same
            distribution than the data used in fit.

        Returns
        -------
        dist : ndarray of shape (n_samples,)
            Squared Mahalanobis distances of the observations.
        """
        ...
    


