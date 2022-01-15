"""
This type stub file was generated by pyright.
"""

from . import MinCovDet
from ..utils.validation import _deprecate_positional_args
from ..base import OutlierMixin

class EllipticEnvelope(OutlierMixin, MinCovDet):
    """An object for detecting outliers in a Gaussian distributed dataset.

    Read more in the :ref:`User Guide <outlier_detection>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, the support of robust location and covariance estimates
        is computed, and a covariance estimate is recomputed from it,
        without centering the data.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, the robust location and covariance are directly computed
        with the FastMCD algorithm without additional treatment.

    support_fraction : float, default=None
        The proportion of points to be included in the support of the raw
        MCD estimate. If None, the minimum value of support_fraction will
        be used within the algorithm: `[n_sample + n_features + 1] / 2`.
        Range is (0, 1).

    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Range is (0, 0.5).

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling
        the data. Pass an int for reproducible results across multiple function
        calls. See :term: `Glossary <random_state>`.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated robust location.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated robust covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute the
        robust estimates of location and shape.

    offset_ : float
        Offset used to define the decision function from the raw scores.
        We have the relation: ``decision_function = score_samples - offset_``.
        The offset depends on the contamination parameter and is defined in
        such a way we obtain the expected number of outliers (samples with
        decision function < 0) in training.

        .. versionadded:: 0.20

    raw_location_ : ndarray of shape (n_features,)
        The raw robust estimated location before correction and re-weighting.

    raw_covariance_ : ndarray of shape (n_features, n_features)
        The raw robust estimated covariance before correction and re-weighting.

    raw_support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute
        the raw robust estimates of location and shape, before correction
        and re-weighting.

    dist_ : ndarray of shape (n_samples,)
        Mahalanobis distances of the training set (on which :meth:`fit` is
        called) observations.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import EllipticEnvelope
    >>> true_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> X = np.random.RandomState(0).multivariate_normal(mean=[0, 0],
    ...                                                  cov=true_cov,
    ...                                                  size=500)
    >>> cov = EllipticEnvelope(random_state=0).fit(X)
    >>> # predict returns 1 for an inlier and -1 for an outlier
    >>> cov.predict([[0, 0],
    ...              [3, 3]])
    array([ 1, -1])
    >>> cov.covariance_
    array([[0.7411..., 0.2535...],
           [0.2535..., 0.3053...]])
    >>> cov.location_
    array([0.0813... , 0.0427...])

    See Also
    --------
    EmpiricalCovariance, MinCovDet

    Notes
    -----
    Outlier detection from covariance estimation may break or not
    perform well in high-dimensional settings. In particular, one will
    always take care to work with ``n_samples > n_features ** 2``.

    References
    ----------
    .. [1] Rousseeuw, P.J., Van Driessen, K. "A fast algorithm for the
       minimum covariance determinant estimator" Technometrics 41(3), 212
       (1999)
    """
    @_deprecate_positional_args
    def __init__(self, *, store_precision=..., assume_centered=..., support_fraction=..., contamination=..., random_state=...) -> None:
        ...
    
    def fit(self, X, y=...): # -> Self@EllipticEnvelope:
        """Fit the EllipticEnvelope model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.
        """
        ...
    
    def decision_function(self, X):
        """Compute the decision function of the given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        decision : ndarray of shape (n_samples,)
            Decision function of the samples.
            It is equal to the shifted Mahalanobis distances.
            The threshold for being an outlier is 0, which ensures a
            compatibility with other outlier detection algorithms.
        """
        ...
    
    def score_samples(self, X): # -> NDArray[_NumberType@__neg__]:
        """Compute the negative Mahalanobis distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        negative_mahal_distances : array-like of shape (n_samples,)
            Opposite of the Mahalanobis distances.
        """
        ...
    
    def predict(self, X): # -> ndarray[Unknown, Unknown]:
        """
        Predict the labels (1 inlier, -1 outlier) of X according to the
        fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        ...
    
    def score(self, X, y, sample_weight=...):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        ...
    


