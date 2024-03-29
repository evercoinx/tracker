"""
This type stub file was generated by pyright.
"""

from ._base import LinearModel
from ..base import MultiOutputMixin, RegressorMixin
from ..utils.validation import _deprecate_positional_args

"""
Least Angle Regression algorithm. See the documentation on the
Generalized Linear Model for a complete discussion.
"""
SOLVE_TRIANGULAR_ARGS = ...
@_deprecate_positional_args
def lars_path(X, y, Xy=..., *, Gram=..., max_iter=..., alpha_min=..., method=..., copy_X=..., eps=..., copy_Gram=..., verbose=..., return_path=..., return_n_iter=..., positive=...):
    """Compute Least Angle Regression or Lasso path using LARS algorithm [1]

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    X : None or array-like of shape (n_samples, n_features)
        Input data. Note that if X is None then the Gram matrix must be
        specified, i.e., cannot be None or False.

    y : None or array-like of shape (n_samples,)
        Input targets.

    Xy : array-like of shape (n_samples,) or (n_samples, n_targets), \
            default=None
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    Gram : None, 'auto', array-like of shape (n_features, n_features), \
            default=None
        Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.

    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, default=True
        If ``False``, ``X`` is overwritten.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_Gram : bool, default=True
        If ``False``, ``Gram`` is overwritten.

    verbose : int, default=0
        Controls output verbosity.

    return_path : bool, default=True
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, default=False
        Whether to return the number of iterations.

    positive : bool, default=False
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.

    Returns
    -------
    alphas : array-like of shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.

    active : array-like of shape (n_alphas,)
        Indices of active variables at the end of the path.

    coefs : array-like of shape (n_features, n_alphas + 1)
        Coefficients along the path

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.

    See Also
    --------
    lars_path_gram
    lasso_path
    lasso_path_gram
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    """
    ...

@_deprecate_positional_args
def lars_path_gram(Xy, Gram, *, n_samples, max_iter=..., alpha_min=..., method=..., copy_X=..., eps=..., copy_Gram=..., verbose=..., return_path=..., return_n_iter=..., positive=...):
    """lars_path in the sufficient stats mode [1]

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    Xy : array-like of shape (n_samples,) or (n_samples, n_targets)
        Xy = np.dot(X.T, y).

    Gram : array-like of shape (n_features, n_features)
        Gram = np.dot(X.T * X).

    n_samples : int or float
        Equivalent size of sample.

    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, default=True
        If ``False``, ``X`` is overwritten.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_Gram : bool, default=True
        If ``False``, ``Gram`` is overwritten.

    verbose : int, default=0
        Controls output verbosity.

    return_path : bool, default=True
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, default=False
        Whether to return the number of iterations.

    positive : bool, default=False
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.

    Returns
    -------
    alphas : array-like of shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.

    active : array-like of shape (n_alphas,)
        Indices of active variables at the end of the path.

    coefs : array-like of shape (n_features, n_alphas + 1)
        Coefficients along the path

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.

    See Also
    --------
    lars_path
    lasso_path
    lasso_path_gram
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    """
    ...

class Lars(MultiOutputMixin, RegressorMixin, LinearModel):
    """Least Angle Regression model a.k.a. LAR

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    normalize : bool, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : bool, 'auto' or array-like , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    n_nonzero_coefs : int, default=500
        Target number of non-zero coefficients. Use ``np.inf`` for no limit.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_path : bool, default=True
        If True the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    jitter : float, default=None
        Upper bound on a uniform noise parameter to be added to the
        `y` values, to satisfy the model's assumption of
        one-at-a-time computations. Might help with stability.

        .. versionadded:: 0.23

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for jittering. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`. Ignored if `jitter` is None.

        .. versionadded:: 0.23

    Attributes
    ----------
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If this is a list of array-like, the length of the outer
        list is `n_targets`.

    active_ : list of shape (n_alphas,) or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of list, the length of the outer list is `n_targets`.

    coef_path_ : array-like of shape (n_features, n_alphas + 1) or list \
            of such arrays
        The varying values of the coefficients along the path. It is not
        present if the ``fit_path`` parameter is ``False``. If this is a list
        of array-like, the length of the outer list is `n_targets`.

    coef_ : array-like of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formulation formula).

    intercept_ : float or array-like of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : array-like or int
        The number of iterations taken by lars_path to find the
        grid of alphas for each target.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.Lars(n_nonzero_coefs=1)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
    Lars(n_nonzero_coefs=1)
    >>> print(reg.coef_)
    [ 0. -1.11...]

    See Also
    --------
    lars_path, LarsCV
    sklearn.decomposition.sparse_encode

    """
    method = ...
    positive = ...
    @_deprecate_positional_args
    def __init__(self, *, fit_intercept=..., verbose=..., normalize=..., precompute=..., n_nonzero_coefs=..., eps=..., copy_X=..., fit_path=..., jitter=..., random_state=...) -> None:
        ...
    
    def fit(self, X, y, Xy=...): # -> Self@Lars:
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Xy : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Xy = np.dot(X.T, y) that can be precomputed. It is useful
            only when the Gram matrix is precomputed.

        Returns
        -------
        self : object
            returns an instance of self.
        """
        ...
    


class LassoLars(Lars):
    """Lasso model fit with Least Angle Regression a.k.a. Lars

    It is a Linear Model trained with an L1 prior as regularizer.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by :class:`LinearRegression`. For numerical reasons, using
        ``alpha = 0`` with the LassoLars object is not advised and you
        should prefer the LinearRegression object.

    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    normalize : bool, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    fit_path : bool, default=True
        If ``True`` the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients will not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.

    jitter : float, default=None
        Upper bound on a uniform noise parameter to be added to the
        `y` values, to satisfy the model's assumption of
        one-at-a-time computations. Might help with stability.

        .. versionadded:: 0.23

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for jittering. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`. Ignored if `jitter` is None.

        .. versionadded:: 0.23

    Attributes
    ----------
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If this is a list of array-like, the length of the outer
        list is `n_targets`.

    active_ : list of length n_alphas or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of list, the length of the outer list is `n_targets`.

    coef_path_ : array-like of shape (n_features, n_alphas + 1) or list \
            of such arrays
        If a list is passed it's expected to be one of n_targets such arrays.
        The varying values of the coefficients along the path. It is not
        present if the ``fit_path`` parameter is ``False``. If this is a list
        of array-like, the length of the outer list is `n_targets`.

    coef_ : array-like of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formulation formula).

    intercept_ : float or array-like of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : array-like or int
        The number of iterations taken by lars_path to find the
        grid of alphas for each target.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.LassoLars(alpha=0.01)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
    LassoLars(alpha=0.01)
    >>> print(reg.coef_)
    [ 0.         -0.963257...]

    See Also
    --------
    lars_path
    lasso_path
    Lasso
    LassoCV
    LassoLarsCV
    LassoLarsIC
    sklearn.decomposition.sparse_encode

    """
    method = ...
    @_deprecate_positional_args
    def __init__(self, alpha=..., *, fit_intercept=..., verbose=..., normalize=..., precompute=..., max_iter=..., eps=..., copy_X=..., fit_path=..., positive=..., jitter=..., random_state=...) -> None:
        ...
    


class LarsCV(Lars):
    """Cross-validated Least Angle Regression model.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    normalize : bool, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : bool, 'auto' or array-like , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    max_n_alphas : int, default=1000
        The maximum number of points on the path used to compute the
        residuals in the cross-validation

    n_jobs : int or None, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    active_ : list of length n_alphas or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of lists, the outer list length is `n_targets`.

    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function

    coef_path_ : array-like of shape (n_features, n_alphas)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha

    alphas_ : array-like of shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array-like of shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds

    mse_path_ : array-like of shape (n_folds, n_cv_alphas)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.

    Examples
    --------
    >>> from sklearn.linear_model import LarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
    >>> reg = LarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9996...
    >>> reg.alpha_
    0.0254...
    >>> reg.predict(X[:1,])
    array([154.0842...])

    See Also
    --------
    lars_path, LassoLars, LassoLarsCV
    """
    method = ...
    @_deprecate_positional_args
    def __init__(self, *, fit_intercept=..., verbose=..., max_iter=..., normalize=..., precompute=..., cv=..., max_n_alphas=..., n_jobs=..., eps=..., copy_X=...) -> None:
        ...
    
    def fit(self, X, y): # -> Self@LarsCV:
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            returns an instance of self.
        """
        ...
    


class LassoLarsCV(LarsCV):
    """Cross-validated Lasso, using the LARS algorithm.

    See glossary entry for :term:`cross-validation estimator`.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    normalize : bool, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : bool or 'auto' , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    max_n_alphas : int, default=1000
        The maximum number of points on the path used to compute the
        residuals in the cross-validation

    n_jobs : int or None, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients do not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.
        As a consequence using LassoLarsCV only makes sense for problems where
        a sparse solution is expected and/or reached.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function.

    coef_path_ : array-like of shape (n_features, n_alphas)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha

    alphas_ : array-like of shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array-like of shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds

    mse_path_ : array-like of shape (n_folds, n_cv_alphas)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.

    active_ : list of int
        Indices of active variables at the end of the path.

    Examples
    --------
    >>> from sklearn.linear_model import LassoLarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4.0, random_state=0)
    >>> reg = LassoLarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9992...
    >>> reg.alpha_
    0.0484...
    >>> reg.predict(X[:1,])
    array([-77.8723...])

    Notes
    -----

    The object solves the same problem as the LassoCV object. However,
    unlike the LassoCV, it find the relevant alphas values by itself.
    In general, because of this property, it will be more stable.
    However, it is more fragile to heavily multicollinear datasets.

    It is more efficient than the LassoCV if only a small number of
    features are selected compared to the total number, for instance if
    there are very few samples compared to the number of features.

    See Also
    --------
    lars_path, LassoLars, LarsCV, LassoCV
    """
    method = ...
    @_deprecate_positional_args
    def __init__(self, *, fit_intercept=..., verbose=..., max_iter=..., normalize=..., precompute=..., cv=..., max_n_alphas=..., n_jobs=..., eps=..., copy_X=..., positive=...) -> None:
        ...
    


class LassoLarsIC(LassoLars):
    """Lasso model fit with Lars using BIC or AIC for model selection

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    AIC is the Akaike information criterion and BIC is the Bayes
    Information criterion. Such criteria are useful to select the value
    of the regularization parameter by making a trade-off between the
    goodness of fit and the complexity of the model. A good model should
    explain well the data while being simple.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    criterion : {'bic' , 'aic'}, default='aic'
        The type of criterion to use.

    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    normalize : bool, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=500
        Maximum number of iterations to perform. Can be used for
        early stopping.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients do not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.
        As a consequence using LassoLarsIC only makes sense for problems where
        a sparse solution is expected and/or reached.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function.

    alpha_ : float
        the alpha parameter chosen by the information criterion

    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If a list, it will be of length `n_targets`.

    n_iter_ : int
        number of iterations run by lars_path to find the grid of
        alphas.

    criterion_ : array-like of shape (n_alphas,)
        The value of the information criteria ('aic', 'bic') across all
        alphas. The alpha which has the smallest information criterion is
        chosen. This value is larger by a factor of ``n_samples`` compared to
        Eqns. 2.15 and 2.16 in (Zou et al, 2007).


    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.LassoLarsIC(criterion='bic')
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
    LassoLarsIC(criterion='bic')
    >>> print(reg.coef_)
    [ 0.  -1.11...]

    Notes
    -----
    The estimation of the number of degrees of freedom is given by:

    "On the degrees of freedom of the lasso"
    Hui Zou, Trevor Hastie, and Robert Tibshirani
    Ann. Statist. Volume 35, Number 5 (2007), 2173-2192.

    https://en.wikipedia.org/wiki/Akaike_information_criterion
    https://en.wikipedia.org/wiki/Bayesian_information_criterion

    See Also
    --------
    lars_path, LassoLars, LassoLarsCV
    """
    @_deprecate_positional_args
    def __init__(self, criterion=..., *, fit_intercept=..., verbose=..., normalize=..., precompute=..., max_iter=..., eps=..., copy_X=..., positive=...) -> None:
        ...
    
    def fit(self, X, y, copy_X=...): # -> Self@LassoLarsIC:
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training data.

        y : array-like of shape (n_samples,)
            target values. Will be cast to X's dtype if necessary

        copy_X : bool, default=None
            If provided, this parameter will override the choice
            of copy_X made at instance creation.
            If ``True``, X will be copied; else, it may be overwritten.

        Returns
        -------
        self : object
            returns an instance of self.
        """
        ...
    


