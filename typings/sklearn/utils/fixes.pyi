"""
This type stub file was generated by pyright.
"""

import scipy
from numpy.ma import MaskedArray as _MaskedArray
from .deprecation import deprecated

"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fixe is no longer needed.
"""
np_version = ...
sp_version = ...
if sp_version >= parse_version('1.4'):
    ...
else:
    ...
class loguniform(scipy.stats.reciprocal):
    """A class supporting log-uniform random variables.

    Parameters
    ----------
    low : float
        The minimum value
    high : float
        The maximum value

    Methods
    -------
    rvs(self, size=None, random_state=None)
        Generate log-uniform random variables

    The most useful method for Scikit-learn usage is highlighted here.
    For a full list, see
    `scipy.stats.reciprocal
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.reciprocal.html>`_.
    This list includes all functions of ``scipy.stats`` continuous
    distributions such as ``pdf``.

    Notes
    -----
    This class generates values between ``low`` and ``high`` or

        low <= loguniform(low, high).rvs() <= high

    The logarithmic probability density function (PDF) is uniform. When
    ``x`` is a uniformly distributed random variable between 0 and 1, ``10**x``
    are random variables that are equally likely to be returned.

    This class is an alias to ``scipy.stats.reciprocal``, which uses the
    reciprocal distribution:
    https://en.wikipedia.org/wiki/Reciprocal_distribution

    Examples
    --------

    >>> from sklearn.utils.fixes import loguniform
    >>> rv = loguniform(1e-3, 1e1)
    >>> rvs = rv.rvs(random_state=42, size=1000)
    >>> rvs.min()  # doctest: +SKIP
    0.0010435856341129003
    >>> rvs.max()  # doctest: +SKIP
    9.97403052786026
    """
    ...


@deprecated('MaskedArray is deprecated in version 0.23 and will be removed in version ' '1.0 (renaming of 0.25). Use numpy.ma.MaskedArray instead.')
class MaskedArray(_MaskedArray):
    ...


def delayed(function): # -> (*args: Unknown, **kwargs: Unknown) -> tuple[_FuncWrapper, tuple[Unknown, ...], dict[str, Unknown]]:
    """Decorator used to capture the arguments of a function."""
    ...

class _FuncWrapper:
    """"Load the global configuration before calling the function."""
    def __init__(self, function) -> None:
        ...
    
    def __call__(self, *args, **kwargs):
        ...
    


