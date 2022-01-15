"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod

"""
Link functions used in GLM
"""
class BaseLink(metaclass=ABCMeta):
    """Abstract base class for Link functions."""
    @abstractmethod
    def __call__(self, y_pred): # -> None:
        """Compute the link function g(y_pred).

        The link function links the mean y_pred=E[Y] to the so called linear
        predictor (X*w), i.e. g(y_pred) = linear predictor.

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Usually the (predicted) mean.
        """
        ...
    
    @abstractmethod
    def derivative(self, y_pred): # -> None:
        """Compute the derivative of the link g'(y_pred).

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Usually the (predicted) mean.
        """
        ...
    
    @abstractmethod
    def inverse(self, lin_pred): # -> None:
        """Compute the inverse link function h(lin_pred).

        Gives the inverse relationship between linear predictor and the mean
        y_pred=E[Y], i.e. h(linear predictor) = y_pred.

        Parameters
        ----------
        lin_pred : array of shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        ...
    
    @abstractmethod
    def inverse_derivative(self, lin_pred): # -> None:
        """Compute the derivative of the inverse link function h'(lin_pred).

        Parameters
        ----------
        lin_pred : array of shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        ...
    


class IdentityLink(BaseLink):
    """The identity link function g(x)=x."""
    def __call__(self, y_pred):
        ...
    
    def derivative(self, y_pred):
        ...
    
    def inverse(self, lin_pred):
        ...
    
    def inverse_derivative(self, lin_pred):
        ...
    


class LogLink(BaseLink):
    """The log link function g(x)=log(x)."""
    def __call__(self, y_pred): # -> Any:
        ...
    
    def derivative(self, y_pred):
        ...
    
    def inverse(self, lin_pred): # -> Any:
        ...
    
    def inverse_derivative(self, lin_pred): # -> Any:
        ...
    


class LogitLink(BaseLink):
    """The logit link function g(x)=logit(x)."""
    def __call__(self, y_pred):
        ...
    
    def derivative(self, y_pred):
        ...
    
    def inverse(self, lin_pred):
        ...
    
    def inverse_derivative(self, lin_pred):
        ...
    


