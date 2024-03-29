"""
This type stub file was generated by pyright.
"""

import copy
import functools
import sys
import warnings
import grpc
from grpc._cython import cygrpc as _cygrpc
from grpc._simple_stubs import stream_stream, stream_unary, unary_stream, unary_unary

"""
This type stub file was generated by pyright.
"""
_EXPERIMENTAL_APIS_USED = ...
class ChannelOptions:
    """Indicates a channel option unique to gRPC Python.

     This enumeration is part of an EXPERIMENTAL API.

     Attributes:
       SingleThreadedUnaryStream: Perform unary-stream RPCs on a single thread.
    """
    SingleThreadedUnaryStream = ...


class UsageError(Exception):
    """Raised by the gRPC library to indicate usage not allowed by the API."""
    ...


_insecure_channel_credentials = ...
def insecure_channel_credentials():
    """Creates a ChannelCredentials for use with an insecure channel.

    THIS IS AN EXPERIMENTAL API.
    """
    ...

class ExperimentalApiWarning(Warning):
    """A warning that an API is experimental."""
    ...


def experimental_api(f):
    ...

def wrap_server_method_handler(wrapper, handler):
    """Wraps the server method handler function.

    The server implementation requires all server handlers being wrapped as
    RpcMethodHandler objects. This helper function ease the pain of writing
    server handler wrappers.

    Args:
        wrapper: A wrapper function that takes in a method handler behavior
          (the actual function) and returns a wrapped function.
        handler: A RpcMethodHandler object to be wrapped.

    Returns:
        A newly created RpcMethodHandler.
    """
    ...

__all__ = ('ChannelOptions', 'ExperimentalApiWarning', 'UsageError', 'insecure_channel_credentials', 'wrap_server_method_handler')
if sys.version_info > (3, 6):
    ...
