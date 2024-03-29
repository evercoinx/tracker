"""
This type stub file was generated by pyright.
"""

import collections
import grpc

"""
This type stub file was generated by pyright.
"""
_LOGGER = ...
class _AuthMetadataContext(collections.namedtuple('AuthMetadataContext', ('service_url', 'method_name')), grpc.AuthMetadataContext):
    ...


class _CallbackState:
    def __init__(self) -> None:
        ...
    


class _AuthMetadataPluginCallback(grpc.AuthMetadataPluginCallback):
    def __init__(self, state, callback) -> None:
        ...
    
    def __call__(self, metadata, error):
        ...
    


class _Plugin:
    def __init__(self, metadata_plugin) -> None:
        ...
    
    def __call__(self, service_url, method_name, callback):
        ...
    


def metadata_plugin_call_credentials(metadata_plugin, name):
    ...

