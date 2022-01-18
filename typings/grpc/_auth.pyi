"""
This type stub file was generated by pyright.
"""

import grpc

"""
This type stub file was generated by pyright.
"""
class GoogleCallCredentials(grpc.AuthMetadataPlugin):
    """Metadata wrapper for GoogleCredentials from the oauth2client library."""
    def __init__(self, credentials) -> None:
        ...
    
    def __call__(self, context, callback):
        ...
    


class AccessTokenAuthMetadataPlugin(grpc.AuthMetadataPlugin):
    """Metadata wrapper for raw access token credentials."""
    def __init__(self, access_token) -> None:
        ...
    
    def __call__(self, context, callback):
        ...
    

