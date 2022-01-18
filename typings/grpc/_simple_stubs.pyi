"""
This type stub file was generated by pyright.
"""

import datetime
import os
import threading
import grpc
from typing import Any, AnyStr, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union
from grpc.experimental import experimental_api

"""
This type stub file was generated by pyright.
"""
RequestType = ...
ResponseType = ...
OptionsType = Sequence[Tuple[str, str]]
CacheKey = Tuple[str, OptionsType, Optional[grpc.ChannelCredentials], Optional[grpc.Compression]]
_LOGGER = ...
_EVICTION_PERIOD_KEY = ...
if _EVICTION_PERIOD_KEY in os.environ:
    _EVICTION_PERIOD = ...
else:
    _EVICTION_PERIOD = ...
_MAXIMUM_CHANNELS_KEY = ...
if _MAXIMUM_CHANNELS_KEY in os.environ:
    _MAXIMUM_CHANNELS = ...
else:
    _MAXIMUM_CHANNELS = ...
_DEFAULT_TIMEOUT_KEY = ...
if _DEFAULT_TIMEOUT_KEY in os.environ:
    _DEFAULT_TIMEOUT = ...
else:
    _DEFAULT_TIMEOUT = ...
class ChannelCache:
    _singleton = ...
    _lock: threading.RLock = ...
    _condition: threading.Condition = ...
    _eviction_ready: threading.Event = ...
    _mapping: Dict[CacheKey, Tuple[grpc.Channel, datetime.datetime]]
    _eviction_thread: threading.Thread
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def get():
        ...
    
    def get_channel(self, target: str, options: Sequence[Tuple[str, str]], channel_credentials: Optional[grpc.ChannelCredentials], insecure: bool, compression: Optional[grpc.Compression]) -> grpc.Channel:
        ...
    


@experimental_api
def unary_unary(request: RequestType, target: str, method: str, request_serializer: Optional[Callable[[Any], bytes]] = ..., response_deserializer: Optional[Callable[[bytes], Any]] = ..., options: Sequence[Tuple[AnyStr, AnyStr]] = ..., channel_credentials: Optional[grpc.ChannelCredentials] = ..., insecure: bool = ..., call_credentials: Optional[grpc.CallCredentials] = ..., compression: Optional[grpc.Compression] = ..., wait_for_ready: Optional[bool] = ..., timeout: Optional[float] = ..., metadata: Optional[Sequence[Tuple[str, Union[str, bytes]]]] = ...) -> ResponseType:
    """Invokes a unary-unary RPC without an explicitly specified channel.

    THIS IS AN EXPERIMENTAL API.

    This is backed by a per-process cache of channels. Channels are evicted
    from the cache after a fixed period by a background. Channels will also be
    evicted if more than a configured maximum accumulate.

    The default eviction period is 10 minutes. One may set the environment
    variable "GRPC_PYTHON_MANAGED_CHANNEL_EVICTION_SECONDS" to configure this.

    The default maximum number of channels is 256. One may set the
    environment variable "GRPC_PYTHON_MANAGED_CHANNEL_MAXIMUM" to configure
    this.

    Args:
      request: An iterator that yields request values for the RPC.
      target: The server address.
      method: The name of the RPC method.
      request_serializer: Optional :term:`serializer` for serializing the request
        message. Request goes unserialized in case None is passed.
      response_deserializer: Optional :term:`deserializer` for deserializing the response
        message. Response goes undeserialized in case None is passed.
      options: An optional list of key-value pairs (:term:`channel_arguments` in gRPC Core
        runtime) to configure the channel.
      channel_credentials: A credential applied to the whole channel, e.g. the
        return value of grpc.ssl_channel_credentials() or
        grpc.insecure_channel_credentials().
      insecure: If True, specifies channel_credentials as
        :term:`grpc.insecure_channel_credentials()`. This option is mutually
        exclusive with the `channel_credentials` option.
      call_credentials: A call credential applied to each call individually,
        e.g. the output of grpc.metadata_call_credentials() or
        grpc.access_token_call_credentials().
      compression: An optional value indicating the compression method to be
        used over the lifetime of the channel, e.g. grpc.Compression.Gzip.
      wait_for_ready: An optional flag indicating whether the RPC should fail
        immediately if the connection is not ready at the time the RPC is
        invoked, or if it should wait until the connection to the server
        becomes ready. When using this option, the user will likely also want
        to set a timeout. Defaults to True.
      timeout: An optional duration of time in seconds to allow for the RPC,
        after which an exception will be raised. If timeout is unspecified,
        defaults to a timeout controlled by the
        GRPC_PYTHON_DEFAULT_TIMEOUT_SECONDS environment variable. If that is
        unset, defaults to 60 seconds. Supply a value of None to indicate that
        no timeout should be enforced.
      metadata: Optional metadata to send to the server.

    Returns:
      The response to the RPC.
    """
    ...

@experimental_api
def unary_stream(request: RequestType, target: str, method: str, request_serializer: Optional[Callable[[Any], bytes]] = ..., response_deserializer: Optional[Callable[[bytes], Any]] = ..., options: Sequence[Tuple[AnyStr, AnyStr]] = ..., channel_credentials: Optional[grpc.ChannelCredentials] = ..., insecure: bool = ..., call_credentials: Optional[grpc.CallCredentials] = ..., compression: Optional[grpc.Compression] = ..., wait_for_ready: Optional[bool] = ..., timeout: Optional[float] = ..., metadata: Optional[Sequence[Tuple[str, Union[str, bytes]]]] = ...) -> Iterator[ResponseType]:
    """Invokes a unary-stream RPC without an explicitly specified channel.

    THIS IS AN EXPERIMENTAL API.

    This is backed by a per-process cache of channels. Channels are evicted
    from the cache after a fixed period by a background. Channels will also be
    evicted if more than a configured maximum accumulate.

    The default eviction period is 10 minutes. One may set the environment
    variable "GRPC_PYTHON_MANAGED_CHANNEL_EVICTION_SECONDS" to configure this.

    The default maximum number of channels is 256. One may set the
    environment variable "GRPC_PYTHON_MANAGED_CHANNEL_MAXIMUM" to configure
    this.

    Args:
      request: An iterator that yields request values for the RPC.
      target: The server address.
      method: The name of the RPC method.
      request_serializer: Optional :term:`serializer` for serializing the request
        message. Request goes unserialized in case None is passed.
      response_deserializer: Optional :term:`deserializer` for deserializing the response
        message. Response goes undeserialized in case None is passed.
      options: An optional list of key-value pairs (:term:`channel_arguments` in gRPC Core
        runtime) to configure the channel.
      channel_credentials: A credential applied to the whole channel, e.g. the
        return value of grpc.ssl_channel_credentials().
      insecure: If True, specifies channel_credentials as
        :term:`grpc.insecure_channel_credentials()`. This option is mutually
        exclusive with the `channel_credentials` option.
      call_credentials: A call credential applied to each call individually,
        e.g. the output of grpc.metadata_call_credentials() or
        grpc.access_token_call_credentials().
      compression: An optional value indicating the compression method to be
        used over the lifetime of the channel, e.g. grpc.Compression.Gzip.
      wait_for_ready: An optional flag indicating whether the RPC should fail
        immediately if the connection is not ready at the time the RPC is
        invoked, or if it should wait until the connection to the server
        becomes ready. When using this option, the user will likely also want
        to set a timeout. Defaults to True.
      timeout: An optional duration of time in seconds to allow for the RPC,
        after which an exception will be raised. If timeout is unspecified,
        defaults to a timeout controlled by the
        GRPC_PYTHON_DEFAULT_TIMEOUT_SECONDS environment variable. If that is
        unset, defaults to 60 seconds. Supply a value of None to indicate that
        no timeout should be enforced.
      metadata: Optional metadata to send to the server.

    Returns:
      An iterator of responses.
    """
    ...

@experimental_api
def stream_unary(request_iterator: Iterator[RequestType], target: str, method: str, request_serializer: Optional[Callable[[Any], bytes]] = ..., response_deserializer: Optional[Callable[[bytes], Any]] = ..., options: Sequence[Tuple[AnyStr, AnyStr]] = ..., channel_credentials: Optional[grpc.ChannelCredentials] = ..., insecure: bool = ..., call_credentials: Optional[grpc.CallCredentials] = ..., compression: Optional[grpc.Compression] = ..., wait_for_ready: Optional[bool] = ..., timeout: Optional[float] = ..., metadata: Optional[Sequence[Tuple[str, Union[str, bytes]]]] = ...) -> ResponseType:
    """Invokes a stream-unary RPC without an explicitly specified channel.

    THIS IS AN EXPERIMENTAL API.

    This is backed by a per-process cache of channels. Channels are evicted
    from the cache after a fixed period by a background. Channels will also be
    evicted if more than a configured maximum accumulate.

    The default eviction period is 10 minutes. One may set the environment
    variable "GRPC_PYTHON_MANAGED_CHANNEL_EVICTION_SECONDS" to configure this.

    The default maximum number of channels is 256. One may set the
    environment variable "GRPC_PYTHON_MANAGED_CHANNEL_MAXIMUM" to configure
    this.

    Args:
      request_iterator: An iterator that yields request values for the RPC.
      target: The server address.
      method: The name of the RPC method.
      request_serializer: Optional :term:`serializer` for serializing the request
        message. Request goes unserialized in case None is passed.
      response_deserializer: Optional :term:`deserializer` for deserializing the response
        message. Response goes undeserialized in case None is passed.
      options: An optional list of key-value pairs (:term:`channel_arguments` in gRPC Core
        runtime) to configure the channel.
      channel_credentials: A credential applied to the whole channel, e.g. the
        return value of grpc.ssl_channel_credentials().
      call_credentials: A call credential applied to each call individually,
        e.g. the output of grpc.metadata_call_credentials() or
        grpc.access_token_call_credentials().
      insecure: If True, specifies channel_credentials as
        :term:`grpc.insecure_channel_credentials()`. This option is mutually
        exclusive with the `channel_credentials` option.
      compression: An optional value indicating the compression method to be
        used over the lifetime of the channel, e.g. grpc.Compression.Gzip.
      wait_for_ready: An optional flag indicating whether the RPC should fail
        immediately if the connection is not ready at the time the RPC is
        invoked, or if it should wait until the connection to the server
        becomes ready. When using this option, the user will likely also want
        to set a timeout. Defaults to True.
      timeout: An optional duration of time in seconds to allow for the RPC,
        after which an exception will be raised. If timeout is unspecified,
        defaults to a timeout controlled by the
        GRPC_PYTHON_DEFAULT_TIMEOUT_SECONDS environment variable. If that is
        unset, defaults to 60 seconds. Supply a value of None to indicate that
        no timeout should be enforced.
      metadata: Optional metadata to send to the server.

    Returns:
      The response to the RPC.
    """
    ...

@experimental_api
def stream_stream(request_iterator: Iterator[RequestType], target: str, method: str, request_serializer: Optional[Callable[[Any], bytes]] = ..., response_deserializer: Optional[Callable[[bytes], Any]] = ..., options: Sequence[Tuple[AnyStr, AnyStr]] = ..., channel_credentials: Optional[grpc.ChannelCredentials] = ..., insecure: bool = ..., call_credentials: Optional[grpc.CallCredentials] = ..., compression: Optional[grpc.Compression] = ..., wait_for_ready: Optional[bool] = ..., timeout: Optional[float] = ..., metadata: Optional[Sequence[Tuple[str, Union[str, bytes]]]] = ...) -> Iterator[ResponseType]:
    """Invokes a stream-stream RPC without an explicitly specified channel.

    THIS IS AN EXPERIMENTAL API.

    This is backed by a per-process cache of channels. Channels are evicted
    from the cache after a fixed period by a background. Channels will also be
    evicted if more than a configured maximum accumulate.

    The default eviction period is 10 minutes. One may set the environment
    variable "GRPC_PYTHON_MANAGED_CHANNEL_EVICTION_SECONDS" to configure this.

    The default maximum number of channels is 256. One may set the
    environment variable "GRPC_PYTHON_MANAGED_CHANNEL_MAXIMUM" to configure
    this.

    Args:
      request_iterator: An iterator that yields request values for the RPC.
      target: The server address.
      method: The name of the RPC method.
      request_serializer: Optional :term:`serializer` for serializing the request
        message. Request goes unserialized in case None is passed.
      response_deserializer: Optional :term:`deserializer` for deserializing the response
        message. Response goes undeserialized in case None is passed.
      options: An optional list of key-value pairs (:term:`channel_arguments` in gRPC Core
        runtime) to configure the channel.
      channel_credentials: A credential applied to the whole channel, e.g. the
        return value of grpc.ssl_channel_credentials().
      call_credentials: A call credential applied to each call individually,
        e.g. the output of grpc.metadata_call_credentials() or
        grpc.access_token_call_credentials().
      insecure: If True, specifies channel_credentials as
        :term:`grpc.insecure_channel_credentials()`. This option is mutually
        exclusive with the `channel_credentials` option.
      compression: An optional value indicating the compression method to be
        used over the lifetime of the channel, e.g. grpc.Compression.Gzip.
      wait_for_ready: An optional flag indicating whether the RPC should fail
        immediately if the connection is not ready at the time the RPC is
        invoked, or if it should wait until the connection to the server
        becomes ready. When using this option, the user will likely also want
        to set a timeout. Defaults to True.
      timeout: An optional duration of time in seconds to allow for the RPC,
        after which an exception will be raised. If timeout is unspecified,
        defaults to a timeout controlled by the
        GRPC_PYTHON_DEFAULT_TIMEOUT_SECONDS environment variable. If that is
        unset, defaults to 60 seconds. Supply a value of None to indicate that
        no timeout should be enforced.
      metadata: Optional metadata to send to the server.

    Returns:
      An iterator of responses.
    """
    ...
