"""
This type stub file was generated by pyright.
"""

from typing import Any, AsyncIterable, Callable, Iterable, Sequence, Tuple, Union
from ._metadata import Metadata, MetadataKey, MetadataValue

"""
This type stub file was generated by pyright.
"""
RequestType = ...
ResponseType = ...
SerializingFunction = Callable[[Any], bytes]
DeserializingFunction = Callable[[bytes], Any]
MetadatumType = Tuple[MetadataKey, MetadataValue]
MetadataType = Union[Metadata, Sequence[MetadatumType]]
ChannelArgumentType = Sequence[Tuple[str, Any]]
EOFType = ...
DoneCallbackType = Callable[[Any], None]
RequestIterableType = Union[Iterable[Any], AsyncIterable[Any]]
ResponseIterableType = AsyncIterable[Any]