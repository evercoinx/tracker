"""
This type stub file was generated by pyright.
"""

NoCompression = ...
Deflate = ...
Gzip = ...
_METADATA_STRING_MAPPING = ...
def compression_algorithm_to_metadata(compression): # -> tuple[Unknown, str]:
    ...

def create_channel_option(compression): # -> tuple[tuple[Unknown, int]] | tuple[()]:
    ...

def augment_metadata(metadata, compression): # -> tuple[Unknown, ...] | tuple[tuple[Unknown, str]] | tuple[()] | None:
    ...

__all__ = ("NoCompression", "Deflate", "Gzip")
