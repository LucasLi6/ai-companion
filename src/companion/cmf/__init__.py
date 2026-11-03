"""Companion Memory Format (CMF) 1.0 constants.

The file operations live in :mod:`companion.cmf.service`; it is imported
explicitly to avoid a circular import with the storage layer.
"""

from .constants import APPLICATION_ID, CMF_FORMAT_VERSION

__all__ = ["APPLICATION_ID", "CMF_FORMAT_VERSION"]
