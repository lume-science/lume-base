import importlib

# prevent circular import by only importing on type-check
from typing import TYPE_CHECKING

import h5py

from .base import ClassInitError, ModuleImportError, SerializerBase

if TYPE_CHECKING:
    from lume.base import Base


class HDF5Serializer(SerializerBase):
    """Class used for serializing Base objects to hdf5 files."""

    # semantic version tuple (major, minor, patch)
    _version = (0, 0, 0)

    # make effort at backwards compat to 1.8
    _libver = ("earliest", "v108")

    def serialize(self, filename: str, object: "Base") -> None:
        """Serialize Base object into a self-describing hdf5 file.

        Args:
            filename (str): Name of saved file
            object (Base): Object to serialize

        """

        with h5py.File(filename, "w", libver=self._libver) as f:
            f.attrs.create(
                "_version",
                self._version,
            )

            object_import_path = f"{object.__module__}.{type(object).__name__}"
            f.attrs.create(
                "object",
                object_import_path,
            )

            parent_module = importlib.import_module(object.__module__.split(".")[0])

            f.attrs.create(
                "_pkg_version",
                parent_module.__version__,
            )

            object.archive(f)

    @classmethod
    def deserialize(cls, filename: str) -> "Base":
        """Deserialize hdf5 file and load described object.

        Args:
            filename (str): Name of file to load.

        Return
            Base

        """

        with h5py.File(filename, "r", libver=cls._libver) as f:
            # Because we've stored things like version, object, and package version, we can do integrity checks
            cls._check_compat(f.attrs.get("_version"))

            object_import_split = f.attrs.get("object").split(".")
            package_str = ".".join(object_import_split[:-1])
            object_name = object_import_split[-1]
            package_version = f.attrs.get("_pkg_version")

            # check version compat of serialized vs installed package
            try:
                object_import_module = importlib.import_module(package_str)
            except ImportError:
                raise ModuleImportError(package_str, package_version)

            object_type = getattr(object_import_module, object_name, None)

            if object_type is None:
                raise ClassInitError(object_name, package_str, package_version)

            object = object_type()
            object.load_archive(f)

            return object

    @classmethod
    def _check_compat(cls, version):
        """Check class version against version for compatability. Currently no constraints."""
        pass
