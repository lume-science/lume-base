"""N-dimensional array variable definitions for LUME-model variables.

This module provides concrete variable classes for handling N-dimensional
array data. The base NDVariable class works directly with NumPy ndarrays,
and the design allows for easy extensibility to support other array types
(e.g., PyTorch tensors) by subclassing NDVariable and overriding
array_type, dtype, and dtype_attribute.

"""

import typing
from typing import Any, ClassVar, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    ConfigDict,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from lume.variables.variable import ConfigEnum, Variable


class NDVariable(Variable):
    """Base class for N-dimensional array variables.

    This class provides validation for NumPy ndarray data with specific shape
    requirements.

    Subclasses can implement other array types (e.g., PyTorch tensors)
    by overriding array_type, dtype, and dtype_attribute.

    Attributes
    ----------
    shape : Tuple[int, ...]
        Expected shape of the array. Values must match this shape exactly.
    dtype : np.dtype
        Expected NumPy data type of the array. Defaults to np.float64.
    default_value : NDArray | None
        Default value for the variable. Must match the expected shape and dtype
        if provided. Defaults to None.
    unit : str | None
        Physical unit associated with the variable (e.g., "m", "GeV", "rad").
        Defaults to None.

    Notes
    -----
    Subclasses should override:
    - array_type: The expected array class (default: np.ndarray)
    - dtype: With the appropriate type annotation for their array implementation
    - dtype_attribute: The attribute name to access dtype (default: "dtype")

    Examples
    --------
    >>> import numpy as np
    >>> from lume.variables.ndvariable import NDVariable
    >>>
    >>> # Create a variable for 2D arrays with shape (3, 4)
    >>> var = NDVariable(name="my_array", shape=(3, 4))
    >>>
    >>> # Validate a matching array
    >>> arr = np.ones((3, 4))
    >>> var.validate_value(arr, config="error")

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    shape: Tuple[int, ...]
    dtype: np.dtype = np.dtype(np.float64)
    default_value: Optional[NDArray] = None
    unit: Optional[str] = None

    # Class attributes - subclasses can override
    array_type: ClassVar[type] = np.ndarray
    dtype_attribute: ClassVar[str] = "dtype"

    @classmethod
    def _dtype_coerce(cls, value: Any) -> Any:
        """Coerce a raw value (e.g. a string from JSON) to the appropriate dtype object.

        Override this in subclasses that use a dtype type other than
        np.dtype. The base implementation tries np.dtype(value).

        Parameters
        ----------
        value : Any
            The raw value to coerce.

        Returns
        -------
        Any
            The coerced dtype, e.g. np.dtype("float64").

        Raises
        ------
        TypeError
            If the value cannot be coerced.

        """
        return np.dtype(value)

    @field_serializer("dtype")
    def serialize_dtype(self, value: np.dtype) -> str:
        """Serialize np.dtype to its string name (e.g. 'float64').

        Parameters
        ----------
        value : np.dtype
            The dtype to serialize.

        Returns
        -------
        str
            The NumPy dtype name string (e.g. "float64", "int32").

        """
        return np.dtype(value).name

    @field_serializer("default_value")
    def serialize_default_value(self, value: Optional[NDArray]) -> Optional[List]:
        """Serialize a NumPy ndarray to a nested Python list.

        Parameters
        ----------
        value : NDArray or None
            The array to serialize, or None.

        Returns
        -------
        list or None
            A nested Python list representation of the array, or None if
            no default value is set.

        """
        if value is None:
            return None
        return value.tolist()

    @field_validator("default_value", mode="before")
    @classmethod
    def coerce_default_value(cls, value: Any, info: ValidationInfo) -> Any:
        """Coerce list or tuple input to np.ndarray for round-trip deserialization.

        When a model is reconstructed from a serialized dict (e.g. loaded
        from JSON or YAML), default_value arrives as a nested list.
        This validator converts it back to a NumPy array so that the model
        invariants are maintained.

        The array is built using the model's own ``dtype`` (already validated,
        since ``dtype`` is declared before ``default_value``). Without this the
        list would default to ``float64``/``int64`` and fail the exact-dtype
        check in ``validate_default_value`` for any other dtype.

        Parameters
        ----------
        value : Any
            Raw input value. If it is a list or tuple it is
            converted to np.ndarray; otherwise it is returned unchanged.
        info : ValidationInfo
            Pydantic validation context, used to read the already-validated
            ``dtype`` field.

        Returns
        -------
        np.ndarray or None or Any
            The (possibly converted) value.

        """
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=info.data.get("dtype"))
        return value

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype_field(cls, value: Any) -> Any:
        """Validate and coerce the dtype field.

        Accepts np.dtype instances directly, or a string/type that can be
        coerced to np.dtype (e.g. "float64" from a JSON round-trip).
        Rejects values that cannot be interpreted as a NumPy dtype.

        Parameters
        ----------
        value : Any
            The dtype value to validate.

        Returns
        -------
        np.dtype
            The validated (and possibly coerced) dtype.

        Raises
        ------
        TypeError
            If the value cannot be coerced to the expected dtype type.

        Notes
        -----
        Subclasses that use a different dtype type (e.g. torch.dtype) should
        annotate their dtype field accordingly; the string-coercion path is
        only applied when the expected annotation is np.dtype.

        """
        # Get the expected type from the field annotation
        if hasattr(cls, "__annotations__") and "dtype" in cls.__annotations__:
            expected_type = cls.__annotations__["dtype"]

            # Handle Optional types (unwrap Optional to get the actual type)
            if (
                hasattr(typing, "get_origin")
                and typing.get_origin(expected_type) is typing.Union
            ):
                args = typing.get_args(expected_type)
                expected_type = next(
                    (arg for arg in args if arg is not type(None)), expected_type
                )

            if not isinstance(value, expected_type):
                # Attempt to coerce from string or type (e.g. round-trip from
                # JSON/YAML).  Subclasses provide the mapping via _dtype_coerce.
                try:
                    return cls._dtype_coerce(value)
                except (TypeError, KeyError, AttributeError):
                    pass
                raise TypeError(
                    f"dtype must be a {expected_type.__name__} instance, "
                    f"got {type(value).__name__}. "
                    f"Received value: {repr(value)}"
                )

        return value

    def _validate_array_type(self, value: Any) -> None:
        """Validate that value is the correct array type.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value is not of the expected array type (np.ndarray by default).

        """
        if not isinstance(value, self.array_type):
            raise TypeError(
                f"Expected value to be a {self.array_type.__name__}, "
                f"but received {type(value).__name__}."
            )

    def _validate_dtype(self, value: Any, expected_dtype: Any) -> None:
        """Validate the dtype of the array.

        Parameters
        ----------
        value : Any
            The array whose dtype should be validated.
        expected_dtype : Any
            The expected data type.

        Raises
        ------
        AttributeError
            If the array does not have the expected dtype attribute.
        ValueError
            If the array's dtype does not match the expected dtype.

        Notes
        -----
        Subclasses can override dtype_attribute if their array type uses
        a different attribute name to access dtype (default is "dtype").
        The dtype must match exactly; no implicit type conversions are performed.

        """
        if expected_dtype is None:
            return

        if not hasattr(value, self.dtype_attribute):
            raise AttributeError(
                f"Array value does not have a '{self.dtype_attribute}' attribute"
            )

        actual_dtype = getattr(value, self.dtype_attribute)

        if actual_dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype}, got {actual_dtype}")

    def _validate_shape(
        self, value: Any, expected_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """Validate that the array shape exactly matches expected_shape.

        Parameters
        ----------
        value : Any
            The array whose shape should be validated.
        expected_shape : Tuple[int, ...] | None, optional
            The expected shape. If None, no shape validation is performed.

        Raises
        ------
        ValueError
            If the array's shape does not exactly match the expected shape.

        """
        if expected_shape is not None:
            actual_shape = tuple(value.shape)

            if actual_shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {actual_shape}")

    @model_validator(mode="after")
    def validate_default_value(self) -> "NDVariable":
        """Validate the default_value if provided.

        Returns
        -------
        NDVariable
            The validated instance.

        Raises
        ------
        TypeError, ValueError
            If default_value fails validation (wrong type, dtype, or shape).

        """
        if self.default_value is not None:
            self.validate_value(self.default_value, ConfigEnum.ERROR)
        return self

    def validate_value(self, value: Any, config: Optional[ConfigEnum] = None) -> None:
        """Validate the given array value.

        Performs mandatory validation (type, dtype, shape) and optional
        validation based on the configuration.

        Parameters
        ----------
        value : Any
            The NumPy ndarray value to validate.
        config : ConfigEnum | None, optional
            The validation configuration. Defaults to None (uses
            default_validation_config). Allowed values are "none", "warn",
            and "error".

        Raises
        ------
        TypeError
            If the value is not an ndarray of the expected type.
        ValueError
            If the value's dtype or shape does not match expectations.

        """
        # Mandatory validation
        self._validate_array_type(value)
        self._validate_dtype(value, self.dtype)
        self._validate_shape(value, expected_shape=self.shape)

        # Optional validation
        config = self._validation_config_as_enum(config)

        if config != ConfigEnum.NULL:
            pass  # implement optional validation if needed (e.g., value ranges)
