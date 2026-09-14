"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ConfigEnum(str, Enum):
    """Enum for configuration options during validation."""

    NULL = "none"
    WARN = "warn"
    ERROR = "error"


class Variable(BaseModel, ABC):
    """Abstract variable base class.

    Attributes
    ----------
    name : str
        Name of the variable.
    description : str
        Description of the variable.
    read_only : bool
        Flag indicating whether the variable can be set.
    default_validation_config : ConfigEnum
        Default validation configuration to use when validating values.
        Valid options are "none" (no validation), "warn" (warn on invalid values),
        or "error" (raise error on invalid values). Defaults to "none".

    """

    # store/serialize as string
    model_config = ConfigDict(use_enum_values=True)

    name: str
    description: str = ""
    read_only: bool = False
    default_validation_config: ConfigEnum = "none"

    def _validation_config_as_enum(self, config: ConfigEnum = None) -> ConfigEnum:
        """Convert validation config to enum type.

        Parameters
        ----------
        config : ConfigEnum, optional
            The configuration for validation. If None, uses default_validation_config.

        Returns
        -------
        ConfigEnum
            The config as a ConfigEnum instance.
        """
        if config is None:
            config = self.default_validation_config
        if isinstance(config, str):
            config = ConfigEnum(config)
        return config

    @abstractmethod
    def validate_value(self, value: Any, config: ConfigEnum = None):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        return {"variable_class": self.__class__.__name__} | config
