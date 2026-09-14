"""Tests for Variable base class."""

import pytest

from lume.variables.variable import ConfigEnum, Variable


class ConcreteVariable(Variable):
    """Concrete implementation of Variable for testing purposes."""

    def validate_value(self, value, config=None):
        """Simple validation that accepts only integers."""
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value)}")


class TestConfigEnum:
    """Test ConfigEnum functionality."""

    def test_config_enum_values(self):
        """Test ConfigEnum has expected values and string conversion."""
        assert ConfigEnum.NULL == "none"
        assert ConfigEnum.WARN == "warn"
        assert ConfigEnum.ERROR == "error"
        assert ConfigEnum("none") == ConfigEnum.NULL
        assert ConfigEnum("warn") == ConfigEnum.WARN

    def test_invalid_config_enum(self):
        """Test that invalid config values raise error."""
        with pytest.raises(ValueError):
            ConfigEnum("invalid")


class TestVariable:
    """Test Variable base class functionality."""

    def test_creation_and_defaults(self):
        """Test creating a variable with defaults and custom values."""
        var = ConcreteVariable(name="test_var")
        assert var.name == "test_var"
        assert var.read_only is False
        assert var.default_validation_config == "none"

        var2 = ConcreteVariable(
            name="custom", read_only=True, default_validation_config="error"
        )
        assert var2.read_only is True
        assert var2.default_validation_config == "error"

    def test_validation_config_as_enum(self):
        """Test _validation_config_as_enum method."""
        var = ConcreteVariable(name="test", default_validation_config="warn")
        # None uses default
        assert var._validation_config_as_enum(None) == ConfigEnum.WARN
        # String converted
        assert var._validation_config_as_enum("error") == ConfigEnum.ERROR
        # Enum preserved
        assert var._validation_config_as_enum(ConfigEnum.NULL) == ConfigEnum.NULL

    def test_model_dump(self):
        """Test that model_dump includes variable_class and all fields."""
        var = ConcreteVariable(
            name="test_var", read_only=True, default_validation_config="error"
        )
        dumped = var.model_dump()
        assert dumped["variable_class"] == "ConcreteVariable"
        assert dumped["name"] == "test_var"
        assert dumped["read_only"] is True
        assert dumped["default_validation_config"] == "error"
        assert (
            ConcreteVariable(**dumped).model_dump()["variable_class"]
            == "ConcreteVariable"
        )

    def test_validate_value_implementation(self):
        """Test that concrete implementation works."""
        var = ConcreteVariable(name="test")
        var.validate_value(5)  # Should not raise
        with pytest.raises(TypeError):
            var.validate_value("not an int")

    def test_cannot_instantiate_abstract(self):
        """Test that Variable cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Variable(name="test")
