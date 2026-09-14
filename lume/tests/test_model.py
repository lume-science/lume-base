from typing import Any
import json
import yaml
import pytest
from lume.model import LUMEModel
from lume.variables import Variable, ScalarVariable, ConfigEnum
from lume.exceptions import ReadOnlyError


# Mock Variable subclass for testing
class MockVariable(Variable):
    """Mock Variable implementation for testing purposes."""

    def validate_value(self, value: Any, config=None):
        """Simple validation - just check if value is not None"""
        if value is None:
            raise ValueError("Value cannot be None")


# Concrete implementation for testing
class MockLUMEModel(LUMEModel):
    """Mock implementation of LUMEModel for testing abstract methods."""

    def __init__(self):
        self._variables = {
            "input_var": ScalarVariable(
                name="input_var",
                default_value=1.0,
                value_range=(0.0, 10.0),
                read_only=False,
                default_validation_config="warn",
            ),
            "output_var": ScalarVariable(
                name="output_var",
                default_value=2.0,
                value_range=(0.0, 20.0),
                read_only=True,
            ),
            "control_var": MockVariable(
                name="control_var", read_only=False, default_validation_config="warn"
            ),
        }
        self._state = {"input_var": 1.0, "output_var": 2.0, "control_var": "initial"}
        self._initial_state = self._state.copy()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        # Simulate setting values and running simulation
        for name, value in values.items():
            self._state[name] = value
        # Simulate computation that updates output_var based on input_var
        if "input_var" in values:
            self._state["output_var"] = values["input_var"] * 2.0

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class TestLUMEModel:
    """Test suite for LUMEModel abstract base class."""

    @pytest.fixture
    def model(self):
        """Fixture providing a MockLUMEModel instance."""
        return MockLUMEModel()

    def test_supported_variables_property(self, model):
        """Test that supported_variables property returns correct variables."""
        variables = model.supported_variables
        assert isinstance(variables, dict)
        assert "input_var" in variables
        assert "output_var" in variables
        assert "control_var" in variables
        assert isinstance(variables["input_var"], ScalarVariable)
        assert isinstance(variables["output_var"], ScalarVariable)
        assert isinstance(variables["control_var"], MockVariable)

    def test_get_value_valid_variable(self, model):
        """Test that get_value() returns a single variable value."""
        result = model.get_value("input_var")
        assert result == 1.0

    def test_get_value_invalid_variable(self, model):
        """Test getting an unsupported variable with get_value() raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.get_value("invalid_var")

    def test_get_value_validation_error_includes_name(self, model):
        """Test get_value() validation errors include the variable name."""
        model._state["control_var"] = None

        with pytest.raises(
            ValueError,
            match="Validation failed for variable 'control_var': Value cannot be None",
        ):
            model.get_value("control_var")

    def test_get_valid_variables(self, model):
        """Test getting valid variable values."""
        result = model.get(["input_var", "output_var"])
        assert isinstance(result, dict)
        assert result["input_var"] == 1.0
        assert result["output_var"] == 2.0

    def test_get_single_variable(self, model):
        """Test getting a single variable."""
        result = model.get(["input_var"])
        assert result == {"input_var": 1.0}

    def test_get_invalid_variable(self, model):
        """Test getting an unsupported variable raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.get(["invalid_var"])

    def test_get_mixed_valid_invalid_variables(self, model):
        """Test getting mix of valid and invalid variables."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.get(["input_var", "invalid_var"])

    def test_get_empty_list(self, model):
        """Test getting empty list of variables."""
        result = model.get([])
        assert result == {}

    def test_get_variable_validation_error_includes_name(self, model):
        """Test get() validation errors include the variable name."""
        model._state["control_var"] = None

        with pytest.raises(
            ValueError,
            match="Validation failed for variable 'control_var': Value cannot be None",
        ):
            model.get(["control_var"])

    def test_set_valid_variables(self, model):
        """Test setting valid variable values."""
        model.set({"input_var": 5.0, "control_var": "updated"})

        # Check that values were set
        result = model.get(["input_var", "control_var"])
        assert result["input_var"] == 5.0
        assert result["control_var"] == "updated"

        # Check that simulation ran (output_var should be updated)
        output_result = model.get(["output_var"])
        assert output_result["output_var"] == 10.0  # input_var * 2

    def test_set_single_variable(self, model):
        """Test setting a single variable."""
        model.set({"input_var": 3.5})
        result = model.get(["input_var"])
        assert result["input_var"] == 3.5

    def test_set_value_valid_variable(self, model):
        """Test that set_value() sets a single variable value."""
        model.set_value("input_var", 3.5)

        assert model.get_value("input_var") == 3.5
        assert model.get_value("output_var") == 7.0

    def test_set_value_invalid_variable(self, model):
        """Test setting an unsupported variable with set_value() raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.set_value("invalid_var", 123)

    def test_set_value_read_only_variable(self, model):
        """Test setting a read-only variable with set_value() raises ReadOnlyError."""
        with pytest.raises(ReadOnlyError, match="Variable 'output_var' is read-only"):
            model.set_value("output_var", 100.0)

    def test_set_value_validation_error_includes_name(self, model):
        """Test set_value() validation errors include the variable name."""
        with pytest.raises(
            ValueError,
            match="Validation failed for variable 'control_var': Value cannot be None",
        ):
            model.set_value("control_var", None)

    def test_set_invalid_variable(self, model):
        """Test setting an unsupported variable raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.set({"invalid_var": 123})

    def test_set_read_only_variable(self, model):
        """Test setting a read-only variable raises ValueError."""
        with pytest.raises(ReadOnlyError, match="Variable 'output_var' is read-only"):
            model.set({"output_var": 100.0})

    def test_set_variable_validation_error(self, model):
        """Test setting variable with invalid value raises validation error."""
        # This should trigger validation in ScalarVariable
        with pytest.raises(TypeError, match="Expected value to be of type"):
            model.set({"input_var": "not_a_float"})

    def test_set_variable_out_of_range(self, model):
        """Test setting variable with out-of-range value triggers validation."""
        # This should trigger range validation warning for ScalarVariable
        # The test captures stdout to verify warning is printed

        # Set value outside range - should print warning but not raise error
        with pytest.warns(UserWarning, match="Value .* is out of valid range"):
            model.set({"input_var": 15.0})  # Range is (0.0, 10.0)

    def test_set_variable_none_validation(self, model):
        """Test setting variable to None with custom validation."""
        with pytest.raises(ValueError, match="Value cannot be None"):
            model.set({"control_var": None})

    def test_set_variable_validation_error_includes_name(self, model):
        """Test set() validation errors include the variable name."""
        with pytest.raises(
            ValueError,
            match="Validation failed for variable 'control_var': Value cannot be None",
        ):
            model.set({"control_var": None})

    def test_set_mixed_valid_invalid_variables(self, model):
        """Test setting mix of valid and invalid variables."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.set({"input_var": 2.0, "invalid_var": 123})

    def test_set_empty_dict(self, model):
        """Test setting empty dict of variables."""
        initial_state = model.get(["input_var", "output_var", "control_var"])
        model.set({})
        final_state = model.get(["input_var", "output_var", "control_var"])
        assert initial_state == final_state

    def test_workflow_set_get_reset(self, model):
        """Test complete workflow of set, get, and reset operations."""
        # Initial state
        initial = model.get(["input_var", "output_var", "control_var"])
        assert initial == {
            "input_var": 1.0,
            "output_var": 2.0,
            "control_var": "initial",
        }

        # Set new values
        model.set({"input_var": 4.0, "control_var": "workflow_test"})

        # Get updated state
        updated = model.get(["input_var", "output_var", "control_var"])
        assert updated == {
            "input_var": 4.0,
            "output_var": 8.0,
            "control_var": "workflow_test",
        }

        # Reset
        model.reset()

        # Verify reset
        reset_state = model.get(["input_var", "output_var", "control_var"])
        assert reset_state == initial

    def test_variable_validation_called_during_set(self, model):
        """Test that variable validation is properly called during set operations."""
        # Test with ScalarVariable validation
        with pytest.raises(TypeError):
            model.set({"input_var": [1, 2, 3]})  # Invalid type

        # Test with custom MockVariable validation
        with pytest.raises(ValueError, match="Value cannot be None"):
            model.set({"control_var": None})

    def test_supported_variables_with_non_variable_value(self):
        """Test behavior when supported_variables contains non-Variable values."""

        class InvalidVariableModel(LUMEModel):
            """Model with non-Variable in supported_variables for testing."""

            @property
            def supported_variables(self):
                return {
                    "valid_var": ScalarVariable(
                        name="valid_var", default_value=1.0, read_only=False
                    ),
                    "invalid_var": "not_a_variable_object",  # This is not a Variable!
                }

            def _get(self, names):
                return {"valid_var": 1.0, "invalid_var": "some_value"}

            def _set(self, values):
                pass

            def reset(self):
                pass

        model = InvalidVariableModel()

        # Test that the invalid entry exists in supported_variables
        variables = model.supported_variables
        assert "invalid_var" in variables
        assert not isinstance(variables["invalid_var"], Variable)

        # Test that set() fails when trying to validate the non-Variable
        with pytest.raises(ValueError, match="is not a valid Variable instance"):
            # This should fail because strings don't have read_only or validate_value attributes
            model.set({"invalid_var": "some_value"})


class TestVariableModelDump:
    """Test that model_dump properly serializes ConfigEnum values."""

    def test_scalar_variable_model_dump_serializes_config_enum(self):
        """Test that ScalarVariable.model_dump() converts ConfigEnum to string."""
        # Test with each ConfigEnum value
        for config_value in [ConfigEnum.NULL, ConfigEnum.WARN, ConfigEnum.ERROR]:
            var = ScalarVariable(
                name="test_var",
                default_value=1.0,
                value_range=(0.0, 10.0),
                default_validation_config=config_value,
            )

            dumped = var.model_dump()

            # Check that variable_class is included
            assert "variable_class" in dumped
            assert dumped["variable_class"] == "ScalarVariable"

            # Check that default_validation_config is a string, not an enum
            assert "default_validation_config" in dumped
            assert isinstance(dumped["default_validation_config"], str)
            assert not isinstance(dumped["default_validation_config"], ConfigEnum)

            # Check that the value matches the enum's string value
            assert dumped["default_validation_config"] == config_value.value

    def test_scalar_variable_model_dump_json_serializable(self):
        """Test that model_dump output can be serialized to JSON."""
        var = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
            default_validation_config=ConfigEnum.WARN,
            unit="meters",
        )

        dumped = var.model_dump()

        # Should be able to serialize to JSON without errors
        json_str = json.dumps(dumped)
        assert json_str is not None

        # Should be able to parse it back
        parsed = json.loads(json_str)
        assert parsed["name"] == "test_var"
        assert parsed["default_value"] == 5.0
        assert parsed["default_validation_config"] == "warn"
        assert parsed["unit"] == "meters"

    def test_scalar_variable_model_dump_yaml_serializable(self):
        """Test that model_dump output can be serialized to YAML."""
        var = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
            default_validation_config=ConfigEnum.ERROR,
            unit="seconds",
        )

        dumped = var.model_dump()

        # Should be able to serialize to YAML without errors
        yaml_str = yaml.safe_dump(dumped)
        assert yaml_str is not None

        # Should be able to parse it back
        parsed = yaml.safe_load(yaml_str)
        assert parsed["name"] == "test_var"
        assert parsed["default_value"] == 5.0
        assert parsed["default_validation_config"] == "error"
        assert parsed["unit"] == "seconds"

    def test_scalar_variable_model_dump_with_none_config(self):
        """Test model_dump when default_validation_config defaults to NULL."""
        var = ScalarVariable(
            name="test_var",
            default_value=1.0,
            value_range=(0.0, 10.0),
        )

        dumped = var.model_dump()

        # Should have the default NULL config as string
        assert dumped["default_validation_config"] == "none"
        assert isinstance(dumped["default_validation_config"], str)

        # Should be JSON serializable
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        assert parsed["default_validation_config"] == "none"

    def test_scalar_variable_list_serialization(self):
        """Test that a list of variables can be serialized (common use case)."""
        variables = [
            ScalarVariable(
                name="var1",
                default_value=1.0,
                value_range=(0.0, 10.0),
                default_validation_config=ConfigEnum.WARN,
            ),
            ScalarVariable(
                name="var2",
                default_value=2.0,
                value_range=(0.0, 20.0),
                default_validation_config=ConfigEnum.ERROR,
            ),
            ScalarVariable(
                name="var3",
                default_value=3.0,
                value_range=(0.0, 30.0),
                default_validation_config=ConfigEnum.NULL,
            ),
        ]

        # Serialize all variables
        dumped_vars = [var.model_dump() for var in variables]

        # Should be JSON serializable
        json_str = json.dumps(dumped_vars)
        parsed = json.loads(json_str)

        assert len(parsed) == 3
        assert parsed[0]["default_validation_config"] == "warn"
        assert parsed[1]["default_validation_config"] == "error"
        assert parsed[2]["default_validation_config"] == "none"

        # Should be YAML serializable
        yaml_str = yaml.safe_dump(dumped_vars)
        parsed_yaml = yaml.safe_load(yaml_str)

        assert len(parsed_yaml) == 3
        assert parsed_yaml[0]["default_validation_config"] == "warn"
        assert parsed_yaml[1]["default_validation_config"] == "error"
        assert parsed_yaml[2]["default_validation_config"] == "none"


class TestConfigEnumValidation:
    """Test that validate_value() properly handles ConfigEnum and string parameters."""

    def test_validate_value_accepts_enum_parameter(self):
        """Test that validate_value() accepts ConfigEnum as config parameter."""
        var = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
        )

        # Should not raise with value in range
        var.validate_value(7.0, config=ConfigEnum.ERROR)

        # Should raise with value out of range and ERROR config
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0, config=ConfigEnum.ERROR)

        # Should warn with value out of range and WARN config
        with pytest.warns(UserWarning, match="out of valid range"):
            var.validate_value(15.0, config=ConfigEnum.WARN)

        # Should not raise/warn with NULL config even if out of range
        var.validate_value(15.0, config=ConfigEnum.NULL)  # No exception

    def test_validate_value_accepts_string_parameter(self):
        """Test that validate_value() accepts string as config parameter."""
        var = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
        )

        # Should not raise with value in range
        var.validate_value(7.0, config="error")

        # Should raise with value out of range and "error" config
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0, config="error")

        # Should warn with value out of range and "warn" config
        with pytest.warns(UserWarning, match="out of valid range"):
            var.validate_value(15.0, config="warn")

        # Should not raise/warn with "none" config even if out of range
        var.validate_value(15.0, config="none")  # No exception

    def test_validate_value_uses_default_config_when_none(self):
        """Test that validate_value() uses default_validation_config when config=None."""
        # Variable with WARN default
        var_warn = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
            default_validation_config="warn",
        )

        # Should warn when config=None (uses default "warn")
        with pytest.warns(UserWarning, match="out of valid range"):
            var_warn.validate_value(15.0, config=None)

        # Variable with ERROR default
        var_error = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
            default_validation_config=ConfigEnum.ERROR,
        )

        # Should raise when config=None (uses default "error")
        with pytest.raises(ValueError, match="out of valid range"):
            var_error.validate_value(15.0, config=None)

        # Variable with NULL default
        var_null = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
            default_validation_config="none",
        )

        # Should not raise/warn when config=None (uses default "none")
        var_null.validate_value(15.0, config=None)  # No exception

    def test_validate_value_explicit_config_overrides_default(self):
        """Test that explicit config parameter overrides default_validation_config."""
        var = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
            default_validation_config="error",  # Default is ERROR
        )

        # Override with NULL - should not raise even though default is ERROR
        var.validate_value(15.0, config="none")  # No exception

        # Override with WARN - should warn instead of raising
        with pytest.warns(UserWarning, match="out of valid range"):
            var.validate_value(15.0, config=ConfigEnum.WARN)

    def test_invalid_config_string_raises_error(self):
        """Test that invalid config string raises ValueError."""
        var = ScalarVariable(
            name="test_var",
            default_value=5.0,
            value_range=(0.0, 10.0),
        )

        # Should raise ValueError for invalid config string
        with pytest.raises(ValueError, match="'invalid' is not a valid ConfigEnum"):
            var.validate_value(7.0, config="invalid")
