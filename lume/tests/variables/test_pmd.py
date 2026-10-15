"""Tests for PMDVariable and the pmd: variable factory."""

from typing import Any

import numpy as np
import pytest

from lume.variables.pmd import (
    PMDVariable,
    PMDbeta_x,
    PMDmodel_index,
    create_pmd_variable,
)


class _ConcretePMDVariable(PMDVariable):
    """Concrete PMDVariable subclass for direct instantiation in tests."""

    def _get(self, simulator: Any) -> Any:
        raise NotImplementedError


class TestPMDVariable:
    """Tests for the PMDVariable base class."""

    def test_dtype_defaults_to_float32(self):
        """PMDVariable defaults to float32 per OpenPMD Beamphysics convention."""
        var = _ConcretePMDVariable(name="pmd:x", shape=(1,), read_only=True)
        assert var.dtype == np.dtype(np.float32)

    def test_name_must_start_with_pmd(self):
        """Names not starting with 'pmd:' are rejected."""
        with pytest.raises(ValueError, match="Name must start with 'pmd:'"):
            _ConcretePMDVariable(name="beta_x", shape=(1,), read_only=True)

    def test_shape_must_be_1d(self):
        """Non-1D shapes are rejected."""
        with pytest.raises(ValueError, match="Shape must be 1D"):
            _ConcretePMDVariable(name="pmd:x", shape=(3, 2), read_only=True)

    def test_shape_1d_is_accepted(self):
        """1D shapes are accepted."""
        var = _ConcretePMDVariable(name="pmd:x", shape=(3,), read_only=True)
        assert var.shape == (3,)


class TestCreatePMDVariable:
    """Tests for create_pmd_variable."""

    def test_creates_subclass_with_fixed_name_and_unit(self):
        assert PMDbeta_x.model_fields["name"].default == "pmd:beta_x"
        assert PMDbeta_x.model_fields["unit"].default == "m"
        assert issubclass(PMDbeta_x, PMDVariable)

    def test_name_cannot_be_overridden(self):
        """The generated class's name is fixed, not just a default."""

        class _ConcretePMDbeta_x(PMDbeta_x):
            def _get(self, simulator: Any) -> Any:
                raise NotImplementedError

        with pytest.raises(ValueError):
            _ConcretePMDbeta_x(name="pmd:something_else", shape=(1,), read_only=True)

    def test_unit_cannot_be_overridden(self):
        """The generated class's unit is fixed, not just a default."""

        class _ConcretePMDbeta_x(PMDbeta_x):
            def _get(self, simulator: Any) -> Any:
                raise NotImplementedError

        with pytest.raises(ValueError):
            _ConcretePMDbeta_x(unit="mm", shape=(1,), read_only=True)

    def test_read_only_cannot_be_overridden(self):
        """The generated class's read_only is fixed to True."""

        class _ConcretePMDbeta_x(PMDbeta_x):
            def _get(self, simulator: Any) -> Any:
                raise NotImplementedError

        with pytest.raises(ValueError):
            _ConcretePMDbeta_x(shape=(1,), read_only=False)

    def test_generated_class_rejects_non_1d_shape(self):
        """Generated subclasses inherit the 1D shape validator."""
        generated = create_pmd_variable(
            "PMDtest_bad_shape", "pmd:test_bad_shape", "m", "test"
        )

        class _ConcreteGenerated(generated):
            def _get(self, simulator: Any) -> Any:
                raise NotImplementedError

        with pytest.raises(ValueError, match="Shape must be 1D"):
            _ConcreteGenerated(shape=(2, 2), read_only=True)


class TestPMDmodel_index:
    """Tests for the PMDmodel_index class."""

    def test_name_and_unit_are_fixed(self):
        assert PMDmodel_index.model_fields["name"].default == "pmd:model_index"
        assert PMDmodel_index.model_fields["unit"].default == ""

    def test_dtype_is_int64(self):
        var = PMDmodel_index(
            shape=(3,), default_value=np.array([0, 0, 1], dtype=np.int64)
        )
        assert var.dtype == np.dtype(np.int64)

    def test_is_read_only_pmd_variable(self):
        var = PMDmodel_index(shape=(3,))
        assert isinstance(var, PMDVariable)
        assert var.read_only is True
