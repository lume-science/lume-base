"""Tests for PMDVariable and the pmd: variable factory/registry."""

from typing import Any

import numpy as np
import pytest

from lume.variables.pmd import (
    PMD_VARIABLE_REGISTRY,
    PMDVariable,
    PMDbeta_x,
    create_pmd_variable,
    get_pmd_variable_class,
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
        """Names not starting with 'pmd' are rejected."""
        with pytest.raises(ValueError, match="Name must start with 'pmd'"):
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
    """Tests for create_pmd_variable and the PMD_VARIABLE_REGISTRY."""

    def test_creates_subclass_with_fixed_name_and_unit(self):
        assert PMDbeta_x.model_fields["name"].default == "pmd:beta_x"
        assert PMDbeta_x.model_fields["unit"].default == "m"
        assert issubclass(PMDbeta_x, PMDVariable)

    def test_registers_class_by_name(self):
        assert PMD_VARIABLE_REGISTRY["pmd:beta_x"] is PMDbeta_x
        assert get_pmd_variable_class("pmd:beta_x") is PMDbeta_x

    def test_unknown_name_returns_none(self):
        assert get_pmd_variable_class("pmd:not_a_real_variable") is None

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
