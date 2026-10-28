"""Tests for PMDVariable and PMDmodel_index."""

from typing import Any

import numpy as np
import pytest

from lume.variables.pmd import (
    PMDVariable,
    PMDmodel_index,
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

    def test_custom_name_is_accepted(self):
        """Custom variable names do not require the default 'pmd:' prefix."""
        var = _ConcretePMDVariable(name="beta_x", shape=(1,), read_only=True)
        assert var.name == "beta_x"

    def test_shape_must_be_1d(self):
        """Non-1D shapes are rejected."""
        with pytest.raises(ValueError, match="Shape must be 1D"):
            _ConcretePMDVariable(name="pmd:x", shape=(3, 2), read_only=True)

    def test_shape_1d_is_accepted(self):
        """1D shapes are accepted."""
        var = _ConcretePMDVariable(name="pmd:x", shape=(3,), read_only=True)
        assert var.shape == (3,)


class TestPMDmodel_index:
    """Tests for the PMDmodel_index class."""

    def test_name_and_unit_defaults(self):
        assert PMDmodel_index.model_fields["name"].default == "pmd:model_index"
        assert PMDmodel_index.model_fields["unit"].default == ""

    def test_custom_name_preserves_index_metadata(self):
        values = np.array([0, 0, 1], dtype=np.int64)
        var = PMDmodel_index(name="stage", shape=(3,), default_value=values)
        assert var.name == "stage"
        assert var.unit == ""
        assert var.dtype == np.dtype(np.int64)
        assert var.read_only is True
        var.validate_value(values)

    def test_dtype_is_int64(self):
        var = PMDmodel_index(
            shape=(3,), default_value=np.array([0, 0, 1], dtype=np.int64)
        )
        assert var.dtype == np.dtype(np.int64)

    def test_is_read_only_pmd_variable(self):
        var = PMDmodel_index(shape=(3,))
        assert isinstance(var, PMDVariable)
        assert var.read_only is True
