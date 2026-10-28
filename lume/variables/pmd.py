"""Set of common action base classes for OpenPMD Beamphysics variables."""

from typing import Any, Literal

import numpy as np
from pydantic import field_validator

from lume.variables import NDVariable
from lume.actions import ReadOnlyActionMixin


class PMDVariable(NDVariable, ReadOnlyActionMixin):
    """Base class for all OpenPMD Beamphysics variables."""

    # pmd: data is stored/transmitted as float32 by OpenPMD Beamphysics convention
    dtype: np.dtype = np.dtype(np.float32)
    read_only: bool = True

    @field_validator("shape")
    def validate_shape(cls, value):
        # make sure that the shape is 1D
        if len(value) != 1:
            raise ValueError("Shape must be 1D")
        return value


class PMDbeta_x(PMDVariable):
    """Read-only variable for the horizontal beta function in OpenPMD Beamphysics units."""

    name: str = "pmd:beta_x"  # default name
    unit: Literal["m"] = "m"


class PMDbeta_y(PMDVariable):
    """Read-only variable for the vertical beta function in OpenPMD Beamphysics units."""

    name: str = "pmd:beta_y"  # default name
    unit: Literal["m"] = "m"


class PMDalpha_x(PMDVariable):
    """Read-only variable for the horizontal alpha function in OpenPMD Beamphysics units."""

    name: str = "pmd:alpha_x"  # default name
    unit: Literal[""] = ""


class PMDalpha_y(PMDVariable):
    """Read-only variable for the vertical alpha function in OpenPMD Beamphysics units."""

    name: str = "pmd:alpha_y"  # default name
    unit: Literal[""] = ""


class PMDgamma_x(PMDVariable):
    """Read-only variable for the horizontal gamma function in OpenPMD Beamphysics units."""

    name: str = "pmd:gamma_x"  # default name
    unit: Literal[""] = ""


class PMDgamma_y(PMDVariable):
    """Read-only variable for the vertical gamma function in OpenPMD Beamphysics units."""

    name: str = "pmd:gamma_y"  # default name
    unit: Literal[""] = ""


class PMDnorm_emit_x(PMDVariable):
    """Read-only variable for the horizontal normalized emittance in OpenPMD Beamphysics units."""

    name: str = "pmd:norm_emit_x"  # default name
    unit: Literal["m"] = "m"


class PMDnorm_emit_y(PMDVariable):
    """Read-only variable for the vertical normalized emittance in OpenPMD Beamphysics units."""

    name: str = "pmd:norm_emit_y"  # default name
    unit: Literal["m"] = "m"


class PMDsigma_x(PMDVariable):
    """Read-only variable for the horizontal beam size in OpenPMD Beamphysics units."""

    name: str = "pmd:sigma_x"  # default name
    unit: Literal["m"] = "m"


class PMDsigma_y(PMDVariable):
    """Read-only variable for the vertical beam size in OpenPMD Beamphysics units."""

    name: str = "pmd:sigma_y"  # default name
    unit: Literal["m"] = "m"


class PMDsigma_z(PMDVariable):
    """Read-only variable for the longitudinal beam size in OpenPMD Beamphysics units."""

    name: str = "pmd:sigma_z"  # default name
    unit: Literal["m"] = "m"


class PMDkinetic_energy(PMDVariable):
    """Read-only variable for the kinetic energy in OpenPMD Beamphysics units."""

    name: str = "pmd:kinetic_energy"  # default name
    unit: Literal["eV"] = "eV"


class PMDp(PMDVariable):
    """Read-only variable for the momentum in OpenPMD Beamphysics units."""

    name: str = "pmd:p"  # default name
    unit: Literal["eV/c"] = "eV/c"


class PMDs(PMDVariable):
    """Read-only variable for the longitudinal beam position s in OpenPMD Beamphysics units."""

    name: str = "pmd:s"  # default name
    unit: Literal["m"] = "m"


class PMDmodel_index(PMDVariable):
    """Identifies, for each entry of a combined pmd: array, which staged model produced it."""

    name: str = "pmd:model_index"
    unit: Literal[""] = ""
    read_only: Literal[True] = True
    dtype: np.dtype = np.dtype(np.int64)

    def _get(self, simulator: Any) -> Any:
        raise NotImplementedError(
            "pmd:model_index is computed directly by StagedModel, not fetched "
            "from a simulator."
        )
