"""Set of common action classes for OpenPMD Beamphysics variables."""

import numpy as np
from pydantic import field_validator

from lume.variables import NDVariable
from lume.actions import ReadOnlyActionMixin


class PMDVariable(NDVariable, ReadOnlyActionMixin):
    """Base class for all OpenPMD Beamphysics variables."""

    # pmd: data is stored/transmitted as float32 by OpenPMD Beamphysics convention
    dtype: np.dtype = np.dtype(np.float32)

    @field_validator("name")
    def validate_name(cls, value):
        if not value.startswith("pmd"):
            raise ValueError("Name must start with 'pmd'")
        return value

    @field_validator("shape")
    def validate_shape(cls, value):
        # make sure that the shape is 1D
        if len(value) != 1:
            raise ValueError("Shape must be 1D")
        return value


# Registry of canonical pmd: variable name -> generated PMDVariable subclass,
# used to look up the expected unit for a given pmd: variable name.
PMD_VARIABLE_REGISTRY: dict[str, type[PMDVariable]] = {}


def create_pmd_variable(
    class_name: str, name: str, unit: str, description: str
) -> type[PMDVariable]:
    """Create a `PMDVariable` subclass with a fixed name and unit.

    Parameters
    ----------
    class_name : str
        Name of the generated class.
    name : str
        Fixed value of the ``name`` field (must start with ``"pmd"``).
    unit : str
        Fixed value of the ``unit`` field.
    description : str
        Docstring describing the variable.
    """
    cls = type(
        class_name,
        (PMDVariable,),
        {
            "__annotations__": {"name": str, "unit": str, "read_only": bool},
            "name": name,
            "unit": unit,
            "read_only": True,
            "__doc__": description,
        },
    )
    PMD_VARIABLE_REGISTRY[name] = cls
    return cls


def get_pmd_variable_class(name: str) -> type[PMDVariable] | None:
    """Look up the canonical `PMDVariable` subclass for a pmd: variable name.

    Parameters
    ----------
    name : str
        The pmd: variable name to look up.

    Returns
    -------
    type[PMDVariable] | None
        The registered subclass, or `None` if `name` isn't a known pmd: variable.
    """
    return PMD_VARIABLE_REGISTRY.get(name)


PMDbeta_x = create_pmd_variable(
    "PMDbeta_x",
    "pmd:beta_x",
    "m",
    "Read-only variable for the horizontal beta function in OpenPMD Beamphysics units.",
)

PMDDbeta_y = create_pmd_variable(
    "PMDbeta_y",
    "pmd:beta_y",
    "m",
    "Read-only variable for the vertical beta function in OpenPMD Beamphysics units.",
)

PMDalpha_x = create_pmd_variable(
    "PMDalpha_x",
    "pmd:alpha_x",
    "",
    "Read-only variable for the horizontal alpha function in OpenPMD Beamphysics units.",
)

PMDalpha_y = create_pmd_variable(
    "PMDalpha_y",
    "pmd:alpha_y",
    "",
    "Read-only variable for the vertical alpha function in OpenPMD Beamphysics units.",
)

PMDnorm_emit_x = create_pmd_variable(
    "PMDnorm_emit_x",
    "pmd:norm_emit_x",
    "mm.mrad",
    "Read-only variable for the horizontal normalized emittance in OpenPMD Beamphysics units.",
)

PMDnorm_emit_y = create_pmd_variable(
    "PMDnorm_emit_y",
    "pmd:norm_emit_y",
    "mm.mrad",
    "Read-only variable for the vertical normalized emittance in OpenPMD Beamphysics units.",
)

PMDsigma_x = create_pmd_variable(
    "PMDsigma_x",
    "pmd:sigma_x",
    "m",
    "Read-only variable for the horizontal beam size in OpenPMD Beamphysics units.",
)

PMDsigma_y = create_pmd_variable(
    "PMDsigma_y",
    "pmd:sigma_y",
    "m",
    "Read-only variable for the vertical beam size in OpenPMD Beamphysics units.",
)

PMDsigma_z = create_pmd_variable(
    "PMDsigma_z",
    "pmd:sigma_z",
    "m",
    "Read-only variable for the longitudinal beam size in OpenPMD Beamphysics units.",
)

PMDkinetic_energy = create_pmd_variable(
    "PMDkinetic_energy",
    "pmd:kinetic_energy",
    "eV",
    "Read-only variable for the kinetic energy in OpenPMD Beamphysics units.",
)

PMDp = create_pmd_variable(
    "PMDp",
    "pmd:p",
    "eV/c",
    "Read-only variable for the momentum in OpenPMD Beamphysics units.",
)

PMDs = create_pmd_variable(
    "PMDs",
    "pmd:s",
    "m",
    "Read-only variable for the longitudinal beam position s.",
)
