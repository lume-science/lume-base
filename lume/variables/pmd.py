""" Set of common action classes for OpenPMD Beamphysics variables. """

from lume.variables import NDVariable
from lume.actions import ReadOnlyActionMixin

class PMDbeta_x(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the horizontal beta function in OpenPMD Beamphysics units."""

    unit: str = "m"

class PMDDbeta_y(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the vertical beta function in OpenPMD Beamphysics units."""

    unit: str = "m"

class PMDDalpha_x(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the horizontal alpha function in OpenPMD Beamphysics units."""

    unit: str = ""

class PMDDalpha_y(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the vertical alpha function in OpenPMD Beamphysics units."""

    unit: str = ""

class PMDnorm_emit_x(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the horizontal normalized emittance in OpenPMD Beamphysics units."""

    unit: str = "mm.mrad"

class PMDnorm_emit_y(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the vertical normalized emittance in OpenPMD Beamphysics units."""

    unit: str = "mm.mrad"

class PMDsigma_x(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the horizontal beam size in OpenPMD Beamphysics units."""

    unit: str = "m"

class PMDsigma_y(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the vertical beam size in OpenPMD Beamphysics units."""

    unit: str = "m"

class PMDsigma_z(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the longitudinal beam size in OpenPMD Beamphysics units."""

    unit: str = "m"

class PMDkinetic_energy(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the kinetic energy in OpenPMD Beamphysics units."""

    unit: str = "eV"

class PMDp(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the momentum in OpenPMD Beamphysics units."""

    unit: str = "eV/c"

class PMDs(NDVariable, ReadOnlyActionMixin):
    """Read-only variable for the longitudinal beam position s."""
    
    unit: str = "m"