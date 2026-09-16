from typing import Any

import numpy as np
import pytest

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn, InitialParticlesMixIn, StagedModel
from lume.variables import ScalarVariable, Variable
from lume.variables.pmd import PMDVariable, PMDbeta_x


def make_test_particle_group(
    n_particles: int = 8,
    x_offset: float = 0.0,
    pz: float = 1.0e7,
) -> ParticleGroup:
    x = np.linspace(-1.0e-3, 1.0e-3, n_particles) + x_offset
    zeros = np.zeros(n_particles)
    return ParticleGroup(
        data={
            "x": x,
            "px": zeros,
            "y": zeros,
            "py": zeros,
            "z": zeros,
            "pz": np.full(n_particles, pz),
            "t": zeros,
            "status": np.ones(n_particles),
            "weight": np.full(n_particles, 1.0e-12),
            "species": "electron",
        }
    )


class BeamSourceTestModel(LUMEModel, InitialParticlesMixIn, FinalParticlesMixIn):
    """Source model: forwards initial_particles to final_particles on set."""

    def __init__(
        self,
        pmd_variables: dict[str, Variable] | None = None,
        pmd_values: dict[str, Any] | None = None,
    ):
        self.call_count_set = 0
        self._variables = {
            "source_phase": ScalarVariable(
                name="source_phase", default_value=0.0, read_only=False
            ),
        }
        self._variables.update(pmd_variables or {})
        self._state = {"source_phase": 0.0}
        self._state.update(pmd_values or {})
        self._initial_state = self._state.copy()
        beam = make_test_particle_group(x_offset=0.0)
        self._initial_particles = beam
        self._final_particles = beam

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def initial_particles(self) -> ParticleGroup:
        return self._initial_particles

    @initial_particles.setter
    def initial_particles(self, val: ParticleGroup) -> None:
        self._initial_particles = val

    @property
    def final_particles(self) -> ParticleGroup:
        return self._final_particles

    def set(self, values: dict[str, Any]) -> None:
        self.call_count_set += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "source_phase" in values:
            self._state["source_phase"] = values["source_phase"]
        self._final_particles = self._initial_particles

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class BeamTransportTestModel(LUMEModel, InitialParticlesMixIn, FinalParticlesMixIn):
    """Transport model: forwards initial_particles to final_particles on set."""

    def __init__(
        self,
        pmd_variables: dict[str, Variable] | None = None,
        pmd_values: dict[str, Any] | None = None,
    ):
        self.call_count_set = 0
        self._variables = {
            "transport_scale": ScalarVariable(
                name="transport_scale", default_value=1.0, read_only=False
            ),
        }
        self._variables.update(pmd_variables or {})
        self._state = {"transport_scale": 1.0}
        self._state.update(pmd_values or {})
        self._initial_state = self._state.copy()
        beam = make_test_particle_group(x_offset=2.0e-4)
        self._initial_particles = beam
        self._final_particles = beam

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def initial_particles(self) -> ParticleGroup:
        return self._initial_particles

    @initial_particles.setter
    def initial_particles(self, val: ParticleGroup) -> None:
        self._initial_particles = val

    @property
    def final_particles(self) -> ParticleGroup:
        return self._final_particles

    def set(self, values: dict[str, Any]) -> None:
        self.call_count_set += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "transport_scale" in values:
            self._state["transport_scale"] = values["transport_scale"]
        self._final_particles = self._initial_particles

    def reset(self) -> None:
        self._state = self._initial_state.copy()


def test_staged_model_init() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])
    assert model._is_init is False


def test_staged_correct_initialization() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])
    assert model._is_init is False

    # test function explicitly triggers model evaluation
    model.coerce_evaluated_once()
    assert model._is_init is True

    model.reset()
    assert model._is_init is False

    # test get triggers evaluation if not already initialized
    values = model.get(["source_phase", "transport_scale"])
    assert "source_phase" in values
    assert "transport_scale" in values
    assert model._is_init is True

    model.reset()
    assert model._is_init is False

    # test set triggers evaluation if not already initialized
    model.set({"source_phase": 3.0})
    assert model._is_init is True


def test_staged_model_supported_variables_union() -> None:
    model = StagedModel([BeamSourceTestModel(), BeamTransportTestModel()])
    assert set(model.supported_variables.keys()) == {"source_phase", "transport_scale"}


def test_staged_model_initial_and_final_particles_properties() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])

    new_beam = make_test_particle_group(x_offset=9.0e-4)
    model.initial_particles = new_beam

    assert np.allclose(model.initial_particles.x, new_beam.x)
    assert np.allclose(beam_source.initial_particles.x, new_beam.x)

    model.set({"source_phase": 2.0, "transport_scale": 3.0})

    assert np.allclose(model.final_particles.x, new_beam.x)
    assert np.allclose(beam_transport.final_particles.x, new_beam.x)


def test_staged_model_propagates_beam_to_next_stage() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])
    model.coerce_evaluated_once()

    source_calls_before = beam_source.call_count_set
    transport_calls_before = beam_transport.call_count_set

    new_beam = make_test_particle_group(x_offset=7.5e-4)
    beam_source.initial_particles = new_beam
    model.set({"source_phase": 5.0})

    assert np.allclose(beam_source.final_particles.x, new_beam.x)
    assert np.allclose(beam_transport.initial_particles.x, new_beam.x)
    assert beam_source.call_count_set == source_calls_before + 1
    assert beam_transport.call_count_set == transport_calls_before


def test_staged_model_only_updates_later_stage_when_requested() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])
    model.coerce_evaluated_once()

    source_calls_before = beam_source.call_count_set
    transport_calls_before = beam_transport.call_count_set

    model.set({"transport_scale": 2.5})

    assert beam_transport.get_value("transport_scale") == 2.5
    assert beam_source.call_count_set == source_calls_before
    assert beam_transport.call_count_set == transport_calls_before + 1


def test_staged_model_beam_always_propagates() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])

    new_beam = make_test_particle_group(x_offset=7.5e-4)
    beam_source.initial_particles = new_beam
    model.set({"source_phase": 5.0})

    # Even when only transport_scale changes, source final_particles propagate to transport
    model.set({"transport_scale": 2.5})
    assert np.allclose(beam_transport.initial_particles.x, new_beam.x)


class NoParticlesModel(LUMEModel):
    @property
    def supported_variables(self):
        return {"x": ScalarVariable(name="x", default_value=0.0, read_only=False)}

    def _get(self, names):
        return {name: 0.0 for name in names}

    def _set(self, values):
        pass

    def reset(self):
        pass


def test_staged_model_requires_final_particles_for_non_last_stage() -> None:
    with pytest.raises(ValueError, match="FinalParticlesMixIn"):
        StagedModel([NoParticlesModel(), BeamTransportTestModel()])


def test_staged_model_requires_initial_particles_after_first_stage() -> None:
    with pytest.raises(ValueError, match="InitialParticlesMixIn"):
        StagedModel([BeamSourceTestModel(), NoParticlesModel()])


def test_staged_model_rejects_conflicting_variable_names() -> None:
    model_a = BeamSourceTestModel()
    model_b = BeamSourceTestModel()  # both have "source_phase"

    with pytest.raises(ValueError, match="source_phase"):
        StagedModel([model_a, model_b])


def test_staged_model_raises_when_first_model_lacks_initial_particles() -> None:
    class NoInitialParticlesModel(LUMEModel):
        @property
        def supported_variables(self):
            return {
                "x": ScalarVariable(name="x", default_value=0.0, read_only=False),
            }

        def _get(self, names):
            return {name: 0.0 for name in names}

        def _set(self, values):
            pass

        def reset(self):
            pass

    model = StagedModel([NoInitialParticlesModel()])

    with pytest.raises(
        AttributeError,
        match="Cannot access initial_particles because the first model does not implement InitialParticlesMixIn",
    ):
        _ = model.initial_particles

    with pytest.raises(
        AttributeError,
        match="Cannot set initial_particles because the first model does not implement InitialParticlesMixIn",
    ):
        model.initial_particles = make_test_particle_group()


def test_staged_model_raises_when_last_model_lacks_final_particles() -> None:
    class NoFinalParticlesModel(LUMEModel):
        @property
        def supported_variables(self):
            return {
                "x": ScalarVariable(name="x", default_value=0.0, read_only=False),
            }

        def _get(self, names):
            return {name: 0.0 for name in names}

        def _set(self, values):
            pass

        def reset(self):
            pass

    model = StagedModel([NoFinalParticlesModel()])

    with pytest.raises(
        AttributeError,
        match="Cannot access final_particles because the last model does not implement FinalParticlesMixIn",
    ):
        _ = model.final_particles


class _TestPMDVariable(PMDVariable):
    """Concrete PMDVariable subclass for use as a stage's pmd: variable in tests."""

    def _get(self, simulator: Any) -> Any:
        raise NotImplementedError


class _OtherTestPMDVariable(PMDVariable):
    """A second, distinct PMDVariable subclass used to test subclass-mismatch validation."""

    def _get(self, simulator: Any) -> Any:
        raise NotImplementedError


def _pmd_variable(
    name: str,
    shape: tuple[int, ...],
    unit: str | None = None,
    cls: type[PMDVariable] = _TestPMDVariable,
) -> PMDVariable:
    return cls(name=name, shape=shape, unit=unit, read_only=True)


def test_staged_model_pmd_variable_registered_and_concatenated() -> None:
    beam_source = BeamSourceTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (1,))},
        pmd_values={"pmd:beta_x": np.array([1.0], dtype=np.float32)},
    )
    beam_transport = BeamTransportTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (1,))},
        pmd_values={"pmd:beta_x": np.array([2.0], dtype=np.float32)},
    )
    model = StagedModel([beam_source, beam_transport])

    combined = model.supported_variables["pmd:beta_x"]
    assert isinstance(combined, _TestPMDVariable)
    assert combined.shape == (2,)
    assert combined.read_only is True

    values = model.get(["pmd:beta_x"])["pmd:beta_x"]
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float32
    assert np.allclose(values, [1.0, 2.0])


def test_staged_model_pmd_model_index_variable() -> None:
    beam_source = BeamSourceTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (2,))},
        pmd_values={"pmd:beta_x": np.array([1.0, 2.0], dtype=np.float32)},
    )
    beam_transport = BeamTransportTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (3,))},
        pmd_values={"pmd:beta_x": np.array([3.0, 4.0, 5.0], dtype=np.float32)},
    )
    model = StagedModel([beam_source, beam_transport])

    assert "pmd:model_index" in model.supported_variables
    combined = model.supported_variables["pmd:model_index"]
    assert combined.shape == (5,)
    assert combined.dtype == np.int64
    assert combined.read_only is True

    model_index = model.get(["pmd:model_index"])["pmd:model_index"]
    assert np.array_equal(model_index, [0, 0, 1, 1, 1])


def test_staged_model_no_pmd_model_index_when_no_pmd_variables() -> None:
    model = StagedModel([BeamSourceTestModel(), BeamTransportTestModel()])
    assert "pmd:model_index" not in model.supported_variables


def test_staged_model_pmd_variable_partial_support_excluded() -> None:
    beam_source = BeamSourceTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (1,))},
        pmd_values={"pmd:beta_x": np.array([1.0])},
    )
    beam_transport = BeamTransportTestModel()  # does not support pmd:beta_x

    with pytest.warns(
        UserWarning, match=r"not supported by model 1 \(BeamTransportTestModel\)"
    ):
        model = StagedModel([beam_source, beam_transport])

    assert "pmd:beta_x" not in model.supported_variables
    with pytest.raises(ValueError, match="not supported"):
        model.get(["pmd:beta_x"])


def test_staged_model_pmd_variable_wrong_type_excluded_with_warning() -> None:
    beam_source = BeamSourceTestModel(
        pmd_variables={
            "pmd:beta_x": ScalarVariable(
                name="pmd:beta_x", default_value=1.0, read_only=True
            )
        },
        pmd_values={"pmd:beta_x": 1.0},
    )
    beam_transport = BeamTransportTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (1,))},
        pmd_values={"pmd:beta_x": np.array([2.0])},
    )

    with pytest.warns(
        UserWarning, match=r"not a PMDVariable on model 0 \(BeamSourceTestModel\)"
    ):
        model = StagedModel([beam_source, beam_transport])

    assert "pmd:beta_x" not in model.supported_variables


def test_staged_model_pmd_variable_different_subclass_excluded_with_warning() -> None:
    beam_source = BeamSourceTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (1,))},
        pmd_values={"pmd:beta_x": np.array([1.0])},
    )
    beam_transport = BeamTransportTestModel(
        pmd_variables={
            "pmd:beta_x": _pmd_variable("pmd:beta_x", (1,), cls=_OtherTestPMDVariable)
        },
        pmd_values={"pmd:beta_x": np.array([2.0])},
    )

    with pytest.warns(
        UserWarning,
        match=r"model 1 \(BeamTransportTestModel\) than model 0 \(BeamSourceTestModel\)",
    ):
        model = StagedModel([beam_source, beam_transport])

    assert "pmd:beta_x" not in model.supported_variables


class _ImpactLikePMDbeta_x(PMDbeta_x):
    """Simulates one simulator-specific concrete subclass of the canonical PMDbeta_x."""

    def _get(self, simulator: Any) -> Any:
        raise NotImplementedError


class _DistgenLikePMDbeta_x(PMDbeta_x):
    """Simulates a second, distinct simulator-specific concrete subclass of PMDbeta_x."""

    def _get(self, simulator: Any) -> Any:
        raise NotImplementedError


def test_staged_model_pmd_variable_same_canonical_class_combines() -> None:
    """Different concrete subclasses of the same canonical PMDVariable combine."""
    beam_source = BeamSourceTestModel(
        pmd_variables={"pmd:beta_x": _ImpactLikePMDbeta_x(shape=(1,), read_only=True)},
        pmd_values={"pmd:beta_x": np.array([1.0], dtype=np.float32)},
    )
    beam_transport = BeamTransportTestModel(
        pmd_variables={"pmd:beta_x": _DistgenLikePMDbeta_x(shape=(1,), read_only=True)},
        pmd_values={"pmd:beta_x": np.array([2.0], dtype=np.float32)},
    )
    model = StagedModel([beam_source, beam_transport])

    assert "pmd:beta_x" in model.supported_variables
    values = model.get(["pmd:beta_x"])["pmd:beta_x"]
    assert np.allclose(values, [1.0, 2.0])


def test_staged_model_pmd_variable_set_raises_read_only() -> None:
    from lume.exceptions import ReadOnlyError

    beam_source = BeamSourceTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (1,))},
        pmd_values={"pmd:beta_x": np.array([1.0])},
    )
    beam_transport = BeamTransportTestModel(
        pmd_variables={"pmd:beta_x": _pmd_variable("pmd:beta_x", (1,))},
        pmd_values={"pmd:beta_x": np.array([2.0])},
    )
    model = StagedModel([beam_source, beam_transport])

    with pytest.raises(ReadOnlyError):
        model.set({"pmd:beta_x": np.array([1.0, 2.0])})
