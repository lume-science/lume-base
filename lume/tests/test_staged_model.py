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

    def __init__(self):
        self.call_count_set = 0
        self._variables = {
            "source_phase": ScalarVariable(
                name="source_phase", default_value=0.0, read_only=False
            ),
        }
        self._state = {"source_phase": 0.0}
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

    def __init__(self):
        self.call_count_set = 0
        self._variables = {
            "transport_scale": ScalarVariable(
                name="transport_scale", default_value=1.0, read_only=False
            ),
        }
        self._state = {"transport_scale": 1.0}
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
