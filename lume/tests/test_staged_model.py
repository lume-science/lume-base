from typing import Any

import numpy as np
import pytest

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

from lume.model import LUMEModel
from lume.staged_model import (
    FinalParticlesMixIn,
    InitialParticlesMixIn,
    StagedModel,
    get_pmd_class_mapping,
)
from lume.variables import ScalarVariable, Variable
from lume.variables.pmd import PMDVariable, PMDbeta_x, PMDmodel_index


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

    name: str = "pmd:beta_x"

    def _get(self, simulator: Any) -> Any:
        raise NotImplementedError


class _OtherTestPMDVariable(PMDVariable):
    """A second, distinct PMDVariable subclass used to test subclass-mismatch validation."""

    name: str = "pmd:other"

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


def test_staged_model_prefixed_scalar_is_not_treated_as_pmd() -> None:
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
        UserWarning, match=r"not supported by model 0 \(BeamSourceTestModel\)"
    ):
        model = StagedModel([beam_source, beam_transport])

    assert isinstance(model.supported_variables["pmd:beta_x"], ScalarVariable)
    assert model.get_value("pmd:beta_x") == 1.0


def test_staged_model_prefixed_scalar_can_be_set(monkeypatch) -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "pmd:control": ScalarVariable(name="pmd:control", default_value=1.0)
        },
        pmd_values={"pmd:control": 1.0},
    )
    monkeypatch.setattr(source, "_set", source._state.update)
    model = StagedModel([source, BeamTransportTestModel()])

    model.set({"pmd:control": 2.0})

    assert model.get_value("pmd:control") == 2.0
    assert "pmd:model_index" not in model.supported_variables


def test_staged_model_scalar_cannot_be_overwritten_by_synthetic_index() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "beta": _pmd_variable("pmd:beta_x", (1,)),
            "pmd:model_index": ScalarVariable(name="pmd:model_index"),
        }
    )
    transport = BeamTransportTestModel(
        pmd_variables={"beta": _pmd_variable("pmd:beta_x", (1,))}
    )
    with pytest.raises(ValueError, match=r"staged-model name 'pmd:model_index'"):
        StagedModel([source, transport])


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
        match=r"not supported by model",
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


def test_get_pmd_class_mapping() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "source_beta": _ImpactLikePMDbeta_x(name="pmd:source_beta", shape=(2,)),
            "source_only": _pmd_variable("pmd:source_only", (2,)),
        }
    )
    transport = BeamTransportTestModel(
        pmd_variables={
            "transport_beta": _DistgenLikePMDbeta_x(
                name="pmd:transport_beta", shape=(1,)
            ),
            "transport_only": _pmd_variable(
                "pmd:transport_only", (1,), cls=_OtherTestPMDVariable
            ),
        }
    )

    with pytest.warns(UserWarning, match="not supported by model"):
        assert get_pmd_class_mapping([source, transport]) == {
            PMDbeta_x: ["source_beta", "transport_beta"]
        }
    with pytest.warns(UserWarning, match="not supported by model"):
        assert get_pmd_class_mapping([transport, source]) == {
            PMDbeta_x: ["transport_beta", "source_beta"]
        }
    with pytest.warns(UserWarning, match="not supported by model 1"):
        assert get_pmd_class_mapping([source, BeamTransportTestModel()]) == {}


def test_get_pmd_class_mapping_empty_and_single_stage() -> None:
    assert get_pmd_class_mapping([]) == {}
    source = BeamSourceTestModel(
        pmd_variables={"beta": _ImpactLikePMDbeta_x(shape=(1,))}
    )
    assert get_pmd_class_mapping([source]) == {PMDbeta_x: ["beta"]}
    with pytest.warns(UserWarning, match="not supported by model 0"):
        assert get_pmd_class_mapping([BeamTransportTestModel(), source]) == {}


def test_get_pmd_class_mapping_duplicate_class() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "first": _ImpactLikePMDbeta_x(shape=(1,)),
            "second": _DistgenLikePMDbeta_x(shape=(1,)),
        }
    )
    with pytest.raises(ValueError, match="model 0 .*PMDbeta_x.*first.*second"):
        get_pmd_class_mapping([source, BeamTransportTestModel()])


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


@pytest.mark.parametrize(
    ("source_type", "transport_type"),
    [
        (_ImpactLikePMDbeta_x, _ImpactLikePMDbeta_x),  # Same concrete subclass.
        (_ImpactLikePMDbeta_x, _DistgenLikePMDbeta_x),  # Shared canonical PMDbeta_x.
    ],
)
def test_staged_model_pmd_class_matching_uses_model_specific_keys(
    source_type: type[PMDVariable], transport_type: type[PMDVariable]
) -> None:
    source = BeamSourceTestModel(
        pmd_variables={"source_beta": source_type(name="pmd:source_beta", shape=(2,))},
        pmd_values={"source_beta": np.array([1.0, 2.0], dtype=np.float32)},
    )
    transport = BeamTransportTestModel(
        pmd_variables={
            "transport_beta": transport_type(name="pmd:transport_beta", shape=(1,))
        },
        pmd_values={"transport_beta": np.array([3.0], dtype=np.float32)},
    )
    model = StagedModel([source, transport])

    assert set(model.supported_variables) == {
        "source_phase",
        "transport_scale",
        "pmd:beta_x",
        "pmd:model_index",
    }
    combined = model.supported_variables["pmd:beta_x"]
    assert combined.name == "pmd:beta_x"
    assert combined.shape == (3,)
    assert combined.read_only is True
    values = model.get(["pmd:beta_x", "pmd:model_index", "source_phase"])
    np.testing.assert_array_equal(values["pmd:beta_x"], [1.0, 2.0, 3.0])
    assert values["pmd:beta_x"].dtype == np.float32
    np.testing.assert_array_equal(values["pmd:model_index"], [0, 0, 1])
    assert values["source_phase"] == 0.0


def test_staged_model_pmd_custom_class_uses_default_name() -> None:
    source = BeamSourceTestModel(
        pmd_variables={"source_beta": _pmd_variable("pmd:source_beta", (1,))},
        pmd_values={"source_beta": np.array([1.0], dtype=np.float32)},
    )
    transport = BeamTransportTestModel(
        pmd_variables={"transport_beta": _pmd_variable("pmd:transport_beta", (1,))},
        pmd_values={"transport_beta": np.array([2.0], dtype=np.float32)},
    )
    model = StagedModel([source, transport])

    assert model.supported_variables["pmd:beta_x"].name == "pmd:beta_x"
    assert "source_beta" not in model.supported_variables
    np.testing.assert_array_equal(model.get_value("pmd:beta_x"), [1.0, 2.0])


def test_staged_model_pmd_class_without_default_name_raises() -> None:
    class UnnamedPMDVariable(PMDVariable):
        def _get(self, simulator: Any) -> Any:
            raise NotImplementedError

    source = BeamSourceTestModel(
        pmd_variables={
            "source_beta": UnnamedPMDVariable(name="pmd:source_beta", shape=(1,))
        }
    )
    transport = BeamTransportTestModel(
        pmd_variables={
            "transport_beta": UnnamedPMDVariable(name="pmd:transport_beta", shape=(1,))
        }
    )
    with pytest.raises(ValueError, match=r"UnnamedPMDVariable.*non-empty default name"):
        StagedModel([source, transport])


def test_staged_model_pmd_class_with_empty_default_name_raises() -> None:
    class EmptyNamePMDVariable(PMDVariable):
        name: str = ""

        def _get(self, simulator: Any) -> Any:
            raise NotImplementedError

    source = BeamSourceTestModel(
        pmd_variables={
            "source_beta": EmptyNamePMDVariable(name="pmd:source_beta", shape=(1,))
        }
    )
    with pytest.raises(
        ValueError, match=r"EmptyNamePMDVariable.*non-empty default name"
    ):
        StagedModel([source])


def test_staged_model_pmd_duplicate_class_raises() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "first_beta": _ImpactLikePMDbeta_x(shape=(1,)),
            "second_beta": _DistgenLikePMDbeta_x(shape=(1,)),
        }
    )
    with pytest.raises(
        ValueError, match=r"model 0 .*PMDbeta_x.*first_beta.*second_beta"
    ):
        StagedModel([source, BeamTransportTestModel()])


def test_staged_model_pmd_canonical_name_collision_raises() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "beta": _ImpactLikePMDbeta_x(shape=(1,)),
            "pmd:beta_x": _pmd_variable("pmd:beta_x", (1,)),
        }
    )
    transport = BeamTransportTestModel(
        pmd_variables={
            "beta": _DistgenLikePMDbeta_x(shape=(1,)),
            "pmd:beta_x": _pmd_variable("pmd:beta_x", (1,)),
        }
    )
    with pytest.raises(ValueError, match=r"staged-model name 'pmd:beta_x'"):
        StagedModel([source, transport])


def test_staged_model_pmd_missing_class_with_model_specific_key_is_excluded() -> None:
    source = BeamSourceTestModel(
        pmd_variables={"source_beta": _ImpactLikePMDbeta_x(shape=(2,))}
    )
    with pytest.warns(UserWarning, match=r"'pmd:beta_x'.*not supported by model 1"):
        model = StagedModel([source, BeamTransportTestModel()])

    with pytest.warns(UserWarning, match="not supported by model 1"):
        assert set(model.supported_variables) == {"source_phase", "transport_scale"}


def test_staged_model_pmd_custom_name_collision_with_ordinary_variable_raises() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "beta": _pmd_variable("pmd:beta_x", (1,)),
            "pmd:beta_x": ScalarVariable(name="pmd:beta_x"),
        }
    )
    transport = BeamTransportTestModel(
        pmd_variables={"transport_beta": _pmd_variable("pmd:beta_x", (1,))}
    )
    with pytest.raises(ValueError, match=r"staged-model name 'pmd:beta_x'"):
        StagedModel([source, transport])


def test_staged_model_pmd_custom_name_cannot_replace_model_index() -> None:
    class IndexNamedPMDVariable(PMDVariable):
        name: str = "pmd:model_index"

        def _get(self, simulator: Any) -> Any:
            raise NotImplementedError

    source = BeamSourceTestModel(
        pmd_variables={"source_index": IndexNamedPMDVariable(shape=(1,))}
    )
    transport = BeamTransportTestModel(
        pmd_variables={"transport_index": IndexNamedPMDVariable(shape=(1,))}
    )
    with pytest.raises(ValueError, match=r"Reserved.*'pmd:model_index'"):
        StagedModel([source, transport])


def test_staged_model_pmd_multiple_classes_match_independent_of_order() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "source_beta": _ImpactLikePMDbeta_x(shape=(1,)),
            "pmd:other": _pmd_variable("pmd:other", (1,), cls=_OtherTestPMDVariable),
        },
        pmd_values={
            "source_beta": np.array([1.0], dtype=np.float32),
            "pmd:other": np.array([10.0], dtype=np.float32),
        },
    )
    transport = BeamTransportTestModel(
        pmd_variables={
            "transport_other": _pmd_variable(
                "pmd:other", (2,), cls=_OtherTestPMDVariable
            ),
            "transport_beta": _DistgenLikePMDbeta_x(shape=(2,)),
        },
        pmd_values={
            "transport_other": np.array([20.0, 30.0], dtype=np.float32),
            "transport_beta": np.array([2.0, 3.0], dtype=np.float32),
        },
    )
    model = StagedModel([source, transport])
    values = model.get(["pmd:beta_x", "pmd:other", "pmd:model_index"])
    np.testing.assert_array_equal(values["pmd:beta_x"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(values["pmd:other"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(values["pmd:model_index"], [0, 1, 1])


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # Both variables retain their default staged-model names.
        {PMDbeta_x: "beta"},  # Rename beta; keep the default index name.
        {PMDmodel_index: "stage"},  # Rename the index; keep the default beta name.
        {PMDbeta_x: "beta", PMDmodel_index: "stage"},  # Rename both variables.
    ],
)
def test_staged_model_pmd_name_overrides(overrides) -> None:
    original_default = np.array([1.0, 2.0], dtype=np.float32)
    source_variable = _ImpactLikePMDbeta_x(
        name="source_internal", shape=(2,), default_value=original_default
    )
    source = BeamSourceTestModel(
        pmd_variables={"source_key": source_variable},
        pmd_values={"source_key": original_default},
    )
    transport = BeamTransportTestModel(
        pmd_variables={
            "transport_key": _DistgenLikePMDbeta_x(
                name="transport_internal", shape=(1,)
            )
        },
        pmd_values={"transport_key": np.array([3.0], dtype=np.float32)},
    )
    beta_name = overrides.get(PMDbeta_x, "pmd:beta_x")
    index_name = overrides.get(PMDmodel_index, "pmd:model_index")
    names = overrides.copy()
    model = StagedModel([source, transport], pmd_variable_names=names)
    names[PMDbeta_x] = "mutated"

    assert source.call_count_set == transport.call_count_set == 0
    assert set(model.supported_variables) == {
        "source_phase",
        "transport_scale",
        beta_name,
        index_name,
    }
    combined = model.supported_variables[beta_name]
    assert isinstance(combined, _ImpactLikePMDbeta_x)
    assert combined.name == beta_name
    assert combined.shape == (3,)
    assert combined.unit == "m"
    assert combined.dtype == np.float32
    assert combined.read_only is True
    assert combined.default_value is None
    assert source_variable.name == "source_internal"
    assert source_variable.shape == (2,)
    assert source_variable.default_value is original_default

    values = model.get([beta_name, index_name])
    np.testing.assert_array_equal(values[beta_name], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(values[index_name], [0, 0, 1])
    assert values[beta_name].dtype == np.float32
    assert values[index_name].dtype == np.int64
    assert model.supported_variables[index_name].name == index_name
    from lume.exceptions import ReadOnlyError

    with pytest.raises(ReadOnlyError):
        model.set({index_name: values[index_name]})
    with pytest.raises(ReadOnlyError):
        model.set({beta_name: values[beta_name]})


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ([], TypeError),  # Overrides must be a dictionary, not a list.
        ({"pmd:beta_x": "beta"}, ValueError),  # Keys must be classes, not strings.
        ({PMDVariable: "beta"}, ValueError),  # The base class is not a canonical type.
        ({ScalarVariable: "beta"}, ValueError),  # The key is not a PMD variable type.
        (
            {_ImpactLikePMDbeta_x: "beta"},
            ValueError,
        ),  # Use canonical PMDbeta_x instead.
        ({PMDbeta_x: ""}, ValueError),  # Staged-model names must not be empty.
        ({PMDbeta_x: "  "}, ValueError),  # Whitespace-only names are also invalid.
        ({PMDbeta_x: 1}, ValueError),  # Staged-model names must be strings.
    ],
)
def test_staged_model_pmd_invalid_name_overrides(overrides, error) -> None:
    with pytest.raises(error, match="canonical|non-empty|dictionary"):
        StagedModel([BeamSourceTestModel()], pmd_variable_names=overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {PMDbeta_x: "source_phase"},  # Beta collides with an ordinary variable.
        {PMDbeta_x: "shared", _OtherTestPMDVariable: "shared"},  # Two types, one name.
        {PMDmodel_index: "pmd:beta_x"},  # The index collides with beta's default name.
        {
            PMDmodel_index: "source_phase"
        },  # The index collides with an ordinary variable.
    ],
)
def test_staged_model_pmd_override_collisions(overrides) -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "beta": _ImpactLikePMDbeta_x(shape=(1,)),
            "other": _OtherTestPMDVariable(shape=(1,)),
        }
    )
    with pytest.raises(ValueError, match="staged-model name"):
        StagedModel([source], pmd_variable_names=overrides)


def test_staged_model_pmd_override_resolves_default_collisions() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "beta": _ImpactLikePMDbeta_x(shape=(1,)),
            "other_beta": _TestPMDVariable(shape=(1,)),
            "pmd:beta_x": ScalarVariable(name="pmd:beta_x"),
            "pmd:model_index": ScalarVariable(name="pmd:model_index"),
        }
    )
    model = StagedModel(
        [source],
        pmd_variable_names={
            PMDbeta_x: "beta",
            _TestPMDVariable: "other_beta",
            PMDmodel_index: "stage",
        },
    )
    assert set(model.supported_variables) == {
        "source_phase",
        "beta",
        "other_beta",
        "stage",
        "pmd:beta_x",
        "pmd:model_index",
    }
    assert isinstance(model.supported_variables["pmd:model_index"], ScalarVariable)


def test_staged_model_pmd_override_supplies_missing_default_name() -> None:
    class UnnamedPMDVariable(PMDVariable):
        def _get(self, simulator: Any) -> Any:
            raise NotImplementedError

    source = BeamSourceTestModel(
        pmd_variables={"local": UnnamedPMDVariable(name="internal", shape=(1,))},
        pmd_values={"local": np.array([1.0], dtype=np.float32)},
    )
    model = StagedModel([source], pmd_variable_names={UnnamedPMDVariable: "custom"})
    assert model.supported_variables["custom"].name == "custom"
    np.testing.assert_array_equal(model.get_value("custom"), [1.0])


def test_staged_model_pmd_override_does_not_enable_missing_class() -> None:
    source = BeamSourceTestModel(
        pmd_variables={"beta": _ImpactLikePMDbeta_x(shape=(1,))}
    )
    with pytest.warns(UserWarning, match="not supported by model 1"):
        model = StagedModel(
            [source, BeamTransportTestModel()],
            pmd_variable_names={PMDbeta_x: "beta", PMDmodel_index: "stage"},
        )
    with pytest.warns(UserWarning, match="not supported by model 1"):
        assert set(model.supported_variables) == {"source_phase", "transport_scale"}
    empty = StagedModel(
        [BeamSourceTestModel()], pmd_variable_names={PMDbeta_x: "unused"}
    )
    assert set(empty.supported_variables) == {"source_phase"}


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "dtype": np.dtype(np.float64)
        },  # Transport float64 differs from source float32.
        {"unit": "cm"},  # Transport units differ from the source's unset unit.
    ],
)
def test_staged_model_pmd_incompatible_metadata(metadata) -> None:
    source = BeamSourceTestModel(pmd_variables={"beta": _TestPMDVariable(shape=(1,))})
    transport = BeamTransportTestModel(
        pmd_variables={"beta": _TestPMDVariable(shape=(1,), **metadata)}
    )
    with pytest.raises(ValueError, match="incompatible unit or dtype.*model 1"):
        StagedModel([source, transport])


def test_staged_model_pmd_inconsistent_lengths() -> None:
    source = BeamSourceTestModel(
        pmd_variables={
            "beta": _ImpactLikePMDbeta_x(shape=(1,)),
            "other": _OtherTestPMDVariable(shape=(1,)),
        }
    )
    transport = BeamTransportTestModel(
        pmd_variables={
            "beta": _DistgenLikePMDbeta_x(shape=(2,)),
            "other": _OtherTestPMDVariable(shape=(3,)),
        }
    )
    with pytest.raises(ValueError, match="Inconsistent PMD lengths.*model 1"):
        StagedModel([source, transport])


def test_staged_model_pmd_metadata_refreshes_after_evaluation(monkeypatch) -> None:
    source = BeamSourceTestModel(
        pmd_variables={"beta": _ImpactLikePMDbeta_x(shape=(1,))},
        pmd_values={"beta": np.array([1.0], dtype=np.float32)},
    )
    model = StagedModel([source])
    assert model.supported_variables["pmd:beta_x"].shape == (1,)

    def update_shape(values):
        source._variables["beta"].shape = (2,)
        source._state["beta"] = np.array([2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(source, "_set", update_shape)
    values = model.get(["pmd:beta_x", "pmd:model_index"])
    np.testing.assert_array_equal(values["pmd:beta_x"], [2.0, 3.0])
    np.testing.assert_array_equal(values["pmd:model_index"], [0, 0])
    assert model.supported_variables["pmd:beta_x"].shape == (2,)
    assert model.supported_variables["pmd:model_index"].shape == (2,)


def test_staged_model_rejects_empty_models() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        StagedModel([])
