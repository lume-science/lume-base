from pathlib import Path
from typing import Any
import json

import h5py
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


class ArchivedBeamModel(BeamSourceTestModel):
    def __init__(self, name: str, value: float):
        super().__init__()
        self._variables = {name: ScalarVariable(name=name, default_value=value)}
        self._state = {name: value}
        self._initial_state = self._state.copy()

    def _set(self, values: dict[str, Any]) -> None:
        self._state.update(values)
        self._final_particles = self._initial_particles

    def to_archive(self, location: str | Path | h5py.Group) -> None:
        if isinstance(location, (str, Path)):
            with h5py.File(location, "w") as group:
                self.to_archive(group)
            return
        name, value = next(iter(self._state.items()))
        # Store the state value
        location.attrs["value"] = value
        # Serialize the complete ScalarVariable to JSON
        variable = self._variables[name]
        variable_json = json.dumps(variable.model_dump())
        location.attrs["variable_json"] = variable_json

    @classmethod
    def from_archive(cls, location: str | Path | h5py.Group) -> "ArchivedBeamModel":
        if isinstance(location, (str, Path)):
            with h5py.File(location, "r") as group:
                return cls.from_archive(group)

        # Deserialize the ScalarVariable from JSON
        variable_json = location.attrs["variable_json"]
        variable_dict = json.loads(variable_json)
        # Remove the variable_class key as it's not needed for construction
        variable_dict.pop("variable_class", None)
        # Recreate the variable
        variable = ScalarVariable(**variable_dict)

        # Create an instance and set up its variables properly
        instance = cls.__new__(cls)
        BeamSourceTestModel.__init__(instance)
        instance._variables = {variable.name: variable}
        instance._state = {variable.name: float(location.attrs["value"])}
        instance._initial_state = instance._state.copy()
        beam = make_test_particle_group(x_offset=0.0)
        instance._initial_particles = beam
        instance._final_particles = beam
        return instance


@pytest.fixture
def archive_model() -> StagedModel:
    return StagedModel([ArchivedBeamModel(f"stage_{i}", float(i)) for i in range(12)])


@pytest.mark.parametrize("location_type", ["path", "str", "file", "group"])
def test_staged_archive_round_trip(tmp_path, archive_model, location_type):
    registry = {"ArchivedBeamModel": ArchivedBeamModel}
    path = tmp_path / "stages.h5"
    if location_type in ("path", "str"):
        location = path if location_type == "path" else str(path)
        archive_model.to_archive(location)
        restored = StagedModel.from_archive(location, registry)
    else:
        with h5py.File(path, "w") as file:
            location = file if location_type == "file" else file.create_group("stages")
            archive_model.to_archive(location)
            restored = StagedModel.from_archive(location, registry)
            assert file.id.valid

    names = [f"stage_{i}" for i in range(12)]
    assert list(restored.supported_variables) == names
    assert restored.get(names) == dict(zip(names, map(float, range(12))))
    assert all(
        type(model) is ArchivedBeamModel for model in restored.lume_model_instances
    )


@pytest.mark.parametrize("location_type", ["path", "str", "paths", "strings"])
def test_staged_archive_files_preserve_repeated_classes(
    tmp_path, archive_model, location_type
):
    if location_type in ("path", "str"):
        location = tmp_path / "stages"
        expected_paths = [location / f"model_{i}.h5" for i in range(12)]
        if location_type == "str":
            location = str(location)
    else:
        expected_paths = [tmp_path / f"custom_{i}.h5" for i in range(12)]
        location = (
            expected_paths
            if location_type == "paths"
            else [str(path) for path in expected_paths]
        )
    records = archive_model.to_archive_files(location)
    assert records == [("ArchivedBeamModel", path) for path in expected_paths]
    assert all(path.is_file() for _, path in records)

    input_records = (
        [(name, str(path)) for name, path in records]
        if location_type == "strings"
        else records
    )
    restored = StagedModel.from_archive_paths(
        input_records, {"ArchivedBeamModel": ArchivedBeamModel}
    )
    names = [f"stage_{i}" for i in range(12)]
    assert list(restored.supported_variables) == names
    assert restored.get(names) == dict(zip(names, map(float, range(12))))


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("registry", [None, {}])
def test_staged_archive_requires_registered_class(
    tmp_path, archive_model, separate, registry
):
    with pytest.raises(
        ValueError, match="Model class 'ArchivedBeamModel' is not registered"
    ):
        if separate:
            records = archive_model.to_archive_files(tmp_path)
            StagedModel.from_archive_paths(records, registry)
        else:
            path = tmp_path / "stages.h5"
            archive_model.to_archive(path)
            StagedModel.from_archive(path, registry)


@pytest.mark.parametrize("separate", [False, True])
def test_staged_archive_nested_round_trip(tmp_path, archive_model, separate):
    model = StagedModel([archive_model])
    registry = {"StagedModel": StagedModel, "ArchivedBeamModel": ArchivedBeamModel}
    if separate:
        restored = StagedModel.from_archive_paths(
            model.to_archive_files(tmp_path), registry
        )
    else:
        path = tmp_path / "nested.h5"
        model.to_archive(path)
        restored = StagedModel.from_archive(path, registry)
    assert isinstance(restored.lume_model_instances[0], StagedModel)
    assert len(restored.lume_model_instances[0].lume_model_instances) == 12
    names = list(model.supported_variables)
    assert restored.get(names) == model.get(names)


@pytest.mark.parametrize("method", ["to_archive", "from_archive", "to_archive_files"])
def test_staged_archive_rejects_invalid_location(archive_model, method):
    with pytest.raises(TypeError, match="location must be"):
        getattr(archive_model, method)(123)


def test_staged_archive_files_validate_paths_before_writing(tmp_path, archive_model):
    path = tmp_path / "model.h5"
    with pytest.raises(ValueError, match="length"):
        archive_model.to_archive_files([path])
    with pytest.raises(TypeError, match="Each archive path"):
        archive_model.to_archive_files([path] * 11 + [123])
    with pytest.raises(ValueError, match="unique"):
        archive_model.to_archive_files([path] * 12)
    assert not path.exists()


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


def test_archived_beam_model_preserves_scalar_variable_type(tmp_path):
    """Test that ArchivedBeamModel preserves complete ScalarVariable information during archive/restore."""

    # Create a model with a ScalarVariable that has non-default properties
    model = ArchivedBeamModel("dummy", 1.0)
    var = ScalarVariable(
        name="beam_energy",
        default_value=100.0,
        value_range=(50.0, 500.0),
        unit="MeV",
        description="Beam energy setting",
        read_only=False,
    )
    model._variables = {"beam_energy": var}
    model._state = {"beam_energy": 100.0}
    model._initial_state = model._state.copy()

    # Archive the model
    archive_path = tmp_path / "model.h5"
    model.to_archive(str(archive_path))

    # Restore from archive
    restored = ArchivedBeamModel.from_archive(str(archive_path))
    restored_var = restored.supported_variables["beam_energy"]

    # Verify all properties are preserved
    assert restored_var.name == var.name
    assert restored_var.value_range == var.value_range
    assert restored_var.unit == var.unit
    assert restored_var.description == var.description
    assert restored_var.read_only == var.read_only
    assert restored_var.default_value == var.default_value

    # Verify state is also preserved
    assert restored.get(["beam_energy"]) == {"beam_energy": 100.0}
