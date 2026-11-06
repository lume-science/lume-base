from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from beamphysics import ParticleGroup
import h5py

from lume.model import LUMEModel
from lume.variables.variable import Variable


class InitialParticlesMixIn(ABC):
    """
    Mix in to LUMEModel to indicate support for initial particles.
    """

    @property
    @abstractmethod
    def initial_particles(self) -> ParticleGroup: ...

    @initial_particles.setter
    @abstractmethod
    def initial_particles(self, val: ParticleGroup): ...


class FinalParticlesMixIn(ABC):
    """
    Mix in to LUMEModel to indicate support for final particles.
    """

    @property
    @abstractmethod
    def final_particles(self) -> ParticleGroup: ...


class StagedModel(LUMEModel, InitialParticlesMixIn, FinalParticlesMixIn):
    """
    Composes multiple LUMEModel instances in sequence, passing final particles
    from each stage as initial particles to the next.
    """

    def __init__(self, lume_model_instances: list[LUMEModel]):
        """
        Initialize the `StagedModel` with a list of LUMEModel instances.

        Parameters
        ----------
        lume_model_instances: list[LUMEModel]
            Ordered list of LUMEModel instances to stage.
        """
        super().__init__()

        self._is_init = (
            False  # Flag to indicate if the model has been run start to end once
        )
        self.validate_lume_model_instances(lume_model_instances)
        self.lume_model_instances = lume_model_instances

    def coerce_evaluated_once(self) -> None:
        """Run the model at least once if it hasn't been run already."""
        if not self._is_init:
            self.run_start_to_end()
            self._is_init = True

    def run_start_to_end(self) -> None:
        """Run a single start-to-end propagation across all staged models."""

        incoming_particles = None
        for i, model in enumerate(self.lume_model_instances):
            if i > 0 and incoming_particles is not None:
                model.initial_particles = incoming_particles

            model.set({})

            if isinstance(model, FinalParticlesMixIn):
                incoming_particles = model.final_particles

    @classmethod
    def validate_lume_model_instances(cls, models: list[LUMEModel]):
        """
        Parameters
        ----------
        models: list[LUMEModel]
            Models to validate for staging compatibility.
        """
        for i, model in enumerate(models[:-1]):
            if not isinstance(model, FinalParticlesMixIn):
                raise ValueError(
                    f"Model {i} must implement FinalParticlesMixIn to stage models."
                )

        for i, model in enumerate(models[1:], start=1):
            if not isinstance(model, InitialParticlesMixIn):
                raise ValueError(
                    f"Model {i} must implement InitialParticlesMixIn to stage models."
                )

        seen: dict[str, int] = {}
        for i, model in enumerate(models):
            for name in model.supported_variables:
                if name in seen:
                    raise ValueError(
                        f"Variable '{name}' is defined in both model {seen[name]} and model {i}."
                    )
                seen[name] = i

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {
            name: var
            for model in self.lume_model_instances
            for name, var in model.supported_variables.items()
        }

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.coerce_evaluated_once()

        values = {}
        for model in self.lume_model_instances:
            model_names = [n for n in names if n in model.supported_variables]
            if model_names:
                values.update(model.get(model_names))
        return values

    def _set(self, values: dict[str, Any]) -> None:
        """
        Set variable values across the staged models.

        Parameters
        ----------
        values: dict[str, Any]
            Variable names and values to set across the staged models.
        """
        self.coerce_evaluated_once()
        incoming_particles = None
        for i, model in enumerate(self.lume_model_instances):
            model_values = {
                k: v for k, v in values.items() if k in model.supported_variables
            }

            if i > 0 and incoming_particles is not None:
                model.initial_particles = incoming_particles

            if model_values:
                model.set(model_values)

            if isinstance(model, FinalParticlesMixIn):
                incoming_particles = model.final_particles

    @property
    def initial_particles(self):
        first_model = self.lume_model_instances[0]
        if not isinstance(first_model, InitialParticlesMixIn):
            raise AttributeError(
                "Cannot access initial_particles because the first model does not implement InitialParticlesMixIn."
            )
        return first_model.initial_particles

    @initial_particles.setter
    def initial_particles(self, value: ParticleGroup):
        first_model = self.lume_model_instances[0]
        if not isinstance(first_model, InitialParticlesMixIn):
            raise AttributeError(
                "Cannot set initial_particles because the first model does not implement InitialParticlesMixIn."
            )
        first_model.initial_particles = value

    @property
    def final_particles(self):
        last_model = self.lume_model_instances[-1]
        if not isinstance(last_model, FinalParticlesMixIn):
            raise AttributeError(
                "Cannot access final_particles because the last model does not implement FinalParticlesMixIn."
            )
        return last_model.final_particles

    def reset(self):
        for model in self.lume_model_instances:
            model.reset()
        self._is_init = False

    def to_archive(self, location: str | Path | h5py.Group) -> None:
        """
        Archive the current state of the staged model into a single HDF5 file under separate groups or into an already existing HDF5 group as subgroups.

        Parameters
        ----------
        location: str | Path | h5py.Group
            A path to an HDF5 file or an HDF5 group where the staged model state will be archived.

        Returns
        -------
        None
        """
        if isinstance(location, (str, Path)):
            with h5py.File(location, "w") as f:
                self.to_archive(f)
        elif isinstance(location, h5py.Group):
            for i, model in enumerate(self.lume_model_instances):
                model_group = location.create_group(f"model_{i}")
                model.to_archive(model_group)
                model_group.attrs["model_class"] = model.__class__.__name__
        else:
            raise TypeError("location must be a str, Path, or h5py.Group object.")

    def to_archive_files(
        self, location: str | Path | list[str | Path]
    ) -> list[tuple[str, Path]]:
        """
        Archive each model in the staged model into separate HDF5 files.

        Parameters
        ----------
        location: str | Path | list[str | Path]
            A directory path where individual model files will be saved, or a list of file paths for each model.

        Returns
        -------
        archive_paths: list[tuple[str, Path]]
            Ordered (class name, path) pairs, retaining repeated model classes.
        """

        if isinstance(location, (str, Path)):
            location = Path(location)
            location.mkdir(parents=True, exist_ok=True)
            paths = [
                location / f"model_{i}.h5"
                for i in range(len(self.lume_model_instances))
            ]
        elif isinstance(location, list):
            if len(location) != len(self.lume_model_instances):
                raise ValueError(
                    "The length of the location list must match the number of models."
                )
            if not all(isinstance(path, (str, Path)) for path in location):
                raise TypeError("Each archive path must be a str or Path.")
            paths = [Path(path) for path in location]
        else:
            raise TypeError("location must be a str, Path, or list of paths.")

        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("Archive paths must be unique for each model.")
        archive_paths = []
        for model, path in zip(self.lume_model_instances, paths):
            model.to_archive(path)
            archive_paths.append((model.__class__.__name__, path))
        return archive_paths

    @classmethod
    def from_archive(
        cls,
        location: str | Path | h5py.Group,
        model_registry: Mapping[str, type[LUMEModel]] | None = None,
    ) -> "StagedModel":
        """
        Load a staged model from an HDF5 archive.

        Parameters
        ----------
        location: str | Path | h5py.Group
            The path to the HDF5 file or an open HDF5 group containing the archived models.
        model_registry: Mapping[str, type[LUMEModel]] | None
            Model class names mapped to their classes. Every archived class,
            including nested staged models, must be registered.

        Returns
        -------
        StagedModel
            An instance of the staged model containing the loaded models.
        """
        if isinstance(location, (str, Path)):
            with h5py.File(location, "r") as f:
                return cls.from_archive(f, model_registry)
        if not isinstance(location, h5py.Group):
            raise TypeError("location must be a str, Path, or h5py.Group object.")

        models = []
        for i in range(len(location)):
            model_group = location[f"model_{i}"]
            models.append(
                cls._load_archived_model(
                    model_group, model_group.attrs["model_class"], model_registry
                )
            )
        return cls(models)

    @classmethod
    def from_archive_paths(
        cls,
        archive_paths: Sequence[tuple[str, str | Path]],
        model_registry: Mapping[str, type[LUMEModel]] | None = None,
    ) -> "StagedModel":
        """
        Load a staged model from an ordered list of individual model archives.

        Parameters
        ----------
        archive_paths: Sequence[tuple[str, str | Path]]
            Ordered (class name, path) pairs, one per stage.
        model_registry: Mapping[str, type[LUMEModel]] | None
            Model class names mapped to their classes, including nested stages.

        Returns
        -------
        StagedModel
            An instance of the staged model containing the loaded models.
        """
        models = []
        for model_class_name, path in archive_paths:
            with h5py.File(path, "r") as f:
                models.append(
                    cls._load_archived_model(f, model_class_name, model_registry)
                )

        return cls(models)

    @staticmethod
    def _load_archived_model(
        location: h5py.Group,
        class_name: str,
        model_registry: Mapping[str, type[LUMEModel]] | None,
    ) -> LUMEModel:
        """
        Load an archived model from the given location using the specified class name and model registry.

        Parameters
        ----------
        location: h5py.Group
            The HDF5 group containing the archived model.
        class_name: str
            The name of the model class to load.
        model_registry: Mapping[str, type[LUMEModel]] | None
            A mapping of model class names to their corresponding classes.

        Returns
        -------
        LUMEModel
            An instance of the loaded model.
        """

        if model_registry is None or class_name not in model_registry:
            raise ValueError(f"Model class '{class_name}' is not registered.")
        model_class = model_registry[class_name]
        if not isinstance(model_class, type) or not issubclass(model_class, LUMEModel):
            raise TypeError(
                f"Registered class '{class_name}' must be a LUMEModel subclass."
            )
        if issubclass(model_class, StagedModel):
            return model_class.from_archive(location, model_registry)
        return model_class.from_archive(location)
