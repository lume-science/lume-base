import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from beamphysics import ParticleGroup

from lume.model import LUMEModel
from lume.variables.ndvariable import NDVariable
from lume.variables.pmd import PMDVariable
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


def _canonical_pmd_class(var: PMDVariable) -> type[PMDVariable]:
    """Return the `create_pmd_variable`-generated ancestor class for `var`.

    This is the class in `var`'s MRO whose direct base is `PMDVariable` itself
    (e.g. `PMDbeta_x`), used to compare pmd: variables across different
    simulator-specific subclasses (e.g. `ImpactPMDbeta_x` vs `DistgenPMDbeta_x`)
    that share the same canonical name/unit/dtype contract.
    """
    for klass in type(var).__mro__:
        if klass.__bases__ == (PMDVariable,):
            return klass
    return type(var)


def _model_label(index: int, model: LUMEModel) -> str:
    """Identify a staged model in warning messages by index and class name."""
    return f"model {index} ({type(model).__name__})"


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
                # skip shared pmd: variables that are allowed to be in multiple models
                if name.startswith("pmd:"):
                    continue

                if name in seen:
                    raise ValueError(
                        f"Variable '{name}' is defined in both model {seen[name]} and model {i}."
                    )
                seen[name] = i

        # trigger pmd: validation eagerly so problems are warned about at construction
        cls._combined_pmd_variables(models)

    @staticmethod
    def _combined_pmd_variables(models: list[LUMEModel]) -> dict[str, NDVariable]:
        """
        Build read-only PMDVariables representing pmd: outputs, concatenated
        along axis 0 across all models that support a given name. Each staged
        model may use a different simulator-specific PMDVariable subclass for
        a given name (e.g. `ImpactPMDbeta_x` vs `DistgenPMDbeta_x`), as long as
        they share the same canonical `create_pmd_variable`-generated ancestor
        (see `_canonical_pmd_class`). A pmd: name that isn't supported by every
        model as a compatible PMDVariable is excluded from the result and a
        warning is issued (once per unique message, per the default warnings
        filter) rather than raising.

        Parameters
        ----------
        models: list[LUMEModel]
            Models to inspect for pmd: variables.
        """
        pmd_names: set[str] = set()
        for model in models:
            pmd_names.update(
                name for name in model.supported_variables if name.startswith("pmd:")
            )

        combined: dict[str, NDVariable] = {}
        for name in pmd_names:
            missing = [
                _model_label(i, model)
                for i, model in enumerate(models)
                if name not in model.supported_variables
            ]
            if missing:
                warnings.warn(
                    f"pmd: variable '{name}' is not supported by "
                    f"{', '.join(missing)}; excluding it from "
                    "supported_variables.",
                    stacklevel=2,
                )
                continue

            stage_vars = [model.supported_variables[name] for model in models]

            not_pmd = [
                _model_label(i, model)
                for i, (model, var) in enumerate(zip(models, stage_vars))
                if not isinstance(var, PMDVariable)
            ]
            if not_pmd:
                warnings.warn(
                    f"pmd: variable '{name}' is not a PMDVariable on "
                    f"{', '.join(not_pmd)}; excluding it from "
                    "supported_variables.",
                    stacklevel=2,
                )
                continue

            first_type = _canonical_pmd_class(stage_vars[0])
            mismatched = [
                _model_label(i, model)
                for i, (model, var) in enumerate(zip(models, stage_vars))
                if _canonical_pmd_class(var) is not first_type
            ]
            if mismatched:
                warnings.warn(
                    f"pmd: variable '{name}' uses a different canonical "
                    f"PMDVariable class (i.e. a different pmd: name/unit "
                    f"contract) on {', '.join(mismatched)} than "
                    f"{_model_label(0, models[0])}; excluding it from "
                    "supported_variables.",
                    stacklevel=2,
                )
                continue

            # PMDVariable enforces 1D shapes, so only the leading (concatenated) dimension can vary.
            combined_shape = (sum(var.shape[0] for var in stage_vars),)
            combined[name] = stage_vars[0].model_copy(
                update={"shape": combined_shape},
            )

        return combined

    @property
    def supported_variables(self) -> dict[str, Variable]:
        variables = {
            name: var
            for model in self.lume_model_instances
            for name, var in model.supported_variables.items()
            if not name.startswith("pmd:")
        }
        variables.update(self._combined_pmd_variables(self.lume_model_instances))
        return variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.coerce_evaluated_once()

        values = {}

        # concatenate pmd: variable values across all stages, in stage order
        for name in names:
            if name.startswith("pmd:"):
                arrays = [
                    model.get([name])[name] for model in self.lume_model_instances
                ]
                dtype = self.supported_variables[name].dtype
                values[name] = np.concatenate(arrays, axis=0).astype(dtype)

        for model in self.lume_model_instances:
            model_names = [
                n
                for n in names
                if n in model.supported_variables and not n.startswith("pmd:")
            ]
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
