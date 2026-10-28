import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from beamphysics import ParticleGroup

from lume.model import LUMEModel
from lume.variables.pmd import PMDVariable, PMDmodel_index
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
    """Return the canonical PMD ancestor class for `var`.

    This is the class in `var`'s MRO whose direct base is `PMDVariable` itself
    (e.g. `PMDbeta_x`), used to compare PMD variables across different
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


def get_pmd_variables(model: LUMEModel) -> dict[type[PMDVariable], str]:
    """Return a mapping from canonical PMDVariable class types to variable names for a given model."""
    pmd_vars = {}
    for name, var in model.supported_variables.items():
        if isinstance(var, PMDVariable):
            canonical_class = _canonical_pmd_class(var)

            # Check for duplicate canonical classes within the same model
            if canonical_class in pmd_vars:
                raise ValueError(
                    f"{canonical_class.__name__} has duplicate variables "
                    f"'{pmd_vars[canonical_class]}' and '{name}'."
                )
            pmd_vars[canonical_class] = name
    return pmd_vars


def _warn_if_pmd_type_missing(
    pmd_type: type[PMDVariable],
    models: list[LUMEModel],
    model_pmd_variables: list[dict[type[PMDVariable], str]],
) -> bool:
    """Return whether a pmd type is missing and warn about the affected stages."""
    missing = [
        _model_label(index, models[index])
        for index, mapping in enumerate(model_pmd_variables)
        if pmd_type not in mapping
    ]
    if missing:
        default_name = pmd_type.model_fields["name"].default
        warnings.warn(
            f"PMD variable '{default_name}' ({pmd_type.__name__}) is "
            f"not supported by {', '.join(missing)}; excluding it.",
            UserWarning,
            stacklevel=3,
        )
    return bool(missing)


def get_pmd_class_mapping(
    models: list[LUMEModel],
) -> dict[type[PMDVariable], list[str]]:
    """Map shared canonical PMD classes to stage-ordered model-specific keys.

    Submodels must not expose PMDmodel_index variables or its subclasses.
    Duplicate matches raise an error.
    Classes missing from any stage are excluded with a warning.
    """
    model_pmd_variables = []
    for index, model in enumerate(models):
        try:
            model_pmd_variables.append(get_pmd_variables(model))
        except ValueError as exc:
            raise ValueError(f"{_model_label(index, model)}: {exc}") from exc

    pmd_types = dict.fromkeys(
        pmd_type for mapping in model_pmd_variables for pmd_type in mapping
    )
    combined = {}
    for pmd_type in pmd_types:
        if _warn_if_pmd_type_missing(pmd_type, models, model_pmd_variables):
            continue
        combined[pmd_type] = [mapping[pmd_type] for mapping in model_pmd_variables]
    return combined


def _validate_pmd_variable_names(
    names: dict[type[PMDVariable], str] | None,
) -> dict[type[PMDVariable], str]:
    """Validate and copy staged-model name overrides keyed by canonical PMD class."""
    if names is None:
        return {}
    if not isinstance(names, dict):
        raise TypeError("pmd_variable_names must be a dictionary.")
    for pmd_type, name in names.items():
        if (
            not isinstance(pmd_type, type)
            or not issubclass(pmd_type, PMDVariable)
            or pmd_type.__bases__ != (PMDVariable,)
        ):
            raise ValueError(
                "pmd_variable_names keys must be canonical PMDVariable classes, "
                f"not {pmd_type!r}."
            )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("PMD staged-model names must be non-empty strings.")
    return names.copy()


def _resolve_pmd_name(
    pmd_type: type[PMDVariable], names: dict[type[PMDVariable], str]
) -> str:
    """Resolve a staged-model name without using a model-specific instance name."""
    name = names.get(pmd_type, pmd_type.model_fields["name"].default)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"{pmd_type.__name__} must have a non-empty default name "
            "or an explicit pmd_variable_names override."
        )
    return name


def _combine_pmd_variable(
    name: str, variables: list[PMDVariable], models: list[LUMEModel]
) -> PMDVariable:
    """Copy a concrete staged PMD variable with compatible metadata and a combined shape."""
    first = variables[0]
    for index, variable in enumerate(variables):
        if variable.unit != first.unit or variable.dtype != first.dtype:
            raise ValueError(
                f"PMD staged-model name '{name}' has incompatible unit or dtype in "
                f"{_model_label(index, models[index])}."
            )
    combined = first.model_copy(deep=True)
    combined.default_value = None
    combined.name = name
    combined.shape = (sum(variable.shape[0] for variable in variables),)
    combined.read_only = True
    return combined


def _build_pmd_metadata(
    models: list[LUMEModel], name_overrides: dict[type[PMDVariable], str]
) -> tuple[dict[str, PMDVariable], dict[str, list[str]]]:
    """Build staged PMD variables and model-specific lookup keys.

    Only canonical PMD classes present in every stage are combined.
    Ordinary variables are not returned, but their names are checked
    for collisions with the combined PMD names.

    Parameters
    ----------
    models : list[LUMEModel]
        Models in propagation order. Submodels must not expose PMDmodel_index
        variables or its subclasses. All shared PMD outputs within a stage
        must have the same sample count; counts may differ between stages.
    name_overrides : dict[type[PMDVariable], str]
        Validated staged-model name overrides keyed by canonical PMD class.
        Unspecified classes use their default names. These overrides do not
        change the keys used to retrieve values from the individual models.

    Returns
    -------
    staged_pmd_variables : dict[str, PMDVariable]
        Staged-model names mapped to combined PMD variable objects.
    model_specific_keys : dict[str, list[str]]
        Staged-model names mapped to keys in each model's ``supported_variables``,
        in the same order as ``models``.

    Raises
    ------
    ValueError
        If canonical matches are ambiguous, staged-model names are invalid or
        collide, units or dtypes differ across stages, or sample counts are
        negative or inconsistent between shared outputs within a stage.

    Warns
    -----
    UserWarning
        If a PMD class is absent from any stage; that class is excluded.

    Examples
    --------
    Suppose ``source`` and ``transport`` expose concrete subclasses of
    PMDbeta_x under the keys ``source_beta`` and ``transport_beta``, with
    shapes ``(2,)`` and ``(1,)`` and dtype float32. With no name overrides:

    >>> staged_pmd_variables, model_specific_keys = _build_pmd_metadata(
    ...     [source, transport], {}
    ... )
    >>> {name: variable.shape for name, variable in staged_pmd_variables.items()}
    {'pmd:beta_x': (3,), 'pmd:model_index': (3,)}
    >>> model_specific_keys
    {'pmd:beta_x': ['source_beta', 'transport_beta']}
    >>> staged_pmd_variables['pmd:beta_x'].default_value is None
    True
    >>> staged_pmd_variables['pmd:model_index'].default_value.tolist()
    [0, 0, 1]

    Overriding staged-model names changes both returned dictionaries consistently,
    while preserving the model-specific keys:

    >>> overrides = {PMDbeta_x: 'beta', PMDmodel_index: 'stage'}
    >>> staged_pmd_variables, model_specific_keys = _build_pmd_metadata(
    ...     [source, transport], overrides
    ... )
    >>> {name: variable.name for name, variable in staged_pmd_variables.items()}
    {'beta': 'beta', 'stage': 'stage'}
    >>> model_specific_keys
    {'beta': ['source_beta', 'transport_beta']}
    >>> staged_pmd_variables['stage'].default_value.tolist()
    [0, 0, 1]

    If no PMD classes are shared, both dictionaries are empty and no synthetic
    index is created. An empty model list likewise returns ``({}, {})``.
    """
    class_mapping = get_pmd_class_mapping(models)
    ordinary_names = {
        name
        for model in models
        for name, variable in model.supported_variables.items()
        if not isinstance(variable, PMDVariable)
    }
    staged_pmd_variables: dict[str, PMDVariable] = {}
    model_specific_keys: dict[str, list[str]] = {}
    index_name = _resolve_pmd_name(PMDmodel_index, name_overrides)
    stage_lengths: list[int] | None = None
    reference_name = None

    # iterate over each PMD type and its associated keys to get staged PMD variables
    for pmd_type, keys in class_mapping.items():
        name = _resolve_pmd_name(pmd_type, name_overrides)

        # resolve the staged-model name for this PMD type
        if name == index_name:
            raise ValueError(f"Reserved synthetic PMD staged-model name '{name}'.")
        if name in staged_pmd_variables or name in ordinary_names:
            raise ValueError(f"Conflicting PMD staged-model name '{name}'.")

        # collect the variables corresponding to this PMD type from each stage
        variables = [model.supported_variables[key] for model, key in zip(models, keys)]
        lengths = [variable.shape[0] for variable in variables]

        # check that all stage lengths are nonnegative and consistent across stages
        for index, length in enumerate(lengths):
            if length < 0:
                raise ValueError(
                    f"PMD staged-model name '{name}' must have a nonnegative sample count "
                    f"in {_model_label(index, models[index])}."
                )
            if stage_lengths is not None and length != stage_lengths[index]:
                raise ValueError(
                    f"Inconsistent PMD lengths in {_model_label(index, models[index])}: "
                    f"'{reference_name}' has {stage_lengths[index]} samples, "
                    f"but '{name}' has {length}."
                )
        if stage_lengths is None:
            stage_lengths = lengths
            reference_name = name
        staged_pmd_variables[name] = _combine_pmd_variable(name, variables, models)
        model_specific_keys[name] = keys

    # create synthetic PMDmodel_index if stage lengths are defined
    if stage_lengths is not None:
        if index_name in ordinary_names:
            raise ValueError(f"Conflicting PMD staged-model name '{index_name}'.")
        indices = np.repeat(np.arange(len(models), dtype=np.int64), stage_lengths)
        staged_pmd_variables[index_name] = PMDmodel_index(
            name=index_name, shape=indices.shape, default_value=indices
        )
    return staged_pmd_variables, model_specific_keys


def _get_pmd_values(
    models: list[LUMEModel],
    names: list[str],
    staged_pmd_variables: dict[str, PMDVariable],
    model_specific_keys: dict[str, list[str]],
) -> dict[str, Any]:
    """Read model-specific PMD arrays and return them under staged-model names.

    For each requested PMD output, retrieve its mapped key through each
    model's ``get`` method, concatenate the arrays along axis 0 in model
    order, and cast the result to the staged PMD variable's dtype. The
    synthetic PMDmodel_index is returned from its variable's default
    value without fetching it from any submodel.

    Parameters
    ----------
    models : list[LUMEModel]
        Models in propagation order. The caller is responsible for running
        the staged propagation before requesting values with this helper.
    names : list[str]
        Requested staged-model names, including any custom-name overrides.
        Names absent from ``staged_pmd_variables`` are skipped, allowing the caller
        to handle ordinary variables separately.
    staged_pmd_variables : dict[str, PMDVariable]
        Combined PMD variable objects returned by ``_build_pmd_metadata``. These
        provide output dtypes and the synthetic index's default value.
    model_specific_keys : dict[str, list[str]]
        Staged-model names mapped to one submodel lookup key per entry in
        ``models``, in the same order. Keys may differ between submodels
        and from the staged-model name. No entry is needed for the synthetic index.

    Returns
    -------
    values : dict[str, Any]
        Requested PMD staged-model names mapped to concatenated arrays or the
        generated index array. Unrequested PMD outputs and names absent
        from ``staged_pmd_variables`` are not included.

    Examples
    --------
    Suppose evaluated ``source`` and ``transport`` models provide float32
    PMDbeta_x arrays ``[1.0, 2.0]`` and ``[3.0]`` under the model-specific keys
    ``source_beta`` and ``transport_beta``. With custom staged-model names:

    >>> overrides = {PMDbeta_x: 'beta', PMDmodel_index: 'stage'}
    >>> staged_pmd_variables, model_specific_keys = _build_pmd_metadata(
    ...     [source, transport], overrides
    ... )
    >>> model_specific_keys
    {'beta': ['source_beta', 'transport_beta']}
    >>> values = _get_pmd_values(
    ...     [source, transport], ['beta', 'stage', 'source_phase'],
    ...     staged_pmd_variables, model_specific_keys,
    ... )
    >>> {name: value.tolist() for name, value in values.items()}
    {'beta': [1.0, 2.0, 3.0], 'stage': [0, 0, 1]}
    >>> values['beta'].dtype.name, values['stage'].dtype.name
    ('float32', 'int64')

    Here ``source_phase`` is an ordinary variable, so it is left for the
    caller to retrieve. An empty request returns ``{}`` without reading
    any submodel values.
    """
    values = {}
    for name in names:
        if name not in staged_pmd_variables:
            continue
        variable = staged_pmd_variables[name]
        if isinstance(variable, PMDmodel_index):
            values[name] = variable.default_value
        else:
            arrays = [
                model.get([key])[key]
                for model, key in zip(models, model_specific_keys[name])
            ]
            values[name] = np.concatenate(arrays, axis=0).astype(
                variable.dtype, copy=False
            )
    return values


class StagedModel(LUMEModel, InitialParticlesMixIn, FinalParticlesMixIn):
    """
    Composes multiple LUMEModel instances in sequence, passing final particles
    from each stage as initial particles to the next.

    PMD outputs are concatenated by canonical variable class, exposed under
    default names unless overridden by ``pmd_variable_names``.
    A class missing from any stage is excluded with a warning; duplicate matches raise an error.
    Submodels must not expose PMDmodel_index variables; the index is generated
    exclusively by this model.
    """

    def __init__(
        self,
        lume_model_instances: list[LUMEModel],
        *,
        pmd_variable_names: dict[type[PMDVariable], str] | None = None,
    ):
        """
        Initialize the `StagedModel` with a list of LUMEModel instances.

        Parameters
        ----------
        lume_model_instances: list[LUMEModel]
            Ordered list of LUMEModel instances to stage. Their supported
            variables must not include PMDmodel_index or its subclasses.
        pmd_variable_names: dict[type[PMDVariable], str] | None
            Staged-model name overrides keyed by canonical PMD class, including
            PMDmodel_index. Overrides replace default names, not add aliases,
            and never rename submodel variables. Unspecified classes use their
            default names; overrides do not enable unsupported outputs.

        Examples
        --------
        Given compatible ``source`` and ``transport`` models that both expose
        concrete PMDbeta_x variables:

        >>> from lume.staged_model import StagedModel
        >>> from lume.variables.pmd import PMDbeta_x, PMDmodel_index
        >>> staged = StagedModel(
        ...     [source, transport],
        ...     pmd_variable_names={PMDbeta_x: "beta", PMDmodel_index: "stage"},
        ... )
        >>> values = staged.get(["beta", "stage"])
        """
        super().__init__()

        self._is_init = (
            False  # Flag to indicate if the model has been run start to end once
        )
        self.validate_lume_model_instances(lume_model_instances)
        self.lume_model_instances = lume_model_instances
        self._pmd_variable_names = _validate_pmd_variable_names(pmd_variable_names)
        _build_pmd_metadata(self.lume_model_instances, self._pmd_variable_names)

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
        if not models:
            raise ValueError("StagedModel requires at least one model.")
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
            for name, var in model.supported_variables.items():
                if isinstance(var, PMDVariable):
                    continue

                if name in seen:
                    raise ValueError(
                        f"Variable '{name}' is defined in both model {seen[name]} and model {i}."
                    )
                seen[name] = i

    @property
    def supported_variables(self) -> dict[str, Variable]:
        variables = {
            name: var
            for model in self.lume_model_instances
            for name, var in model.supported_variables.items()
            if not isinstance(var, PMDVariable)
        }
        staged_pmd_variables, _ = _build_pmd_metadata(
            self.lume_model_instances, self._pmd_variable_names
        )
        variables.update(staged_pmd_variables)
        return variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.coerce_evaluated_once()

        staged_pmd_variables, model_specific_keys = _build_pmd_metadata(
            self.lume_model_instances, self._pmd_variable_names
        )
        values = _get_pmd_values(
            self.lume_model_instances, names, staged_pmd_variables, model_specific_keys
        )

        for model in self.lume_model_instances:
            model_names = [
                n
                for n in names
                if n in model.supported_variables
                and not isinstance(model.supported_variables[n], PMDVariable)
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
