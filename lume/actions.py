from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, Union

from pydantic import field_validator

from lume.exceptions import ReadOnlyError
from lume.model import LUMEModel
from lume.variables import Variable

SimulatorT = TypeVar("SimulatorT")

###############################################################################
# Action definitions


class Action(ABC, Generic[SimulatorT]):
    """
    Parent class for testing if something is an Action.

    Do not subclass directly. Use ``ReadOnlyActionMixin`` or
    ``WritableActionMixin`` mixed into a ``Variable`` subclass.
    """


class ReadOnlyActionMixin(Action[SimulatorT], Generic[SimulatorT]):
    """Mixin for read-only variable-actions.

    Mix into a ``Variable`` subclass. The variable must be constructed with
    ``read_only=True``; the field validator enforces this at construction time.
    Subclasses must implement ``_get``.
    """

    @field_validator("read_only", mode="after")
    @classmethod
    def _check_read_only(cls, v: bool) -> bool:
        if not v:
            raise ReadOnlyError(f"{cls.__name__} requires read_only=True")
        return v

    @abstractmethod
    def _get(self, simulator: SimulatorT) -> Any:
        """Return the current value from ``simulator``. User-implemented version, not ``get``.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to read from.

        Returns
        -------
        Any
            The current value of this variable in the simulator.
        """
        ...


class WritableActionMixin(Action[SimulatorT], Generic[SimulatorT]):
    """Mixin for writable variable-actions.

    Mix into a ``Variable`` subclass. Subclasses must implement ``_get`` and
    ``_set``.

    Calling ``set`` on a variable with ``read_only=True`` raises
    ``ReadOnlyError`` at runtime.
    """

    @abstractmethod
    def _get(self, simulator: SimulatorT) -> Any:
        """Return the current value from ``simulator``. User implemented version.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to read from.

        Returns
        -------
        Any
            The current value of this variable in the simulator.
        """
        ...

    @abstractmethod
    def _set(self, simulator: SimulatorT, value: Any) -> None:
        """Write ``value`` to ``simulator``. User implemented version.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to write to.
        value : Any
            The value to assign to this variable in the simulator.
        """
        ...


class AffineTransformMixin(Action[SimulatorT], Generic[SimulatorT]):
    """
    Mixin for variable-actions that apply an affine transformation to the value on set 
    and undo the transformation on get. Defaults to the identity transformation (scale=1, offset=0).

    Mix into a ``Action`` subclass.
    """
    scale: Optional[float] = 1.0
    offset: Optional[float] = 0.0

    def _get(self, simulator: SimulatorT) -> Any:
        """Return the current value from ``simulator`` after undoing the affine transformation.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to read from.

        Returns
        -------
        Any
            The current value of this variable in the simulator after undoing the affine transformation.
        """
        raw_value = super()._get(simulator)
        return (raw_value - self.offset) / self.scale
    
    def _set(self, simulator: SimulatorT, value: Any) -> None:
        """Write ``value`` to ``simulator`` after applying the affine transformation.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to write to.
        value : Any
            The value to assign to this variable in the simulator after applying the affine transformation.
        """
        transformed_value = value * self.scale + self.offset

        try:
            super()._set(simulator, transformed_value)
        except AttributeError:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement _set, cannot apply affine transformation"
            )


###############################################################################
# Type hint stubs: these classes are used only to get correct type hints for
# the action variable objects in ActionModel below and are not meant for use
# outside of this module.


class ReadOnlyActionVariable(
    ReadOnlyActionMixin[SimulatorT], Variable, Generic[SimulatorT]
):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This is a type hinting stub")

    def _get(self, simulator: SimulatorT) -> Any:
        raise NotImplementedError("This is a type hinting stub")


class WritableActionVariable(
    WritableActionMixin[SimulatorT], Variable, Generic[SimulatorT]
):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This is a type hinting stub")

    def _get(self, simulator: SimulatorT) -> Any:
        raise NotImplementedError("This is a type hinting stub")

    def _set(self, simulator: SimulatorT, value: Any) -> None:
        raise NotImplementedError("This is a type hinting stub")


# Type which will have the correct interface from user-provided variables
ActionVariable = Union[
    ReadOnlyActionVariable[SimulatorT], WritableActionVariable[SimulatorT]
]


###############################################################################
# Model definition


class ActionModel(LUMEModel, Generic[SimulatorT]):
    """
    LUMEModel backed by a collection of action variables.

    Each entry in ``action_variables`` must be an ``Action`` instance (i.e. a
    ``Variable`` subclass that also mixes in ``ReadOnlyActionMixin`` or
    ``WritableActionMixin``). The model delegates ``get`` and ``set`` calls to
    the corresponding action variable.

    Parameters
    ----------
    simulator : SimulatorT
        The simulator object passed through to each action variable's
        ``get`` / ``set`` implementation.
    action_variables : list[ActionVariable[SimulatorT]]
        The action variables managed by this model.

    Raises
    ------
    ValueError
        If any entry in ``action_variables`` is not an ``Action`` instance.
    """

    def __init__(
        self,
        simulator: SimulatorT,
        action_variables: list[ActionVariable[SimulatorT]],
    ) -> None:
        self.simulator = simulator
        self._action_variable_by_name: dict[str, ActionVariable[SimulatorT]] = {}
        for av in action_variables:
            self.register_action_variable(av)

    @property
    def supported_variables(self) -> dict[str, ActionVariable[SimulatorT]]:
        return dict(self._action_variable_by_name)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {
            name: self._action_variable_by_name[name]._get(self.simulator)
            for name in names
        }

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            av = self._action_variable_by_name[name]
            if isinstance(av, WritableActionMixin):
                av._set(self.simulator, value)

    def reset(self) -> None:
        self._set(
            {
                av.name: av.default_value
                for av in self._action_variable_by_name.values()
                if isinstance(av, WritableActionMixin) and hasattr(av, "default_value")
            }
        )

    def register_action_variable(
        self, action_variable: ActionVariable[SimulatorT]
    ) -> None:
        """
        Add an action variable to the model, replacing any with the same name.

        Parameters
        ----------
        action_variable : ActionVariable[SimulatorT]
            The action variable to register. Ie a class that inherits from both a `Variable`
            and either `ReadOnlyActionMixin` or `WritableActionMixin`.

        Raises
        ------
        ValueError
            If ``action_variable`` is not an ``Action`` instance.
        """
        if not isinstance(action_variable, Action):
            raise ValueError(
                f"Expected an Action instance, got {type(action_variable).__name__!r}"
            )
        self._action_variable_by_name[action_variable.name] = action_variable

    def unregister_action_variable(self, name: str) -> ActionVariable[SimulatorT]:
        """
        Remove an action variable from the model by name.

        Parameters
        ----------
        name : str
            Name of the action variable to remove.

        Returns
        -------
        ActionVariable
            The removed action variable.

        Raises
        ------
        KeyError
            If no action variable with the given name is registered.
        """
        if name not in self._action_variable_by_name:
            raise KeyError(f"No action variable named '{name}' is registered")
        return self._action_variable_by_name.pop(name)
