from abc import ABC, abstractmethod
from typing import Any

from lume.variables import Variable
from lume.exceptions import ReadOnlyError


class LUMEModel(ABC):
    """
    Abstract base class for creating virtual accelerator models and digital twins.
    Simulation specific implementations should inherit from this class and implement
    the abstract methods. Virtual accelerator models that require multiple simulation
    types can also inherit from this class and utilize composition to manage multiple
    simulators.

    Attributes
    ----------
    supported_variables: dict[str, Variable]
        A dictionary of Variable instances that the model supports.

    Methods
    -------
    get(names: list[str] | str) -> dict[str, Any] | Any:
        Get measurements/state from the simulator.

    set(values: dict[str, Any]) -> None:
        Set control parameters of the simulator.

    reset() -> None:
        Reset the simulator to its initial state.
    """

    def get(self, names: list[str]) -> dict[str, Any]:
        """
        Get measurements/state from the simulator. Should do the following:
        - validate input names against supported_variables
        - return cached measurements/state for the requested names

        Parameters
        ----------
        names: list[str]
            List of variable names to get from the simulator.

        Returns
        -------
        dict[str, Any]
            Dictionary of variable names and their corresponding values.

        """
        if not isinstance(names, list):
            raise TypeError("names must be a list of strings.")

        # Validate input names
        for name in names:
            if name not in self.supported_variables:
                raise ValueError(f"Variable '{name}' is not supported by the model.")

        outputs = self._get(names)

        # Validate outputs
        svars = self.supported_variables
        for name in names:
            try:
                svars[name].validate_value(outputs[name])
            except (ValueError, TypeError) as exc:
                raise type(exc)(
                    f"Validation failed for variable '{name}': {exc}"
                ) from exc

        return outputs

    def get_value(self, name: str) -> Any:
        """
        Get the value of a single variable from the simulator.

        Parameters
        ----------
        name: str
            The name of the variable to get from the simulator.

        Returns
        -------
        Any
            The value of the requested variable.
        """
        return self.get([name])[name]

    @abstractmethod
    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Internal method to get measurements/state from the simulator.
        Should be implemented by subclasses.

        Parameters
        ----------
        names: list[str]
            List of variable names to get from the simulator.

        Returns
        -------
        dict[str, Any]
            Dictionary of variable names and their corresponding values.
        """
        raise NotImplementedError(
            "private _get method must be implemented by subclasses."
        )

    def set(self, values: dict[str, Any]) -> None:
        """
        Set control parameters of the simulator. Should do the following:
        - validate input values against supported_variables
        - set the control parameters of the simulator
        - run the simulator
        - update cached measurements/state

        Parameters
        ----------
        values: dict[str, Any]
            Dictionary of variable names and their corresponding values to set in the simulator.

        Returns
        -------
        None
        """
        # Validate input values
        for name in values.keys():
            if name not in self.supported_variables:
                raise ValueError(f"Variable '{name}' is not supported by the model.")
            else:
                variable = self.supported_variables[name]
                if not isinstance(variable, Variable):
                    raise ValueError(
                        f"Variable '{name}' is not a valid Variable instance."
                    )

                if variable.read_only:
                    raise ReadOnlyError(
                        f"Variable '{name}' is read-only. Cannot be set."
                    )
                try:
                    variable.validate_value(values[name])
                except (ValueError, TypeError) as exc:
                    raise type(exc)(
                        f"Validation failed for variable '{name}': {exc}"
                    ) from exc

        # Set the control parameters of the simulator
        self._set(values)

    def set_value(self, name: str, value: Any) -> None:
        """
        Set the value of a single variable in the simulator.

        Parameters
        ----------
        name: str
            The name of the variable to set in the simulator.
        value: Any
            The value to set for the specified variable.

        Returns
        -------
        None
        """
        self.set({name: value})

    @abstractmethod
    def _set(self, values: dict[str, Any]) -> None:
        """
        Internal method to set control parameters of the simulator and run the simulation.
        Should be implemented by subclasses.

        Parameters
        ----------
        values: dict[str, Any]
            Dictionary of variable names and their corresponding values to set in the simulator.

        Returns
        -------
        None
        """
        raise NotImplementedError(
            "private _set method must be implemented by subclasses."
        )

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the simulator to its initial state.
        """
        raise NotImplementedError("reset method must be implemented by subclasses.")

    @property
    @abstractmethod
    def supported_variables(self) -> dict[str, Variable]:
        """
        Return a dict of variables supported by the model. It is expected that
        keys of this dict are valid keys for get() and set() methods.

        Returns
        -------
        dict[str, Variable]
            A dictionary of Variable instances that the model supports.
        """
        raise NotImplementedError(
            "supported_variables property must be implemented by subclasses."
        )
