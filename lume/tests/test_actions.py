import pytest

from lume.actions import ActionModel, ReadOnlyActionMixin, WritableActionMixin
from lume.exceptions import ReadOnlyError
from lume.variables.scalar import ScalarVariable


class MockSim:
    def __init__(self):
        self.values = {}


class MockReadOnlyVar(ReadOnlyActionMixin, ScalarVariable):
    def _get(self, simulator: MockSim) -> float:
        return simulator.values.get(self.name, 0.0)


class MockWritableVar(WritableActionMixin, ScalarVariable):
    def _get(self, simulator: MockSim) -> float:
        return simulator.values.get(self.name, 0.0)

    def _set(self, simulator: MockSim, value: float) -> None:
        simulator.values[self.name] = value


@pytest.fixture
def sim():
    return MockSim()


@pytest.fixture
def readonly_var():
    return MockReadOnlyVar(name="x", read_only=True)


@pytest.fixture
def writable_var():
    return MockWritableVar(name="y")


class TestReadOnlyActionMixin:
    def test_construction_requires_read_only(self):
        with pytest.raises(ReadOnlyError):
            MockReadOnlyVar(name="x", read_only=False)

    def test_get(self, sim, readonly_var):
        sim.values["x"] = 3.14
        assert readonly_var._get(sim) == 3.14

    def test_no_set_method(self, readonly_var):
        assert not hasattr(readonly_var, "set")


class TestWritableActionMixin:
    def test_construction_writable(self, writable_var):
        assert writable_var.read_only is False

    def test_get(self, sim, writable_var):
        sim.values["y"] = 42.0
        assert writable_var._get(sim) == 42.0

    def test_set(self, sim, writable_var):
        writable_var._set(sim, 7.0)
        assert sim.values["y"] == 7.0


@pytest.fixture
def action_model():
    s = MockSim()
    avs = [
        MockWritableVar(name="x", default_value=1.0),
        MockReadOnlyVar(name="y", read_only=True),
    ]
    return ActionModel(s, avs)


class TestActionModel:
    def test_supported_variables(self, action_model):
        assert set(action_model.supported_variables) == {"x", "y"}

    def test_get(self, action_model):
        action_model.simulator.values["y"] = 9.9
        assert action_model.get_value("y") == 9.9

    def test_set(self, action_model):
        action_model.set({"x": 5.0})
        assert action_model.simulator.values["x"] == 5.0

    def test_set_read_only_raises(self, action_model):
        with pytest.raises(ReadOnlyError):
            action_model.set({"y": 1.0})

    def test_reset(self, action_model):
        action_model.simulator.values["x"] = 99.0
        action_model.reset()
        assert action_model.simulator.values["x"] == 1.0

    def test_register_replaces_existing(self, action_model):
        new_x = MockWritableVar(name="x", default_value=42.0)
        action_model.register_action_variable(new_x)
        assert action_model._action_variable_by_name["x"] is new_x
        assert len(action_model.supported_variables) == 2

    def test_register_rejects_non_action(self, action_model):
        with pytest.raises(ValueError):
            action_model.register_action_variable(ScalarVariable(name="plain"))

    def test_unregister(self, action_model):
        removed = action_model.unregister_action_variable("x")
        assert "x" not in action_model.supported_variables
        assert removed.name == "x"

    def test_unregister_missing_raises(self, action_model):
        with pytest.raises(KeyError):
            action_model.unregister_action_variable("nonexistent")
