from abc import ABC, abstractmethod


def _module_base(module_name):
    """Top-level package name from a dotted module string.

    Accepts a module object as well, so the error path never crashes on a
    caller that passes the imported module instead of its name.
    """
    name = getattr(module_name, "__name__", module_name)
    return str(name).split(".")[0]


class ModuleImportError(Exception):
    def __init__(self, module_name, module_version):
        self.module_name = module_name
        module_base = _module_base(module_name)
        self.message = (
            f"Unable to import module {module_name}. Object was serialized with {module_version}. "
            f"Is a compatible version of {module_base} installed?"
        )
        super().__init__(self.message)


class ClassInitError(Exception):
    def __init__(self, class_name, module_name, module_version):
        self.module_name = module_name
        self.class_name = class_name
        module_base = _module_base(module_name)
        self.message = (
            f"Unable to get {class_name} from {module_name}. Object was serialized with {module_version}. "
            f"Is a compatible version of {module_base} installed?"
        )
        super().__init__(self.message)


class SerializerBase(ABC):
    """Base class for serializers."""

    @abstractmethod
    def serialize(self, filename, object):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls, filename):
        raise NotImplementedError
