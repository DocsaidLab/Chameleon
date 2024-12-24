import fnmatch
import inspect
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Type, Union

from rich.console import Console
from rich.table import Table


def build_from_cfg(cfg: dict, registry: "Registry") -> Any:

    if not isinstance(cfg, dict):
        raise TypeError(f'`cfg` should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if 'name' not in cfg:
        raise KeyError('`cfg` must contain the key "name"')

    if not isinstance(registry, Registry):
        raise TypeError(f'registry must be a chameleon.Registry object, but got {type(registry)}')

    kwargs = cfg.copy()
    name = kwargs.pop('name')
    obj_cls = registry.get(name)
    is_model_builder = registry.is_model_builder(name)

    if inspect.isclass(obj_cls) or is_model_builder:
        obj = obj_cls(**kwargs)
    else:
        obj = obj_cls
    return obj


class Registry:

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._type_dict: Dict[str, Type] = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key: str) -> Optional[Type]:
        if not isinstance(key, str):
            raise TypeError(f'key must be a str, but got {type(key)}')

        obj_cls = self.module_dict.get(key, None)

        if obj_cls is None:
            raise KeyError(f'{key} is not in the {self.name} registry')

        return obj_cls

    def is_model_builder(self, key: str) -> bool:
        if not isinstance(key, str):
            raise TypeError(f'key must be a str, but got {type(key)}')

        is_model_builder = self._type_dict.get(key, None)

        if is_model_builder is None:
            raise KeyError(f'{key} is not in the {self.name} registry')

        return is_model_builder

    def build(self, cfg: dict) -> Any:
        return build_from_cfg(cfg, registry=self)

    def _register_module(
        self,
        module: Type,
        module_name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
        is_model_builder: bool = False,
    ) -> None:
        if not callable(module):
            raise TypeError(f'module must be a callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} at {existed_module.__module__}')
            self._module_dict[name] = module
            self._type_dict[name] = is_model_builder

    def register_module(
        self,
        name: str = None,
        force: bool = False,
        module: Optional[Type] = None,
        is_model_builder: bool = False,
    ) -> Union[type, Callable]:

        if not (name is None or isinstance(name, str)):
            raise TypeError(f'`name` must be a str or None, but got {type(name)}')

        if not isinstance(force, bool):
            raise TypeError(f'`force` must be a bool, but got {type(force)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force, is_model_builder=is_model_builder)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force, is_model_builder=is_model_builder)
            return module

        return _register

    def list_module(self, filter: Optional[str] = None) -> List[str]:
        list_modules = list(self._module_dict.keys())
        if filter is not None:
            list_modules = fnmatch.filter(list_modules, filter)
        return list_modules
