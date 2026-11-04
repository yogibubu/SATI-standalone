"""Explicit import compatibility for relocated MATRIX modules.

Aliases expose the canonical owner's objects without copying scientific code.
Registration does not import the target or its optional dependencies.
"""
from __future__ import annotations

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys


class _AliasLoader(Loader):
    def __init__(self, name, target, package):
        self.name, self.target, self.package = name, target, package

    def create_module(self, spec):
        return None

    def get_filename(self, fullname):
        return find_spec(self.target).origin

    def exec_module(self, module):
        owner = import_module(self.target)
        module.__doc__ = owner.__doc__
        module.__getattr__ = lambda name: getattr(owner, name)
        module.__dir__ = lambda: sorted(set(vars(module)) | set(dir(owner)))
        module.__all__ = getattr(owner, "__all__", [
            name for name in vars(owner) if not name.startswith("_")
        ])
        if self.package:
            module.__path__ = []

    def get_code(self, fullname):
        # Support historical ``python -m`` entry points through the same owner.
        source = "import runpy\nrunpy.run_module(%r, run_name='__main__')\n" % self.target
        return compile(source, "<MATRIX compatibility: %s>" % fullname, "exec")

    def get_resource_reader(self, fullname):
        owner = import_module(self.target)
        loader = owner.__spec__.loader
        reader = getattr(loader, "get_resource_reader", None)
        return reader(self.target) if reader else None


class _AliasFinder(MetaPathFinder):
    def __init__(self):
        self.aliases = {}

    def find_spec(self, fullname, path=None, target=None):
        entry = self.aliases.get(fullname)
        if entry is None:
            return None
        destination, package = entry
        return spec_from_loader(
            fullname, _AliasLoader(fullname, destination, package), is_package=package
        )


_finder = _AliasFinder()


def register_module_aliases(aliases):
    """Register exact old names as ``name: (canonical_module, is_package)``."""
    for name, entry in aliases.items():
        previous = _finder.aliases.get(name)
        if previous is not None and previous != entry:
            raise ValueError(f"conflicting module alias: {name}")
        _finder.aliases[name] = entry
    if _finder not in sys.meta_path:
        sys.meta_path.append(_finder)
