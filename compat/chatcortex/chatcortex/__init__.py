"""chatcortex was renamed to frontierkit.

This package holds no code of its own. Importing it loads ``frontierkit``
and warns that the name has changed; ``chatcortex.anything`` is the very
same module object as ``frontierkit.anything``, so code written against
the old name keeps working, including ``isinstance`` checks across the
two names.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import frontierkit
from frontierkit import *  # noqa: F401,F403
from frontierkit import __all__, __version__  # noqa: F401

_OLD = "chatcortex"
_NEW = "frontierkit"

warnings.warn(
    "chatcortex has been renamed to frontierkit: run `pip install frontierkit` "
    "and replace `chatcortex` with `frontierkit` in your imports. This "
    "compatibility package will get no further releases.",
    DeprecationWarning,
    stacklevel=2,
)


class _Renamed(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve ``chatcortex.x`` with the module ``frontierkit.x``."""

    def __init__(self):
        self._specs = {}

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        target_name = _NEW + fullname[len(_OLD):]
        if importlib.util.find_spec(target_name) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = importlib.import_module(_NEW + spec.name[len(_OLD):])
        # The import system is about to stamp the old-name spec on the
        # module; remember the real one so exec_module can put it back.
        self._specs[module.__name__] = module.__spec__
        return module

    def exec_module(self, module):
        """Nothing to run: the module was already loaded under its new name."""
        real = self._specs.pop(module.__name__, None)
        if real is not None:
            module.__spec__ = real


if not any(isinstance(finder, _Renamed) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Renamed())

# Submodules that frontierkit has already imported are reachable at once.
for _name, _module in list(sys.modules.items()):
    if _name.startswith(_NEW + "."):
        sys.modules.setdefault(_OLD + _name[len(_NEW):], _module)


def __getattr__(name):
    return getattr(frontierkit, name)
