# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Deprecations of braincell's public names, parameters and values.

Each deprecation is a module-level :class:`Deprecation` constant beside the
code it deprecates. The constant carries the release that announced it, the
release that removes it and what to use instead; the warning message, the
registry the expiry test reads and the changelog check all derive from it.

A warning is attributed to the line that used the deprecated API, found by
walking the stack rather than by a fixed ``stacklevel``; braincell's own uses
do not warn. The rules and their reasons are in the "Deprecation" section of
the developer conventions.

Run ``python -m braincell._deprecation MAJOR.MINOR`` before a release to list
the deprecations due by that release.

A name that a release removed is not a deprecation: :func:`removed_attributes`
serves a module ``__getattr__`` whose error names the replacement.
"""

from __future__ import annotations

import argparse
import dataclasses
import dis
import importlib
import os
import pkgutil
import re
import sys
import warnings
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "Deprecation",
    "deprecated_attributes",
    "expired",
    "import_all",
    "main",
    "registered",
    "removed_attributes",
    "user_frame",
    "warn_at_user_frame",
]

#: The braincell package directory. Code in files under it is braincell's own,
#: except its ``*_test.py`` files, which call the API as users do.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_DATACLASSES_FILE = os.path.abspath(dataclasses.__file__)
_MINOR = re.compile(r"(\d+)\.(\d+)")
_CATEGORIES = (DeprecationWarning, FutureWarning)
_IMPORT_FROM = dis.opmap["IMPORT_FROM"]
_PLACES = ("caller", "user")

#: Every :class:`Deprecation` constructed so far, by name.
_REGISTRY: dict[str, Deprecation] = {}
_METACLASS_CALLS: frozenset | None = None


def _minor(version: str, field: str = "version") -> tuple[int, int]:
    """Parse ``"MAJOR.MINOR"`` into a tuple of two ints."""
    match = _MINOR.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        raise ValueError(f"{field} must be a 'MAJOR.MINOR' release such as '0.4'; got {version!r}.")
    return int(match[1]), int(match[2])


def _release_minor() -> tuple[int, int]:
    """``(major, minor)`` of the installed ``braincell.__version__``."""
    from braincell._version import __version__

    major, minor = re.match(r"(\d+)\.(\d+)", __version__).groups()
    return int(major), int(minor)


def _metaclass_calls() -> frozenset:
    """Code objects of the ``__call__`` methods along brainstate's node metaclass MRO.

    ``GraphNodeMeta.__call__`` runs ``__init__`` on the caller's behalf, so
    its frame sits between the constructing line and the ``__init__``.
    brainstate is imported on first use, so this module imports nothing but
    the standard library.
    """
    global _METACLASS_CALLS
    if _METACLASS_CALLS is None:
        import brainstate

        metaclass = type(brainstate.graph.Node)
        codes = (getattr(vars(klass).get("__call__"), "__code__", None) for klass in metaclass.__mro__)
        _METACLASS_CALLS = frozenset(code for code in codes if code is not None)
    return _METACLASS_CALLS


def _is_internal(code) -> bool:
    """Whether ``code`` is braincell's own (non-test) code."""
    filename = code.co_filename
    return filename.startswith(_PACKAGE_DIR) and not filename.endswith("_test.py")


def _is_trampoline(code) -> bool:
    """Whether ``code`` calls the deprecated code on someone else's behalf.

    These frames are never the caller: brainstate's node metaclass
    ``__call__``, a dataclass-generated ``__init__`` (which calls
    ``__post_init__``), the standard library's ``dataclasses`` module
    (``replace`` rebuilds an object) and the import machinery.
    """
    filename = code.co_filename
    if filename == "<string>" and code.co_name == "__init__":
        return True
    if filename == _DATACLASSES_FILE or filename.startswith("<frozen importlib"):
        return True
    return code in _metaclass_calls()


def user_frame(frame, skip=frozenset()):
    """Return the nearest frame outside braincell's own code, and its distance.

    Frames in the braincell package are skipped, except its ``*_test.py``
    files, which call the API as users do; so are the frames that call code
    on someone else's behalf (brainstate's node metaclass ``__call__``,
    dataclass-generated ``__init__``, :mod:`dataclasses`, the import
    machinery).

    Parameters
    ----------
    frame : frame or None
        Frame to start from; it is itself a candidate.
    skip : collection of code objects, optional
        Further frames to skip.

    Returns
    -------
    tuple
        ``(frame, hops)``: the first user frame (``None`` if there is none)
        and the number of frames skipped to reach it.
    """
    hops = 0
    while frame is not None:
        code = frame.f_code
        if code not in skip and not _is_internal(code) and not _is_trampoline(code):
            return frame, hops
        frame = frame.f_back
        hops += 1
    return None, hops


def _emit(message: str, category: type[Warning], frame, stacklevel: int) -> None:
    """Issue the warning as ``warnings.warn`` would from ``frame``.

    Without a frame, it falls back to ``warnings.warn``, with ``stacklevel``
    counted as if the caller of ``_emit`` had called it.
    """
    if frame is None:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
        return
    module_globals = frame.f_globals
    # No ``module_globals``: it only feeds linecache, and costs ~6 us a call.
    warnings.warn_explicit(
        message,
        category,
        frame.f_code.co_filename,
        frame.f_lineno,
        module=module_globals.get("__name__", "<string>"),
        registry=module_globals.setdefault("__warningregistry__", {}),
    )


def warn_at_user_frame(message: str, category: type[Warning] = FutureWarning, *, skip=frozenset()) -> None:
    """Issue ``message`` attributed to the nearest user frame (see :func:`user_frame`).

    The caller's own frame counts, so a call from a braincell ``*_test.py``
    is attributed to that line. Where frames are unavailable (``sys._getframe``
    is CPython-specific) the warning points at the caller. ``skip`` is passed
    to :func:`user_frame`.
    """
    getframe = getattr(sys, "_getframe", None)
    frame, _ = user_frame(getframe(1) if getframe is not None else None, skip)
    _emit(message, category, frame, stacklevel=2)


@dataclasses.dataclass(frozen=True)
class Deprecation:
    """One deprecated name, parameter or value, and the warning its use gives.

    Constructing it registers it (see :func:`registered`); define it as a
    module-level constant beside the code it deprecates, and delete it with
    that code.

    Parameters
    ----------
    name : str
        What the user wrote, as they would search for it, e.g.
        ``"braincell.mech.StateProbe"`` or ``"Network(seed=...)"``. The
        changelog's Deprecations entry quotes it verbatim.
    since : str
        The ``"MAJOR.MINOR"`` release whose changelog first announces the
        deprecation with its removal release.
    removed_in : str
        The ``"MAJOR.MINOR"`` release that removes the old form (or changes
        its behaviour); later than ``since``.
    replacement : str or None
        What to use instead. ``None`` only with a ``note`` saying what to do.
    category : {DeprecationWarning, FutureWarning}, optional
        ``DeprecationWarning`` when a name, parameter or value is removed or
        renamed; ``FutureWarning`` when the same call will give a different
        result or raise.
    note : str, optional
        Appended to the message after one space.

    Raises
    ------
    ValueError
        If a version is not ``"MAJOR.MINOR"``, ``removed_in`` is not later
        than ``since``, there is neither ``replacement`` nor ``note``, or
        ``name`` is already registered with different fields.
    TypeError
        If ``replacement`` is missing, ``category`` is neither warning class,
        or ``name`` is not a non-empty string.

    Examples
    --------
    A module-level constant, warned from the deprecated function's body:

    .. code-block:: python

        _OLD = Deprecation("braincell.old", since="0.3", removed_in="0.4", replacement="braincell.new")

        def old(x):
            _OLD.warn()
            return new(x)
    """

    name: str
    _: dataclasses.KW_ONLY
    since: str
    removed_in: str
    replacement: str | None
    category: type[Warning] = DeprecationWarning
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"Deprecation name must be a non-empty string; got {self.name!r}.")
        if _minor(self.removed_in, "removed_in") <= _minor(self.since, "since"):
            raise ValueError(
                f"{self.name}: removed_in={self.removed_in!r} must be a later release than since={self.since!r}."
            )
        if self.category not in _CATEGORIES:
            raise TypeError(
                f"{self.name}: category must be DeprecationWarning or FutureWarning; got {self.category!r}."
            )
        if self.replacement is None and not self.note:
            raise ValueError(f"{self.name}: without a replacement, note= must say what to do instead.")
        previous = _REGISTRY.setdefault(self.name, self)
        if previous != self:
            raise ValueError(f"{self.name!r} is already registered with different fields: {previous!r}.")

    @property
    def message(self) -> str:
        """The standard warning text."""
        text = (
            f"{self.name} is deprecated since braincell {self.since} "
            f"and will be removed in braincell {self.removed_in}."
        )
        if self.replacement is not None:
            text += f" Use {self.replacement} instead."
        if self.note:
            text += f" {self.note}"
        return text

    def warn(self, message: str | None = None, *, at: str = "caller") -> None:
        """Warn about one use of the deprecated API; every call warns.

        Parameters
        ----------
        message : str, optional
            Text to issue instead of :attr:`message`, for a behaviour change
            that needs the details of the call. It must name
            ``"braincell <removed_in>"``.
        at : {'caller', 'user'}, optional
            ``"caller"``: call ``warn()`` directly in the body of the
            deprecated function, method, ``__init__``, ``__post_init__`` or
            module ``__getattr__``; the warning goes to its caller. When that
            caller is braincell's own code, nothing is issued.
            ``"user"``: the warning goes to the nearest frame outside
            braincell, for a check that runs deep inside braincell.
            Both skip the frames that call code on someone else's behalf
            (see :func:`user_frame`).

        Raises
        ------
        ValueError
            If ``at`` is neither place, or ``message`` does not name the
            removal release.
        """
        if at not in _PLACES:
            raise ValueError(f"at must be 'caller' or 'user'; got {at!r}.")
        if message is None:
            message = self.message
        elif f"braincell {self.removed_in}" not in message:
            raise ValueError(f"{self.name}: a custom message must name 'braincell {self.removed_in}'; got {message!r}.")
        getframe = getattr(sys, "_getframe", None)
        frame = getframe(1) if getframe is not None else None
        if frame is None:
            # Frames unavailable: the caller of the deprecated callable.
            warnings.warn(message, self.category, stacklevel=3)
            return
        if at == "user":
            frame, _ = user_frame(frame)
        else:
            frame = frame.f_back
            while frame is not None and _is_trampoline(frame.f_code):
                frame = frame.f_back
            if frame is not None and _is_internal(frame.f_code):
                return  # braincell's own use: nothing for the user to change
        _emit(message, self.category, frame, stacklevel=3)


def _is_fromlist_probe(frame) -> bool:
    """Whether ``frame`` is importlib's ``_handle_fromlist``.

    ``from M import X`` on a package looks ``X`` up twice: first
    ``_handle_fromlist`` (through ``hasattr``), then ``IMPORT_FROM`` in the
    importing frame.
    """
    code = getattr(frame, "f_code", None)
    return code is not None and code.co_name == "_handle_fromlist" and code.co_filename.startswith("<frozen importlib")


def deprecated_attributes(module: str, attributes: Mapping[str, tuple[Deprecation, Any]]) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` that serves deprecated attributes.

    Each access warns once at the accessing line (``from M import X`` too).
    Keep these names out of the module's ``__all__``.

    Parameters
    ----------
    module : str
        The module's ``__name__``, for the ``AttributeError`` of other names.
    attributes : Mapping
        ``{name: (deprecation, value)}``.

    Returns
    -------
    callable
        Assign it to the module's ``__getattr__``.
    """
    attributes = dict(attributes)
    for name, entry in attributes.items():
        if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], Deprecation)):
            raise TypeError(f"{module}.{name}: expected a (Deprecation, value) pair; got {entry!r}.")

    def __getattr__(name: str) -> Any:
        try:
            deprecation, value = attributes[name]
        except KeyError:
            raise AttributeError(f"module {module!r} has no attribute {name!r}") from None
        getframe = getattr(sys, "_getframe", None)
        if getframe is None or not _is_fromlist_probe(getframe(1)):
            deprecation.warn()
        return value

    return __getattr__


def _is_import_from(frame) -> bool:
    """Whether ``frame`` is executing ``IMPORT_FROM``, the lookup of ``from M import X``."""
    try:
        return frame.f_code.co_code[frame.f_lasti] == _IMPORT_FROM
    except (AttributeError, IndexError, TypeError):
        return False


def removed_attributes(module: str, attributes: Mapping[str, str]) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` for names that a release removed.

    ``M.X`` raises ``AttributeError`` with ``attributes[X]`` as its message;
    ``from M import X`` raises ``ImportError`` with the same message (Python
    would otherwise replace the ``AttributeError`` of a module ``__getattr__``
    with a bare "cannot import name"). ``hasattr(M, X)`` is ``False``. These
    names are not deprecations and are not registered.

    Parameters
    ----------
    module : str
        The module's ``__name__``, for the ``AttributeError`` of other names.
    attributes : Mapping
        ``{name: message}``; each message names the release that removed the
        name and what to use instead.

    Returns
    -------
    callable
        Assign it to the module's ``__getattr__``.
    """
    attributes = dict(attributes)
    for name, message in attributes.items():
        if not (isinstance(message, str) and message):
            raise TypeError(f"{module}.{name}: expected a non-empty message; got {message!r}.")

    def __getattr__(name: str) -> Any:
        try:
            message = attributes[name]
        except KeyError:
            raise AttributeError(f"module {module!r} has no attribute {name!r}") from None
        getframe = getattr(sys, "_getframe", None)
        if getframe is not None and _is_import_from(getframe(1)):
            raise ImportError(message, name=module)
        raise AttributeError(message)

    return __getattr__


def registered() -> tuple[Deprecation, ...]:
    """Every deprecation registered by the modules imported so far.

    Call :func:`import_all` first for the complete list.
    """
    return tuple(_REGISTRY.values())


def expired(version: tuple[int, int] | None = None) -> list[Deprecation]:
    """Registered deprecations whose ``removed_in`` is not later than ``version``.

    Parameters
    ----------
    version : tuple of int, optional
        ``(major, minor)``; defaults to that of ``braincell.__version__``.
    """
    version = _release_minor() if version is None else tuple(version)
    return [item for item in _REGISTRY.values() if _minor(item.removed_in) <= version]


def import_all() -> list[str]:
    """Import every non-test braincell module, so that all deprecations register.

    Returns
    -------
    list of str
        The module names, in import order.
    """
    import braincell

    names = [
        info.name
        for info in pkgutil.walk_packages(braincell.__path__, "braincell.")
        if not info.name.endswith("_test") and info.name.rpartition(".")[2] != "__main__"
    ]
    for name in names:
        importlib.import_module(name)
    return names


def main(argv: Sequence[str] | None = None) -> int:
    """List the deprecations due by a release; return 1 when there are any.

    This is ``python -m braincell._deprecation MAJOR.MINOR``, run before a
    release PR: each item it lists is removed (a Breaking Changes entry) or
    postponed (a later ``removed_in`` and a Deprecations entry) first.
    """
    parser = argparse.ArgumentParser(
        prog="python -m braincell._deprecation",
        description="List the braincell deprecations due for removal by a release.",
    )
    parser.add_argument("version", help="the release, as MAJOR.MINOR (e.g. 0.4)")
    args = parser.parse_args(argv)
    try:
        version = _minor(args.version)
    except ValueError as exc:
        parser.error(str(exc))
    modules = import_all()
    due = expired(version)
    count = f"{len(registered())} registered in {len(modules)} modules"
    if not due:
        print(f"No deprecation is due by braincell {args.version} ({count}).")
        return 0
    print(f"{len(due)} deprecation(s) due by braincell {args.version} ({count}):")
    for item in due:
        print(f"  {item.name} (since {item.since}, removed in {item.removed_in})")
    print(
        "Remove each one (a Breaking Changes entry in changelog.md), or postpone it "
        "(a later removed_in and a Deprecations entry)."
    )
    return 1
