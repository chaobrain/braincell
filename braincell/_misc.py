# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import os
from typing import Any, Callable, TYPE_CHECKING

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

_BOUND_OPERATORS = ("ge", "gt", "le", "lt")

#: Longest profiler label XLA keeps intact; longer names are truncated.
_PROFILER_NAME_LIMIT = 180


def profiler_safe_name(raw: str) -> str:
    """Sanitize ``raw`` into a label safe for JAX scopes and named calls.

    Replaces every character that is not alphanumeric, ``":"`` or ``"_"``
    with an underscore, then truncates to ``_PROFILER_NAME_LIMIT``.

    Parameters
    ----------
    raw : str
        Unsanitized label, typically built from a state path and a class
        name.

    Returns
    -------
    str
        Sanitized label.

    Notes
    -----
    The allowed character set and the length limit must match across every
    caller or profiler traces from different subsystems stop lining up,
    which is why this lives in one place rather than being re-typed at each
    site.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in ":_" else "_" for ch in raw)
    return cleaned[:_PROFILER_NAME_LIMIT]


def profiler_scope_name(prefix: str, path, node) -> str:
    """Build a stable, profiler-safe internal JAX scope name."""
    path_name = "_".join(str(part) for part in path) if path else "root"
    class_name = type(getattr(node, "_channel", node)).__name__
    return profiler_safe_name(f"{prefix}:{path_name}:{class_name}")


def profiler_call_name(prefix: str, path, node) -> str:
    """Build a profiler-safe :func:`jax.named_call` name."""
    return profiler_scope_name(prefix, path, node).replace(":", "_")


def profile_barrier_current(current):
    """Optionally split membrane-current HLO for profiler attribution.

    The barrier is disabled by default because it can inhibit XLA fusion.
    Set ``BRAINCELL_PROFILE_SPLIT_CURRENTS=1`` when collecting profiler
    traces that need per-channel current attribution.

    Parameters
    ----------
    current : ArrayLike or brainunit.Quantity
        Current contribution to optionally fence off.

    Returns
    -------
    ArrayLike or brainunit.Quantity
        ``current`` unchanged, or wrapped in
        :func:`jax.lax.optimization_barrier` when the environment variable
        is set. Any unit is preserved.
    """
    if os.environ.get("BRAINCELL_PROFILE_SPLIT_CURRENTS") != "1":
        return current
    if hasattr(current, "unit"):
        return u.Quantity(jax.lax.optimization_barrier(u.get_mantissa(current)), current.unit)
    return jax.lax.optimization_barrier(current)


def validate_time_quantity(
    value,
    *,
    name: str,
    prefix: str,
    require_scalar: bool = True,
    require_positive: bool = True,
) -> None:
    """Require ``value`` to be a time :class:`brainunit.Quantity`.

    Shared by the single-cell and network run paths so both enforce the
    same contract and report it with their own caller name.

    Parameters
    ----------
    value : object
        Candidate time quantity.
    name : str
        Parameter name used in error messages (``"dt"``, ``"duration"``,
        ``"delay"``).
    prefix : str
        Caller name used to prefix error messages, e.g. ``"Cell.run(...)"``
        or ``"Network.run(...)"``. Without this the network layer reported
        single-cell wording for its own failures.
    require_scalar : bool, default True
        Require a scalar (or length-1) quantity. Pass ``False`` for
        parameters that are legitimately per-element, such as a vector of
        per-contact synaptic delays.
    require_positive : bool, default True
        Require a strictly positive value. Pass ``False`` where zero is
        meaningful, such as a zero delay meaning immediate delivery.

    Raises
    ------
    TypeError
        If ``value`` is not a quantity carrying a time unit.
    ValueError
        If ``value`` violates the requested scalar or positivity contract.
    """
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"{prefix} {name} must be a time quantity, got {value!r}.")
    decimal = np.asarray(value.to_decimal(u.ms), dtype=float)
    if require_scalar and decimal.shape not in ((), (1,)):
        raise ValueError(f"{prefix} {name} must be scalar, got shape {decimal.shape!r}.")
    if require_positive:
        if decimal.shape not in ((), (1,)):
            raise ValueError(f"{prefix} {name} must be scalar, got shape {decimal.shape!r}.")
        if float(decimal.reshape(())) <= 0.0:
            raise ValueError(f"{prefix} {name} must be > 0, got {value!r}.")


def is_traced_value(value) -> bool:
    """Return ``True`` when ``value`` is a live JAX tracer.

    Unwraps a :class:`brainunit.Quantity` first so that a Quantity wrapping
    a tracer is correctly identified as traced. Concrete numpy / JAX arrays
    and plain Python numbers return ``False``.
    """
    if isinstance(value, u.Quantity):
        value = u.get_mantissa(value)
    return isinstance(value, jax.core.Tracer)


def concat_values(values):
    """Concatenate trace values along axis 0, preserving any brainunit unit.

    Shared by the single-cell and network run paths, which both assemble
    per-chunk trace outputs into one array.
    """
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(u.math.concatenate(tuple(value.to_decimal(unit) for value in values), axis=0), unit)
    return u.math.concatenate(values, axis=0)


def same_time_quantity(left, right) -> bool:
    """Return ``True`` when two optional time quantities agree to tolerance.

    ``None`` compares equal only to ``None``.
    """
    if left is None or right is None:
        return left is right
    return bool(
        np.allclose(
            np.asarray(left.to_decimal(u.ms), dtype=float),
            np.asarray(right.to_decimal(u.ms), dtype=float),
            rtol=1e-7,
            atol=1e-9,
        )
    )


def cast_like(value, like):
    """Cast ``value`` to the dtype of ``like``, preserving any brainunit unit.

    Shared helper used across single- and multi-compartment spike
    detection and the Runge-Kutta integrators.
    """
    dtype = jnp.asarray(u.get_magnitude(like)).dtype
    if isinstance(value, u.Quantity):
        unit = u.get_unit(value)
        return jnp.asarray(value.to_decimal(unit), dtype=dtype) * unit
    return jnp.asarray(value, dtype=dtype)


def _to_unit(param: object, name: str, unit: Any) -> np.ndarray:
    """Convert a quantity-like value to a NumPy array in the target unit."""

    try:
        return np.asarray(param.to_decimal(unit))
    except Exception as exc:
        raise TypeError(f"{name} must satisfy unit {unit}.") from exc


def _to_shape(
    array: np.ndarray,
    *,
    name: str,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    """Check and optionally reshape array to match the shape specification."""

    if shape is None:
        return array

    n_expected_dims = len(shape)
    n_actual_dims = array.ndim

    if n_actual_dims < n_expected_dims:
        new_shape = (1,) * (n_expected_dims - n_actual_dims) + array.shape
        array = array.reshape(new_shape)
        n_actual_dims = array.ndim

    if n_actual_dims > n_expected_dims:
        extra_dims = array.shape[: n_actual_dims - n_expected_dims]
        if all(d == 1 for d in extra_dims):
            array = array.reshape(array.shape[n_actual_dims - n_expected_dims :])
        else:
            raise ValueError(f"{name}: expected {n_expected_dims}D, got {n_actual_dims}D with shape {array.shape}")

    for i, (expected, actual) in enumerate(zip(shape, array.shape)):
        if expected is not None and expected != actual:
            raise ValueError(f"{name}: expected dimension {i} to be {expected}, got {actual}")

    return array


def _normalize_bound(bound: object, *, unit: Any, name: str) -> np.ndarray:
    """Convert a quantity bound into a NumPy value in `unit`."""

    try:
        return _to_unit(bound, name, unit)
    except TypeError as exc:
        raise ValueError(f"{name} bounds must satisfy unit {unit}.") from exc


def _check_bounds(array: np.ndarray, *, name: str, unit: Any, bounds: dict[str, object] | None) -> None:
    """Validate a normalized NumPy array against simple scalar comparisons."""

    if not bounds:
        return

    invalid = tuple(key for key in bounds if key not in _BOUND_OPERATORS)
    if invalid:
        raise ValueError(f"{name} received unsupported bound keys {invalid!r}.")

    comparators = {
        "ge": (lambda lhs, rhs: lhs >= rhs, ">="),
        "gt": (lambda lhs, rhs: lhs > rhs, ">"),
        "le": (lambda lhs, rhs: lhs <= rhs, "<="),
        "lt": (lambda lhs, rhs: lhs < rhs, "<"),
    }
    for key in _BOUND_OPERATORS:
        if key not in bounds:
            continue
        bound = _normalize_bound(bounds[key], unit=unit, name=name)
        compare, symbol = comparators[key]
        if not np.all(compare(array, bound)):
            raise ValueError(f"{name} must satisfy {symbol} {bound!r}.")


def normalize_param(
    param: object,
    *,
    name: str,
    unit: Any,
    shape: int | tuple[int | None, ...] | None = None,
    bounds: dict[str, object] | None = None,
    allow_none: bool = False,
) -> Any:
    """Normalize one explicit-unit parameter and validate shape and bounds."""

    if param is None:
        if allow_none:
            return None
        raise TypeError(f"{name} cannot be None.")

    if isinstance(shape, int):
        shape = (shape,)

    array = _to_unit(param, name, unit)
    array = _to_shape(array, name=name, shape=shape)
    _check_bounds(array, name=name, unit=unit, bounds=bounds)
    return u.Quantity(array, unit)


def set_module_as(name: str):
    """Return a decorator that re-homes a function onto a public module path.

    BrainCell defines its public functions inside underscore-prefixed
    private modules and re-exports them from a public package, so a
    function's ``__module__`` points at the path users are not supposed to
    import from. Applying this decorator restores the public path, which
    is what :mod:`sphinx`, :func:`help`, :func:`repr`, and pickling all
    read.

    Parameters
    ----------
    name : str
        Public module path to advertise, such as ``'braincell'`` or
        ``'braincell.quad'``.

    Returns
    -------
    Callable
        A decorator that sets ``__module__`` on its argument and returns
        it unchanged.

    Notes
    -----
    Only ``__module__`` is touched. An earlier version assigned ``name``
    to ``__name__`` instead, which left every decorated function claiming
    to be called ``"braincell.quad"``; ``__name__`` is the function's own
    name and is deliberately left alone.

    Examples
    --------
    .. code-block:: python

        >>> from braincell._misc import set_module_as
        >>> @set_module_as('braincell.quad')
        ... def euler_step():
        ...     pass
        >>> euler_step.__module__
        'braincell.quad'
        >>> euler_step.__name__
        'euler_step'
    """

    def decorator(fun):
        fun.__module__ = name
        return fun

    return decorator


class Container(brainstate.mixin.Mixin):
    """
    A container class that provides a flexible structure for storing and accessing child elements.

    This class extends the brainstate.mixin.Mixin class and implements custom attribute
    and item access methods. It's designed to manage a collection of child elements
    of a specific type, providing type checking and convenient access patterns.

    Attributes:
        _container_name (str): The name of the container attribute that holds the child elements.

    Note:
        Subclasses should implement the `add` method to define how new elements
        are added to the container.
    """

    __module__ = 'braincell'

    _container_name: str

    @staticmethod
    def _format_elements(child_type: type, **children_as_dict):
        """
        Format and validate elements to ensure they are of the correct type.

        This method checks each element in the provided dictionary to ensure
        it is an instance of the specified child_type. It then constructs a
        new dictionary with validated elements.

        Args:
            child_type (type): The expected type of the child elements.
            **children_as_dict: Arbitrary keyword arguments representing
                                the children elements to be formatted and validated.

        Returns:
            dict: A new dictionary containing the validated child elements.

        Raises:
            TypeError: If any element in children_as_dict is not an instance of child_type.
        """
        res = {}

        # add dict-typed components
        for k, v in children_as_dict.items():
            if not isinstance(v, child_type):
                raise TypeError(f'Should be instance of {child_type.__name__}. But we got {type(v)}')
            res[k] = v
        return res

    if not TYPE_CHECKING:

        def __getitem__(self, item):
            """
            Overwrite the slice access (`self['']`).
            """
            children = self.__getattr__(self._container_name)
            if item in children:
                return children[item]
            else:
                raise ValueError(f'Unknown item {item}, we only found {list(children.keys())}')

        def __getattr__(self, item):
            """
            Overwrite the dot access (`self.`).
            """
            name = super().__getattribute__('_container_name')
            if item == '_container_name':
                return name
            children = super().__getattribute__(name)
            if item == name:
                return children
            return children[item] if item in children else super().__getattribute__(item)

    def add(self, *elems, **elements):
        """
        Add new elements to the container.

        This method is intended to be implemented by subclasses to define
        how new elements are added to the container. The base implementation
        raises a NotImplementedError.

        Args:
            *elems: Variable length argument list of elements to be added.
            **elements: Arbitrary keyword arguments representing named elements to be added.

        Raises:
            NotImplementedError: This method must be implemented by the subclass.

        Note:
            Subclasses should override this method to provide specific implementation
            for adding elements to the container.
        """
        raise NotImplementedError('Must be implemented by the subclass.')


class TreeNode(brainstate.mixin.Mixin):
    """
    A base class for tree-like structures that enforces type checking between root and leaf nodes.

    This class provides methods to validate the compatibility between root and leaf nodes
    in a tree-like structure. It's designed to be subclassed by specific node types that
    need to maintain a consistent hierarchy.

    Attributes:
        root_type (type): The expected type of the root node for this TreeNode.

    Note:
        Subclasses should define the `root_type` attribute to specify the expected
        type of their root node.
    """

    __module__ = 'braincell'

    root_type: type

    @staticmethod
    def _root_leaf_pair_check(root: type, leaf: 'TreeNode'):
        """
        Check if the root and leaf types are compatible.

        Args:
            root (type): The type of the root node.
            leaf (TreeNode): The leaf node to check against the root.

        Raises:
            ValueError: If the leaf does not have a 'root_type' attribute.
            TypeError: If the root is not a subclass of the leaf's root_type.
        """
        if hasattr(leaf, 'root_type'):
            root_type = leaf.root_type
        else:
            raise ValueError(
                'Child class should define "root_type" to '
                'specify the type of the root node. '
                f'But we did not found it in {leaf}'
            )
        if not issubclass(root, root_type):
            raise TypeError(
                f'Type does not match. {leaf} requires a root with type '
                f'of {leaf.root_type}, but the root now is {root}.'
            )

    @staticmethod
    def check_hierarchies(root: type, *leaves, check_fun: Callable = None, **named_leaves):
        """
        Recursively check the hierarchies of nodes against a root type.

        This method verifies that all leaves in the hierarchy are compatible with the given root type.
        It can handle leaves passed as positional arguments (which can be individual nodes, lists, tuples, or dicts)
        and as keyword arguments.

        Args:
            root (type): The type of the root node to check against.
            *leaves: Variable length argument list of leaves to check. Can be individual nodes,
                     lists, tuples, or dicts.
            check_fun (Callable, optional): A custom function to use for checking root-leaf compatibility.
                                            If None, uses the default _root_leaf_pair_check method.
            **named_leaves: Arbitrary keyword arguments representing named leaves to check.

        Raises:
            ValueError: If an unsupported type is encountered in leaves or if a named leaf
                        is not an instance of brainstate.graph.Node.
        """
        if check_fun is None:
            check_fun = TreeNode._root_leaf_pair_check

        for leaf in leaves:
            if isinstance(leaf, brainstate.graph.Node):
                check_fun(root, leaf)
            elif isinstance(leaf, (list, tuple)):
                TreeNode.check_hierarchies(root, *leaf, check_fun=check_fun)
            elif isinstance(leaf, dict):
                TreeNode.check_hierarchies(root, **leaf, check_fun=check_fun)
            else:
                raise ValueError(f'Do not support {type(leaf)}.')
        for leaf in named_leaves.values():
            if not isinstance(leaf, brainstate.graph.Node):
                raise ValueError(f'Do not support {type(leaf)}. Must be instance of {brainstate.graph.Node}')
            check_fun(root, leaf)
