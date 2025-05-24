# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import warnings
from typing import Callable, TYPE_CHECKING

import brainstate


def deprecation_getattr(module, deprecations):
    """
    Create a custom getattr function to handle deprecated attributes.

    This function generates a custom getattr function for a module, which
    checks if an attribute is deprecated and handles it accordingly by
    raising an AttributeError or issuing a warning.

    Parameters
    ----------
    module : str
        The name of the module for which the custom getattr function is created.
    deprecations : dict
        A dictionary where keys are attribute names and values are tuples
        containing a deprecation message and an optional function. If the
        function is None, accessing the attribute will raise an AttributeError.

    Returns
    -------
    function
        A custom getattr function that handles deprecated attributes.
    """

    def getattr(name):
        if name in deprecations:
            message, fn = deprecations[name]
            if fn is None:  # Is the deprecation accelerated?
                raise AttributeError(message)
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return getattr


def set_module_as(name: str):
    """
    A decorator factory to set the `__name__` attribute of a module.

    This function returns a decorator that, when applied to a module,
    sets the module's `__name__` attribute to the specified name.

    Args:
        name (str): The new name to assign to the module's `__name__` attribute.

    Returns:
        Callable: A decorator that modifies the `__name__` attribute of the module.
    """

    def decorator(module):
        """
        Decorator to set the `__name__` attribute of the given module.

        Args:
            module: The module whose `__name__` attribute will be modified.

        Returns:
            The modified module with its `__name__` attribute updated.
        """
        module.__name__ = name
        return module

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
        Subclasses should implement the `add_elem` method to define how new elements
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
                raise TypeError(f'Should be instance of {child_type.__name__}. '
                                f'But we got {type(v)}')
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

    def add_elem(self, **elements):
        self.add(**elements)


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
            raise ValueError('Child class should define "root_type" to '
                             'specify the type of the root node. '
                             f'But we did not found it in {leaf}')
        if not issubclass(root, root_type):
            raise TypeError(f'Type does not match. {leaf} requires a root with type '
                            f'of {leaf.root_type}, but the root now is {root}.')

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
