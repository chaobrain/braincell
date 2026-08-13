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

"""Stack-based morphology tree walks shared by the visualisation code.

Every traversal here is deliberately iterative. Morphologies are often
deep rather than bushy — a reconstructed dendrite can be a chain of
thousands of branches — and a recursive walk burns one interpreter frame
per branch of depth, so anything past roughly 400 branches raised
``RecursionError`` well before the tree itself became large.
"""

from braincell.morph import MorphoBranch

__all__ = ["iter_depth_first", "iter_bottom_up"]


def iter_depth_first(root: MorphoBranch) -> list[MorphoBranch]:
    """Return ``root``'s subtree in depth-first, left-to-right order.

    This is the order a ``for child in node.children: visit(child)``
    recursion produces: a node, then its first child's whole subtree,
    then its second child's, and so on.

    Parameters
    ----------
    root : MorphoBranch
        Root of the subtree to traverse.

    Returns
    -------
    list of MorphoBranch
        Every branch in the subtree, parents before their descendants.

    Examples
    --------
    .. code-block:: python

        >>> from braincell.vis._traversal import iter_depth_first
        >>> [branch.index for branch in iter_depth_first(morpho.root)]  # doctest: +SKIP
        [0, 1, 2, 3]
    """
    order: list[MorphoBranch] = []
    stack: list[MorphoBranch] = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        # Reversed, so the leftmost child is popped first.
        stack.extend(reversed(node.children))
    return order


def iter_bottom_up(root: MorphoBranch) -> list[MorphoBranch]:
    """Return ``root``'s subtree with every branch after all its children.

    Reversing a depth-first walk puts each node after everything below
    it, which is all a post-order accumulation (subtree leaf counts, path
    lengths, y-position averaging) needs. It costs one list instead of one
    interpreter frame per level.

    Parameters
    ----------
    root : MorphoBranch
        Root of the subtree to traverse.

    Returns
    -------
    list of MorphoBranch
        Every branch in the subtree, children before their parents;
        ``root`` is last.

    See Also
    --------
    iter_depth_first : The parents-first order this one reverses.

    Examples
    --------
    .. code-block:: python

        >>> from braincell.vis._traversal import iter_bottom_up
        >>> counts = {}
        >>> for node in iter_bottom_up(morpho.root):  # doctest: +SKIP
        ...     counts[node.index] = 1 if node.n_children == 0 else sum(
        ...         counts[child.index] for child in node.children
        ...     )
    """
    order = iter_depth_first(root)
    order.reverse()
    return order
