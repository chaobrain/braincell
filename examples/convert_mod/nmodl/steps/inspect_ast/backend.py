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

from functools import lru_cache


AST_NODE_TYPE_ALIASES = {
    # Current nmodl.dsl exposes the longitudinal diffusion node under this legacy name.
    "LONGITUDINAL_DIFFUSION_BLOCK": ("LON_DIFUSE",),
}


def _nmodl_import_error_message() -> str:
    try:
        import neuron  # type: ignore

        neuron_version = getattr(neuron, "__version__", "unknown")
    except Exception:
        neuron_version = "unavailable"

    return (
        "Unable to import an NMODL Python API backend. "
        f"Detected NEURON version: {neuron_version}. "
        "Tried `neuron.nmodl` and `nmodl.dsl`. "
        "This environment likely has NEURON installed without the external `nmodl` package. "
        "Install Blue Brain's `nmodl` package in the current environment, or switch to an "
        "environment that exposes `neuron.nmodl`."
    )


def load_nmodl():
    try:
        from neuron.nmodl import NmodlDriver, ast, to_json, to_nmodl, visitor
    except ImportError:
        try:
            import nmodl.dsl as nmodl
        except ImportError as exc:
            raise SystemExit(_nmodl_import_error_message()) from exc
        else:
            return (
                nmodl.NmodlDriver,
                nmodl.ast,
                nmodl.visitor,
                nmodl.to_json,
                nmodl.to_nmodl,
            )

    return NmodlDriver, ast, visitor, to_json, to_nmodl


@lru_cache(maxsize=1)
def _bindings():
    return load_nmodl()


def _resolve_ast_node_type(ast, node_type_name: str):
    candidates = (node_type_name, *AST_NODE_TYPE_ALIASES.get(node_type_name, ()))
    for candidate in candidates:
        node_type = getattr(ast.AstNodeType, candidate, None)
        if node_type is not None:
            return node_type
    return None


def lookup(program, node_type_name: str):
    _, ast, visitor, _, _ = _bindings()
    node_type = _resolve_ast_node_type(ast, node_type_name)
    if node_type is None:
        return []
    lookup_visitor = visitor.AstLookupVisitor()
    return list(lookup_visitor.lookup(program, node_type))


def node_text(node) -> str:
    _, _, _, _, to_nmodl = _bindings()
    return to_nmodl(node).strip()


def ast_to_json(program) -> str:
    _, _, _, to_json, _ = _bindings()
    return to_json(program)
