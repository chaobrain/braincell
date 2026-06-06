

import json
from typing import Any

from ...model import to_payload
from ..backend import ast_to_json
from ..canonical import build_canonical_blocks
from ..parser import collect_block_counts
from ..parser import parse_program
from ..parser import reconstruct_nmodl
from ..raw_blocks import extract_raw_blocks
from ..typed_ast import build_typed_ast

VARIANT_NAME = "canonical_default"


def run(mod_file) -> dict[str, Any]:
    program = parse_program(mod_file)
    raw_blocks = extract_raw_blocks(program)
    canonical_blocks = build_canonical_blocks(program)
    reconstructed = reconstruct_nmodl(program)
    typed_ast = build_typed_ast(
        mod_file=str(mod_file),
        source_text=reconstructed,
        canonical_blocks=canonical_blocks,
        raw_blocks=raw_blocks,
    )
    return {
        "variant": VARIANT_NAME,
        "mod_file": mod_file,
        "source_file": str(mod_file),
        "program": program,
        "ast_root_type": program.get_node_type_name(),
        "block_counts": dict(collect_block_counts(program)),
        "reconstructed_nmodl": reconstructed,
        "raw_blocks": raw_blocks,
        "canonical_blocks": canonical_blocks,
        "bc_ast_model": typed_ast,
        "bc_ast": to_payload(typed_ast),
        "ast_json": json.loads(ast_to_json(program)),
    }
