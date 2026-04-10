from __future__ import annotations

from pathlib import Path

from .backend import ast_to_json
from .canonical import HANDLED_CANONICAL_BLOCKS
from .canonical import build_canonical_blocks
from .canonical import collect_unhandled_raw_blocks
from .canonical import maybe_number
from .canonical import pascal_case
from .canonical import strip_unit
from .parser import collect_block_counts
from .parser import parse_program
from .parser import reconstruct_nmodl
from .raw_blocks import BLOCK_ORDER
from .raw_blocks import extract_raw_blocks
from .variants.canonical_default import VARIANT_NAME as CANONICAL_DEFAULT_VARIANT
from .variants.canonical_default import run as run_canonical_default

STEP_RUNNERS = {
    CANONICAL_DEFAULT_VARIANT: run_canonical_default,
}


def default_mod_file() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "hh.mod"


def resolve_mod_file(input_path: str | None) -> Path:
    mod_file = Path(input_path).expanduser().resolve() if input_path else default_mod_file()
    if not mod_file.exists():
        raise SystemExit(f"MOD file not found: {mod_file}")
    return mod_file


def get_variants() -> list[str]:
    return sorted(STEP_RUNNERS)


def run(mod_file: Path, *, variant: str = CANONICAL_DEFAULT_VARIANT):
    try:
        runner = STEP_RUNNERS[variant]
    except KeyError as exc:
        raise SystemExit(f"Unknown Step 1 variant `{variant}`. Available: {', '.join(get_variants())}") from exc
    return runner(mod_file)
