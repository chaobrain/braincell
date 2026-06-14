

import json
from pathlib import Path
from typing import Any

ARTIFACT_FILENAMES = (
    "ast.json",
    "raw_blocks.json",
    "canonical_blocks.json",
    "bc_ast.json",
    "semantic_ir.json",
    "target_ir.json",
    "rendered_channel.py",
    "validation.json",
    "metadata.json",
)


def save_pipeline_artifacts(result: dict[str, Any], artifact_dir: str | Path) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    step1_result = result["step1_result"]
    step2_result = result["step2_result"]
    step3_result = result["step3_result"]
    spec = result["spec"]

    payloads = {
        "ast.json": json.dumps(step1_result["ast_json"], indent=2, ensure_ascii=False),
        "raw_blocks.json": json.dumps(step1_result["raw_blocks"], indent=2, ensure_ascii=False),
        "canonical_blocks.json": json.dumps(step1_result["canonical_blocks"], indent=2, ensure_ascii=False),
        "bc_ast.json": json.dumps(step1_result["bc_ast"], indent=2, ensure_ascii=False),
        "semantic_ir.json": json.dumps(step2_result["semantic_ir"], indent=2, ensure_ascii=False),
        "target_ir.json": json.dumps(step2_result["target_ir"], indent=2, ensure_ascii=False),
        "rendered_channel.py": step3_result["rendered_text"],
        "validation.json": json.dumps(step3_result["validation"], indent=2, ensure_ascii=False),
    }

    managed_files = set(ARTIFACT_FILENAMES)
    for path in artifact_dir.iterdir():
        if path.is_file() and path.name not in managed_files:
            path.unlink()

    for filename, payload in payloads.items():
        (artifact_dir / filename).write_text(payload, encoding="utf-8")

    metadata = {
        "source_file": step3_result["source_file"],
        "pipeline": spec.pipeline_name,
        "saved_files": list(ARTIFACT_FILENAMES),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "artifact_dir": str(artifact_dir),
        "source_file": step3_result["source_file"],
        "pipeline": spec.pipeline_name,
        "saved_files": list(ARTIFACT_FILENAMES),
    }
