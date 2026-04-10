

from dataclasses import dataclass

from .inspect_ast import CANONICAL_DEFAULT_VARIANT
from .inspect_ast import get_variants as get_step1_variants
from .inspect_ir import ONE_ION_HH_OHMIC_VARIANT
from .inspect_ir import get_variants as get_step2_variants
from .render import BRAINCELL_ONE_ION_HH_OHMIC_VARIANT
from .render import get_variants as get_step3_variants


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_name: str
    step1: str
    step2: str
    step3: str


DEFAULT_PIPELINE_NAME = "__".join(
    (
        CANONICAL_DEFAULT_VARIANT,
        ONE_ION_HH_OHMIC_VARIANT,
        BRAINCELL_ONE_ION_HH_OHMIC_VARIANT,
    )
)

PIPELINE_SPECS = {
    DEFAULT_PIPELINE_NAME: PipelineSpec(
        pipeline_name=DEFAULT_PIPELINE_NAME,
        step1=CANONICAL_DEFAULT_VARIANT,
        step2=ONE_ION_HH_OHMIC_VARIANT,
        step3=BRAINCELL_ONE_ION_HH_OHMIC_VARIANT,
    )
}


def get_step_choices() -> dict[str, list[str]]:
    return {
        "step1": get_step1_variants(),
        "step2": get_step2_variants(),
        "step3": get_step3_variants(),
        "pipeline": sorted(PIPELINE_SPECS),
    }


def resolve_pipeline_spec(
    *,
    pipeline_name: str | None = None,
    step1: str | None = None,
    step2: str | None = None,
    step3: str | None = None,
) -> PipelineSpec:
    if pipeline_name:
        spec = PIPELINE_SPECS.get(pipeline_name)
        if spec is None:
            raise SystemExit(
                f"Unknown pipeline `{pipeline_name}`. Available pipelines: {', '.join(sorted(PIPELINE_SPECS))}"
            )
        if step1 and step1 != spec.step1:
            raise SystemExit(f"Pipeline `{pipeline_name}` requires --step1 {spec.step1}.")
        if step2 and step2 != spec.step2:
            raise SystemExit(f"Pipeline `{pipeline_name}` requires --step2 {spec.step2}.")
        if step3 and step3 != spec.step3:
            raise SystemExit(f"Pipeline `{pipeline_name}` requires --step3 {spec.step3}.")
        return spec

    chosen = PipelineSpec(
        pipeline_name="custom",
        step1=step1 or CANONICAL_DEFAULT_VARIANT,
        step2=step2 or ONE_ION_HH_OHMIC_VARIANT,
        step3=step3 or BRAINCELL_ONE_ION_HH_OHMIC_VARIANT,
    )

    for spec in PIPELINE_SPECS.values():
        if (chosen.step1, chosen.step2, chosen.step3) == (spec.step1, spec.step2, spec.step3):
            return spec

    raise SystemExit(
        "Unsupported pipeline combination: "
        f"{chosen.step1} + {chosen.step2} + {chosen.step3}. "
        "Define it in pipeline/registry.py before using it."
    )
