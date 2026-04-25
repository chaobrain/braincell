"""Shared fixtures for multi-compartment cable tests and sweeps."""



from pathlib import Path
import tempfile
import textwrap


MORPHO_FILES = Path("/home/swl/braincell/examples/multi_compartment/morpho_files")
UNBRANCHED_SOMA_SWC = str(MORPHO_FILES / "unbranched_soma.swc")
BRANCHED_DEND_SWC = str(MORPHO_FILES / "branched_dend.swc")
BRANCHED_SOMA_SWC = str(MORPHO_FILES / "branched_soma.swc")
BRANCHED_SOMA_1_SWC = str(MORPHO_FILES / "branched_soma_1.swc")
IO_SWC = str(MORPHO_FILES / "io.swc")
BC_SWC = str(MORPHO_FILES / "bc.swc")
GOC_ASC = str(MORPHO_FILES / "goc.asc")
PC_ASC = str(MORPHO_FILES / "pc.asc")


def base_case_payload(
    *,
    case_id: str = "smoke",
    morphology_kind: str = "swc",
    morphology_path: str = UNBRANCHED_SOMA_SWC,
    dt_ms: float = 0.1,
    duration_ms: float = 2.0,
    v_init_mV: float = -65.0,
    ra_ohm_cm: float = 100.0,
    cm_uF_cm2: float = 1.0,
    cv_per_branch: int = 1,
    stimulus: dict | None = None,
) -> dict:
    payload = {
        "template_family": "multi_compartment_cable",
        "case_id": case_id,
        "morphology": {
            "kind": morphology_kind,
            "path": morphology_path,
        },
        "simulation": {
            "dt_ms": dt_ms,
            "duration_ms": duration_ms,
            "v_init_mV": v_init_mV,
        },
        "cable": {
            "ra_ohm_cm": ra_ohm_cm,
            "cm_uF_cm2": cm_uF_cm2,
        },
        "cv_policy": {
            "kind": "CVPerBranch",
            "cv_per_branch": cv_per_branch,
        },
        "stimulus": stimulus or dc_step_stimulus(),
    }
    return payload


def dc_step_stimulus(
    *,
    delay_ms: float = 0.5,
    dur_ms: float = 1.0,
    amp_nA: float = 0.05,
) -> dict:
    return {
        "kind": "dc_step",
        "target": "root_soma_midpoint",
        "delay_ms": delay_ms,
        "dur_ms": dur_ms,
        "amp_nA": amp_nA,
    }


def piecewise_step_stimulus(
    *,
    start_ms: float = 0.0,
    durations_ms: list[float] | tuple[float, ...] = (0.025, 0.025),
    amplitudes_nA: list[float] | tuple[float, ...] = (0.0, 0.01),
) -> dict:
    return {
        "kind": "piecewise_step",
        "target": "root_soma_midpoint",
        "start_ms": start_ms,
        "durations_ms": list(durations_ms),
        "amplitudes_nA": list(amplitudes_nA),
    }


def sine_stimulus(
    *,
    start_ms: float = 0.0,
    duration_ms: float = 0.1,
    amplitude_nA: float = 0.01,
    frequency_hz: float = 100.0,
    phase_rad: float = 0.0,
    offset_nA: float = 0.0,
) -> dict:
    return {
        "kind": "sine",
        "target": "root_soma_midpoint",
        "start_ms": start_ms,
        "duration_ms": duration_ms,
        "amplitude_nA": amplitude_nA,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "offset_nA": offset_nA,
    }


def write_temp_swc(testcase, body: str, filename: str = "sample.swc") -> Path:
    temp_dir = tempfile.TemporaryDirectory()
    testcase.addCleanup(temp_dir.cleanup)
    path = Path(temp_dir.name) / filename
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


def write_temp_asc(testcase, body: str, filename: str = "sample.asc") -> Path:
    temp_dir = tempfile.TemporaryDirectory()
    testcase.addCleanup(temp_dir.cleanup)
    path = Path(temp_dir.name) / filename
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path
