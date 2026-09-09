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

"""Check paired control comparisons, exclusions and model identity guards."""

from dataclasses import asdict
import csv
import json
import re
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmarks.performance.synapse_events import controls_report as r
from benchmarks.performance.synapse_events.delivery_benchmark import CONTROL_JOBS
from benchmarks.performance.synapse_events.query_benchmark import QueryCase


def _fixture(root):
    directory = root / "matched"
    directory.mkdir()
    cfg = asdict(QueryCase(n=100, m=100))
    protocol = dict(
        phase="controls",
        cases=list(CONTROL_JOBS),
        devices=["cpu", "gpu"],
        rounds=3,
        precision=32,
        methods=["direct", "padded"],
        sources={"driver": "hash"},
        case_configs={k: cfg for k in CONTROL_JOBS},
    )
    manifest = dict(protocol=protocol, rows=[])
    for rep in range(3):
        for job, (control, candidates) in CONTROL_JOBS.items():
            for device in ("cpu", "gpu"):
                methods = ["production", *candidates]
                rows = []
                scale = (2 if device == "gpu" else 1) * (2 if job.endswith("padded") else 1)
                baseline = dict(cell_only=0.01, unconnected=0.03, silent=0.05, active=0.07)[control] * scale
                for method in methods:
                    seconds = baseline * dict(production=1, direct=0.5, padded=0.4)[method]
                    metric = dict(median_s=seconds, min_s=seconds * 0.9, max_s=seconds * 1.1, first_call_s=0.1)
                    v = dict(
                        synapse_count=0 if control == "cell_only" else 10000,
                        connection_rows=10000 if candidates else 0,
                        finite=True,
                        reset_reproducible=True,
                        reset_voltage_max_error_mV=0,
                        reset_g_max_error=0,
                        final_time_ms=100,
                        final_step=4000,
                        arrivals=100000 if control == "active" else 0,
                        source_shape=[10000, 10] if candidates else None,
                        schedule_sha256=f"source-{control}" if candidates else None,
                        drive_sha256=f"drive-{control}",
                    )
                    rows.append(
                        dict(
                            method=method,
                            status="ok",
                            build_s=0.01,
                            prepare_cold_s=0.02,
                            reset_cold_s=0.003,
                            array_bytes=1024,
                            validation=v,
                            reset_timing={"median_s": 0.001},
                            voltage_max_error_mV=0,
                            g_max_error=0,
                            drive_max_error=0,
                            timing={"full": metric},
                        )
                    )
                name = f"{device}_{job}_r{rep}.json"
                data = dict(
                    status="ok", case=cfg, control=control, rows=rows, environment={"versions": {"jax": "0.10.1"}}
                )
                (directory / name).write_text(json.dumps(data))
                manifest["rows"].append(
                    dict(file=name, device=device, case=job, round=rep, control=control, methods=methods)
                )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


class ControlsReportTest(unittest.TestCase):
    def test_generated_reports_stay_local_and_resolve_links_from_custom_output(self):
        with tempfile.TemporaryDirectory(prefix="report paths ") as tmp:
            root = Path(tmp)
            _fixture(root)
            maintained = [r.HERE / "RESULTS.md", *sorted((r.HERE / "results").glob("*.md"))]
            before = {p: p.read_bytes() for p in maintained}
            self.assertEqual(r.main([str(root)]), 0)
            default_output = root / r.REPORT_FILENAME
            self.assertTrue(default_output.is_file())
            output = root / "separate reports" / "nested" / "custom.md"
            self.assertEqual(r.main([str(root), "--report", str(output)]), 0)
            self.assertTrue(output.is_file())
            for document in (default_output, output):
                links = re.findall(r"\]\(<([^>]+)>\)", document.read_text())
                self.assertTrue(links)
                for target in links:
                    self.assertTrue((document.parent / target).exists(), target)
                resolved = {(document.parent / target).resolve() for target in links}
                self.assertIn((r.HERE / "README.md").resolve(), resolved)
            self.assertEqual(before, {p: p.read_bytes() for p in maintained})

    def test_full_export_pairs_differences_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            rows, failures, excluded, configs, environments = r.collect_controls(root)
            self.assertEqual((len(rows), failures, excluded), (60, [], []))
            self.assertEqual(set(configs), set(CONTROL_JOBS))
            self.assertEqual(set(environments), {"cpu", "gpu"})
            summary, deltas = r.summarize_controls(rows)
            self.assertEqual(len(summary), 16)
            self.assertEqual(len(deltas), 14)
            direct = next(x for x in summary if (x['device'], x['case'], x['method']) == ('cpu', 'active', 'direct'))
            self.assertAlmostEqual(direct['speedup'], 2)
            self.assertTrue(direct['stable_gain'])
            self.assertAlmostEqual(direct['one_use_ms'], 133)
            self.assertAlmostEqual(direct['us_per_step'], 35 * 1000 / 4000)
            delta = next(x for x in deltas if (x['device'], x['comparison'], x['method']) == ('cpu', 'D-C', 'direct'))
            self.assertEqual(delta['rounds'], 3)
            self.assertAlmostEqual(delta['median_ms'], 10)
            incomplete = [x for x in rows if x['control'] == 'active' and x['method'] == 'direct']
            self.assertEqual(r.summarize_controls(incomplete)[1], [])
            self.assertIsNone(r.summarize_controls(incomplete)[0][0]['speedup'])
            out = root / 'tables.md'
            result = r.export_report(root, out)
            self.assertEqual(result['failures'], [])
            self.assertTrue((root / 'controls_comparison.svg').exists())
            self.assertIn('不能解释为内核独占时间', out.read_text())
            with (root / 'controls_raw.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 60)
            self.assertEqual(r.main([tmp, '--report', str(out)]), 0)

    def test_reject_incompatible_or_unverified_evidence(self):
        mutations = {
            'assignment': lambda d: d.update(control='silent'),
            'config': lambda d: d['case'].update(n=50),
            'environment': lambda d: d['environment']['versions'].update(jax='different'),
            'count': lambda d: d['rows'][0]['validation'].update(connection_rows=0),
            'finite': lambda d: d['rows'][0].update(g_max_error=float('nan')),
            'signature': lambda d: d['rows'][0]['validation'].update(drive_sha256='different'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                directory, _ = _fixture(root)
                path = directory / 'gpu_active_direct_r0.json'
                data = json.loads(path.read_text())
                mutate(data)
                path.write_text(json.dumps(data))
                self.assertTrue(r.collect_controls(root)[1])

    def test_missing_timeout_exclusion_and_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, manifest = _fixture(root)
            (directory / 'cpu_cell_only_r0.json').unlink()
            (directory / 'gpu_cell_only_r0.json').write_text(json.dumps({'status': 'timeout', 'reason': 'deadline'}))
            path = directory / 'cpu_active_padded_r0.json'
            data = json.loads(path.read_text())
            data['rows'][1] = {'method': 'padded', 'status': 'not_applicable', 'reason': 'resource limit'}
            path.write_text(json.dumps(data))
            rows, failures, excluded, _, _ = r.collect_controls(root)
            self.assertTrue(failures)
            self.assertEqual(len(excluded), 2)
            self.assertEqual(len(rows), 57)
            second = root / 'second'
            shutil.copytree(directory, second)
            altered = json.loads((second / 'manifest.json').read_text())
            altered['protocol']['sources']['driver'] = 'changed'
            (second / 'manifest.json').write_text(json.dumps(altered))
            self.assertTrue(any('different source snapshots' in x for x in r.collect_controls(root)[1]))
            manifest['protocol']['trace_full'] = True
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            self.assertTrue(any('profiling' in x for x in r.collect_controls(root)[1]))
            manifest['protocol'].pop('trace_full')
            # Use a successful worker: failed workers are intentionally excluded
            # before their configuration can enter a comparison.
            manifest['rows'][2]['case'] = 'unknown'
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            self.assertTrue(any('unknown controls job' in x for x in r.collect_controls(root)[1]))
        with tempfile.TemporaryDirectory() as tmp:
            result = r.export_report(Path(tmp), Path(tmp) / 'empty.md')
            self.assertTrue(result['failures'])
