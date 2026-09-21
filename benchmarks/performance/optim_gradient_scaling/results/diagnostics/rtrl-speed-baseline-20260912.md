# RTRL speed baseline across C (2026-09-12)

This baseline uses the existing functional HH validation workload at 1600
steps, x64, GPU 0, and five steady samples per method. It is a diagnostic
baseline for the speed-first exploration, not a replacement for the formal
four-C crossover suite.

| C | method | compile s | steady samples s | median s | temporary bytes | logical carry bytes |
|---:|---|---:|---|---:|---:|---:|
| 1 | BPTT | 2.482 | 0.26804, 0.26700, 0.26858, 0.26815, 0.26784 | 0.26804 | 1,580,968 | — |
| 1 | full RTRL | 1.563 | 0.11024, 0.10741, 0.10751, 0.10376, 0.10589 | 0.10741 | 9,360 | 2,844 |
| 1 | compact RTRL | 1.569 | 0.13382, 0.13371, 0.13393, 0.13362, 0.13422 | 0.13382 | 8,560 | 432 |
| 5 | BPTT | 2.778 | 0.34298, 0.34673, 0.34637, 0.32006, 0.28935 | 0.34298 | 2,750,192 | — |
| 5 | full RTRL | 1.579 | 0.16147, 0.16007, 0.16040, 0.16361, 0.16351 | 0.16147 | 9,872 | 7,380 |
| 5 | compact RTRL | 1.621 | 0.19768, 0.19177, 0.20081, 0.19435, 0.19459 | 0.19459 | 12,144 | 1,200 |
| 21 | BPTT | 5.149 | 0.81255, 0.77214, 0.70156, 0.74298, 0.81357 | 0.77214 | 12,570,016 | — |
| 21 | full RTRL | 2.326 | 0.35197, 0.34847, 0.35405, 0.34958, 0.34961 | 0.34961 | 92,816 | 119,700 |
| 21 | compact RTRL | 2.627 | 0.36817, 0.37655, 0.37066, 0.37384, 0.37225 | 0.37225 | 85,616 | 21,168 |

Full RTRL remains faster than compact RTRL at all three points. The compact
path is about 24.6% slower at C=1, 20.5% slower at C=5, and 6.5% slower at
C=21, while reducing logical carry. This supports keeping full RTRL as the
speed path and treating compact RTRL as a memory path until direct active-state
JVP removes its embed/extract overhead.

The C=5 and C=21 samples show noticeable GPU/process variation, especially
BPTT at C=5 and C=21. These values are a baseline for candidate screening;
formal adoption still requires the independent-worker protocol and the 3%
minimum improvement criterion.
