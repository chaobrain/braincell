# Security Policy

## Supported versions

Security fixes are applied to the latest released version of `braincell`. We do not
backport fixes to older minor versions — please upgrade before reporting an issue.

| Version | Supported |
| --- | --- |
| Latest release on [PyPI](https://pypi.org/project/braincell/) | ✅ |
| Older releases | ❌ — please upgrade |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.** Doing so discloses the problem to everyone before a
fix is available.

Instead, use one of these private channels:

1. **GitHub private vulnerability reporting** (preferred) — go to the
   [Security tab](https://github.com/chaobrain/braincell/security/advisories/new) and
   click **Report a vulnerability**. This keeps the report visible only to the
   maintainers until an advisory is published.
2. **Email** — <chao.brain@qq.com>.

Please include as much of the following as you can:

- The type of issue and the affected component or module.
- Full paths of the source files involved, and the version/commit affected.
- Step-by-step instructions to reproduce, ideally a minimal proof of concept. If the
  issue involves a morphology file, attach the smallest file that triggers it.
- Your environment: `braincell`, `jax`/`jaxlib`, `brainstate`, and Python versions;
  OS; and platform (CPU / GPU / TPU).
- The impact you believe the issue has, including how an attacker might exploit it.

## What to expect

| Stage | Target |
| --- | --- |
| Acknowledgement of your report | within 5 days |
| Detailed response with next steps | within 10 days |
| Progress updates | until the issue is resolved |

These timelines may extend when maintainers are away, particularly around the end of
the year. After the initial reply we will keep you informed of progress toward a fix
and a public announcement, and may ask for additional information or guidance.

We support coordinated disclosure: we ask that you give us a reasonable opportunity to
release a fix before publishing details. With your permission, we will credit you in
the resulting security advisory.

## Scope

### What is in scope

`braincell` parses files and fetches data that a user may not have authored. Bugs that
let such input affect confidentiality, integrity, or availability beyond what the
documented API allows are in scope. In particular:

- **Morphology parsing.** `Morphology.from_swc` / `Morphology.from_asc` and the
  readers under `braincell/io/` consume SWC and Neurolucida ASC files, which are
  routinely downloaded from public archives. Memory-safety or resource-exhaustion
  defects driven by file *content* — unbounded allocation, uncontrolled recursion, or
  a crash that escapes as something other than a clean exception — are worth
  reporting.
- **Checkpoint loading.** `braincell/io/checkpoint.py` reads `.npz` archives. It
  loads them with `allow_pickle=False`, so arbitrary-object deserialization is
  deliberately disabled; a way to reach a pickle-loading path anyway, or to escape the
  intended file set during extraction, would be a vulnerability.
- **The NeuroMorpho.Org client.** `braincell.io.neuromorpho` and the
  `braincell-neuromorpho` console script make network requests and write files to
  disk. Path traversal via server-supplied names, or transport weaknesses that let a
  network attacker substitute content, are in scope.

### What is out of scope

- **`braincell` executes the models you give it.** Model definitions, channel
  classes, and mechanism parameters are ordinary Python: constructing a `Cell` from
  attacker-supplied Python is arbitrary code execution, and is not a vulnerability.
  Treat model code with exactly the same trust you give any other code you import.
- **Numerical disagreement is not a vulnerability.** A simulation that diverges from
  NEURON, an integrator that loses accuracy on a stiff channel, or a solver that fails
  to converge is a correctness bug — please open a normal GitHub issue.
- **Non-public API surface.** Per section 6 of `TODO.md`, only the documented public
  API is a stable surface; anything else is internal and may change without
  deprecation. Reports that depend on reaching into private modules will generally be
  treated as ordinary bugs.
- **Resource use under a legitimately large model.** Simulating a very large cell is
  expected to consume a lot of memory. This is only a security matter when a *small*
  input causes disproportionate consumption.

Vulnerabilities in third-party dependencies (JAX, jaxlib, brainstate, brainunit,
brainevent, braintools, brainpy, NumPy, SciPy, PyVista) are also out of scope. Report
those to their respective maintainers — though we appreciate a heads-up if `braincell`
is affected, so we can adjust our version constraints.

## Reporting a bug in a third party module

Security bugs in third-party modules should be reported to their respective
maintainers. If the issue reaches users *through* `braincell`, please tell us too.
