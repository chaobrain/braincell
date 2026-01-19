# Release Notes

## Version 0.0.7

This release focuses on structural refactoring to improve codebase organization, specifically grouping morphology and integrator components into dedicated sub-packages (`braincell.morph` and `braincell.quad`).

### Refactoring & Code Organization

- **Morphology Sub-package** (`braincell.morph`)
  - Moved and renamed morphology-related modules into `braincell/morph/`:
    - `_morphology.py` -> `morph/_morphology.py`
    - `_morphology_branch_tree.py` -> `morph/_branch_tree.py`
    - `_morphology_from_asc.py` -> `morph/_from_asc.py`
    - `_morphology_from_swc.py` -> `morph/_from_swc.py`
    - `_morphology_utils.py` -> `morph/_utils.py`

- **Integrator Sub-package** (`braincell.quad`)
  - Moved all integrator and solver modules into `braincell/quad/`:
    - `_integrator*.py` files moved to `braincell/quad/`.
  - This improves the clarity of the top-level namespace.

### Documentation

- **Structure Updates**
  - Updated API documentation to reflect the new module structure.
  - Simplified `index.rst` and reorganized API reference pages.
  - Updated copyright to reflect membership in the BrainX Ecosystem.

### CI/CD

- **Workflow Updates**
  - Bumped versions for `actions/checkout`, `upload-artifact`, and `download-artifact`.

## Version 0.0.6

This release focuses on major dependency updates, code modernization, and extensive refactoring to improve compatibility with the latest BrainPy ecosystem.

### Breaking Changes

- **Dependency Version Updates**
  - Updated `brainstate` from `>=0.1.0` to `>=0.2.0`
  - Updated `brainpy` from `>=3.0.0` to `>=2.7.0`
  - These updates may require users to upgrade their BrainPy ecosystem packages

### Refactoring & Code Improvements

- **Core Architecture Simplification** (2acd212)
  - Refactored `HHTypedNeuron` to use `brainpy` directly for better integration
  - Simplified `_base.py` with significant code reduction (221 insertions, 282 deletions)
  - Removed deprecated `_integrator_diffrax.py` module (29 lines removed)
  - Streamlined integrator implementations in `_integrator_runge_kutta.py`
  - Cleaned up `_single_compartment.py` and integration protocol

- **Parameter Initialization Migration** (fa71171, a79c306, 18b053c, 77a11ac)
  - Migrated parameter initialization from `brainstate.nn` to `braintools` across the entire codebase
  - Updated parameter initialization in ion channels (calcium, potassium, sodium, hyperpolarization-activated)
  - Refactored parameter initialization in synapse models (markov)
  - Updated HTC and EINet classes to use `braintools`
  - Updated all example scripts and notebooks to use `braintools` for parameter initialization

- **API Migration** (e84351a, bf50e6e)
  - Migrated from `brainstate.nn` to `brainpy.state` and `braintools`
  - Fixed `_base` errors in brainpy integration
  - Updated `CurrentProj` references across the codebase

### Documentation

- **Updated Documentation** (#54, 2acd212)
  - Updated braincell logo image
  - Refreshed tutorial notebooks (cell, channel, ion tutorials in both English and Chinese)
  - Updated advanced tutorial examples (sc02-sc05 notebooks)
  - Revised quickstart concepts documentation
  - Updated all documentation to reflect API changes and new parameter initialization patterns

### Examples

- **Example Updates**
  - Updated all example scripts to use new APIs:
    - `SC01_fitting_a_hh_neuron.py`
    - `SC03_COBA_HH_2007_braincell.py`
    - `SC05_thalamus_single_compartment_neurons.py`
    - `SC06_unified_thalamus_model.py`
    - `SC07_Straital_beta_oscillation_2011.py`
    - `MC11_simple_dendrite_model.py`
    - `MC13_golgi_model/` simulations

### CI/CD

- **Publishing Workflow Enhancement** (2acd212)
  - Updated `.github/workflows/Publish.yml` with improved configuration

### Code Statistics

- Overall changes: 48 files changed, 1,307 insertions(+), 1,408 deletions(-)
- Net reduction of ~100 lines while improving code quality and maintainability

## Version 0.0.5

This release brings significant performance improvements, new integration methods, enhanced morphology support, expanded documentation, and modernized packaging infrastructure.

### New Features

- **Pallas Kernel Acceleration** (#51)
  - Added Pallas kernel support for voltage solver to accelerate multi-compartment simulations
  - Introduced optimized triangular matrix computation with GPU/CPU backend support
  - Added debug kernels for Pallas backend testing

- **Backward Euler Solver** (#49)
  - Added backward Euler integration method for improved numerical stability
  - Enhanced integration infrastructure with new solver options

- **Morphology Enhancements** (#41, #46, #51)
  - Added support for immutable sections
  - Implemented DHS (Diagonal Hines Solver) support
  - Added lazy loading of networkx for better performance
  - Improved morphology branch tree handling and documentation
  - Enhanced ASC/SWC file support for morphology loading

### Performance Improvements

- **Sodium Channel Integration** (da6697f, 7f91bbe, 7c218f1)
  - Refactored sodium integration from backward Euler to RK4 solver for better accuracy
  - Updated population size handling in simulations
  - Optimized voltage solver performance

- **Integration System Refactoring** (#47)
  - Refactored integrators to get time from `brainstate.environ` for better consistency
  - Streamlined solver logic and improved code structure

### Documentation

- **Expanded Chinese Documentation** (#45)
  - Added comprehensive Chinese language documentation
  - Included advanced tutorial examples and API references

- **New Documentation Structure** (#40, #42)
  - Added quickstart guides, tutorials, and advanced tutorials
  - Reorganized documentation for better navigation
  - Enhanced code documentation and type hints (#44)

### Infrastructure & Dependencies

- **Packaging Modernization**
  - Migrated from `setup.py` to modern `pyproject.toml`-only configuration
  - Updated license format to SPDX identifier (`Apache-2.0`)
  - Improved package metadata and dependency specifications

- **Dependencies**
  - Added `brainpy>=3.0.0` as core dependency
  - Added `braintools>=0.1.0` for enhanced tooling
  - Updated CI/CD configurations for Python 3.13 support

- **CI/CD Updates**
  - Added Python 3.13 support (#50, #48)
  - Updated GitHub Actions: setup-python from 5 to 6, checkout from 4 to 5

### Code Quality

- **Refactoring & Improvements** (#44)
  - Improved external current registration and error handling
  - Enhanced type hints across the codebase
  - Better code organization and readability

### Examples & Testing

- Added linear solver test notebooks
- Enhanced Golgi model simulation examples
- Updated example scripts for better demonstration of features

## Version 0.0.4

Previous release with core functionality.

## Version 0.0.1

The first release of the project.



