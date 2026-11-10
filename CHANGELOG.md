# Changelog
All notable changes to this project will be documented in this file.


## [Unreleased]

### Changed (build modernization)
- Builds with current compilers (tested: GCC 15, Clang 21; CI: GCC 13/14,
  Clang 18, Apple Clang) in C++17 and C++20 mode. Added the missing
  `<cstdint>`/`<memory>` includes that GCC 13+ requires.
- pybind11 is no longer vendored; it comes from PyPI as a build requirement
  (pybind11 ≥ 2.13, tested with 3.1).
- Requires Python ≥ 3.10 (tested 3.10–3.13) and Qiskit ≥ 1.0 (tested 1.0 and
  2.6). Dropped the `astunparse` dependency (`ast.unparse` is used instead) and
  declared the previously missing `python-sat` dependency.
- Vendored C++ dependencies are always used instead of whatever the system
  provides, so builds are reproducible.
- The C++ sources are compiled once into a static library that the Python module
  links against, instead of being compiled twice.
- The Python module links libstdc++ statically when possible, fixing
  `GLIBCXX_... not found` import errors under conda.
- Fixed Qiskit 3.0 deprecation warnings in the Qiskit converters.
- Replaced the Nix/conda-specific `dev.sh`/`shell.nix` with a `Makefile`;
  replaced the outdated Travis and GitHub workflows with a single CI workflow.
- Rewrote the README installation instructions.

### Fixed
- `QuantumCircuitFunction.synthesize_quantum_circuit(optimize_xag=False)` no
  longer fails (`xag_synth` needs a network without dangling nodes, so a
  cleanup pass now always runs).

### Tests
- C++ tests are registered with CTest (`ctest --test-dir build/cpp`).
- pytest now finds the Python tests (`python -m pytest` from the repo root).
- New end-to-end tests for `QuantumCircuitFunction` that check compiled logic
  networks and synthesized circuits against brute-force references, plus
  smoke tests for the synthesis bindings, passes and Qiskit converters.
- Added a `slow` marker (`pytest --run-slow`).
- Tests for APIs that no longer exist are skipped with an explanation.


## [1.1.0] - 2021-06-29
In this release there was many cosmetic changes, such as using clang-format on
all sources and restructuring the project's directory to more closely follow the
pitchfork layout conventions.

### Added
- Analysis pass to count operators.
- Analysis pass to compute instructions' ALAP/ASAP layers.
- Analysis pass to compute critical path.
- Analysis pass to cut circuit.
- Gate cancellation pass.
- Pass to inverse (take adjoint) of a circuit.
- Crude QASM 2.0 parser.
- CX-Dihedral synthesis method.
- Bridge operator.
- Bridge decomposition pass.
- Bridge mapping pass.
- Sqrt(X) operator.
- TFC parser.
- Allow user to define var order for expressions.
- Limited support for `arm64`, `ppc64le` and `s390x`.

### Change
- Rename `depth` pass to `compute_depth`.
- Restructure of the code base.
- Make `Cbit`, `Qubit`, and `Instruction` constructors public.
- Invalidate ancilla reference on release.
- Change `Unitary` operator global phase behavior.
- Rename `euler_decomp` to `one_qubit_decomp`.

### Deprecated
- Circuit `size()` method. (Use `num_instructions` instead.)

### Fixed:
- Operator must clone the ConcreteOp class on copy.


## [1.0.0] - 2021-03-29
- Initial stable version. (Before there was only darkness.)


[Unreleased]: https://github.com/boschmitt/tweedledum/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/boschmitt/tweedledum/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/boschmitt/tweedledum/releases/tag/v0.0.1