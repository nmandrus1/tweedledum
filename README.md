<p align="center">
  <img src="tweedledum.svg" width="128" height="128"/>
</p>

# tweedledum (research fork)

**tweedledum** is a C++ library, with Python bindings, for the synthesis,
compilation and optimization of quantum circuits.  This fork adds a
**Boolean-function compiler** (`tweedledum.bool_function_compiler`) that turns
ordinary, parameterised Python functions into reversible quantum circuits
(oracles) that can be exported to Qiskit.

```python
from tweedledum import BitVec
from tweedledum.bool_function_compiler import QuantumCircuitFunction, circuit_input

@circuit_input(x=lambda n: BitVec(n))
def parity(n: int) -> BitVec(1):
    p = BitVec(1, 0)
    for i in range(n):        # classical loop: unrolled at compile time
        p = p ^ x[i]
    return p

oracle = QuantumCircuitFunction(parity, 4)     # n = 4
print(oracle.simulate(BitVec(4, 0b1011)))      # -> 1
qc = oracle.synthesize_quantum_circuit()       # a qiskit.QuantumCircuit
print(qc)
```

---

## Contents

1. [Install (5 minutes)](#install)
2. [Check that it works](#check-that-it-works)
3. [Running the tests](#running-the-tests)
4. [Developing](#developing)
5. [C++-only build](#c-only-build)
6. [Troubleshooting](#troubleshooting)
7. [Repository layout](#repository-layout)

---

## Install

You need three things: **git**, a **C++17 compiler**, and **Python ≥ 3.10**.
The package contains C++ code, so `pip` compiles it on your machine (this
takes 1–3 minutes).  Everything else is downloaded automatically: CMake,
pybind11, Qiskit and the other Python dependencies come from PyPI, and the C++
dependencies are bundled in `external/`.

> **Windows users:** use [WSL 2](https://learn.microsoft.com/windows/wsl/install)
> (Ubuntu) and follow the Ubuntu instructions inside it.  Native Windows builds
> are not tested.

### Step 1: install the system tools

Pick your operating system:

| OS | Command |
|----|---------|
| **Ubuntu / Debian / WSL** | `sudo apt update && sudo apt install -y build-essential git curl` |
| **Fedora / RHEL** | `sudo dnf install -y gcc-c++ libstdc++-static git curl` |
| **macOS** | `xcode-select --install` (installs Apple Clang and git) |

Compiler versions: GCC ≥ 13, Clang ≥ 18 and recent Apple Clang are tested.
Older compilers with full C++17 support (for example GCC 11 on Ubuntu 22.04)
will probably work, but nobody has checked.

### Step 2: get the code

```bash
git clone https://github.com/nmandrus1/tweedledum.git
cd tweedledum
```

### Step 3: install the Python package

**Option A: uv (recommended).**  [uv](https://docs.astral.sh/uv/) is a fast
Python package manager.  It downloads a suitable Python for you, so you don't
need Python headers or a particular system Python.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (once), then open a new terminal
uv venv --python 3.12                             # creates .venv/
source .venv/bin/activate
uv pip install -e ".[test]"
```

**Option B: plain `venv` + `pip`.**  Uses your system Python.  On Ubuntu you also
need `sudo apt install python3-dev python3-venv`; on Fedora,
`sudo dnf install python3-devel`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[test]"
```

**Option C: conda.**  Works, but read the
[GLIBCXX note](#importerror-glibcxx_3432-not-found) if you are on Fedora.

```bash
conda create -n tweedledum python=3.12 -y
conda activate tweedledum
pip install -e ".[test]"
```

`-e` installs in *editable* mode: edits to the Python files under
`python/tweedledum/` take effect immediately.  `[test]` also installs `pytest`.
If you only want to *use* the library, `pip install .` is enough.

Using it from another project (for example a research repo with its own venv)?
Activate that project's environment and run `pip install -e /path/to/tweedledum`.

## Check that it works

```bash
python -c "import tweedledum; print('tweedledum imported OK')"
python -m pytest            # ~150 tests, takes a few seconds
```

## Running the tests

There are two test suites:

| Suite | Location | What it covers | Command |
|-------|----------|----------------|---------|
| Python | `python/test/` | Boolean-function compiler, bindings, Qiskit converters | `python -m pytest` |
| C++ | `tests/` | Core library: IR, passes, synthesis, mapping (Catch2) | `make test-cpp` |

A `Makefile` wraps the common commands (`make help` lists them):

```bash
make dev              # editable install + test deps (same as step 3)
make test-python      # Python tests
make test-cpp         # configure + build + run the C++ tests (build/cpp/)
make test             # both
make test-python-all  # include the slow tests (wide circuits, a few minutes)
make clean            # delete build artifacts
```

Useful pytest invocations:

```bash
python -m pytest python/test/test_QuantumCircuitFunction.py   # one file
python -m pytest -k clique                                     # tests whose name matches
python -m pytest -x --pdb                                      # stop + debug on first failure
python -m pytest --run-slow                                    # include tests marked slow
```

The C++ test binary can also run a subset directly, by name or tag:

```bash
./build/cpp/tests/run_tests "[xag_synth]"
./build/cpp/tests/run_tests --list-tests
```

## Developing

* **Changed Python code?** Nothing to do; editable installs pick it up.
* **Changed C++ code** (`include/`, `src/`, `python/tweedledum/**/*.cpp`)?
  Re-run `pip install -e ".[test]"` (or `make dev`).  Rebuilds are incremental
  and reuse `build/<python-tag>/`.
* **Switched branches and the build acts strangely?** `make clean` and reinstall.
* Add a Python test by creating `python/test/test_<something>.py`.  Mark slow tests
  with `@pytest.mark.slow`.
* Add a C++ test by creating a file under `tests/` and listing it in
  `tests/CMakeLists.txt`.

CMake options can be passed through pip, for example a debug build:

```bash
pip install -e . -Ccmake.build-type=Debug
```

## C++-only build

You don't need Python for this.  You do need CMake ≥ 3.20 (`pip install cmake`
works if your system one is older).

```bash
cmake -S . -B build/cpp -DTWEEDLEDUM_TESTS=ON      # add -DCMAKE_CXX_STANDARD=20 for C++20
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
```

| CMake option | Default | Meaning |
|--------------|---------|---------|
| `TWEEDLEDUM_TESTS` | `OFF` | Build the Catch2 test suite |
| `TWEEDLEDUM_EXAMPLES` | `OFF` | Build every `examples/*.cpp` into its own executable |
| `TWEEDLEDUM_PYBINDS` | `OFF` (`ON` under pip) | Build the `_tweedledum` Python extension |
| `TWEEDLEDUM_WARNINGS_AS_ERRORS` | `OFF` | Add `-Werror` |
| `TWEEDLEDUM_ENABLE_COVERAGE` | `OFF` | Compile with `--coverage` |
| `TWEEDLEDUM_STATIC_LIBSTDCXX` | `ON` | Link libstdc++ statically into the Python module (Linux/GCC); see below |

To use tweedledum from your own CMake project, `add_subdirectory(tweedledum)` and
link against `tweedledum::tweedledum`.

## Troubleshooting

### `ImportError: ... GLIBCXX_3.4.32 not found`

Your Python (usually conda's) loaded an older `libstdc++` than the compiler
used to build tweedledum.  The build avoids this by linking `libstdc++`
statically when the static library is available.  On Fedora/RHEL that needs
an extra package:

```bash
sudo dnf install libstdc++-static
make clean && pip install -e ".[test]"
```

Alternatively, inside conda: `conda install -c conda-forge libstdcxx-ng`
(installs a newer runtime), or use uv/venv instead of conda.

### `fatal error: Python.h: No such file or directory`

Install the Python headers (`python3-dev` on Ubuntu, `python3-devel` on
Fedora), or use uv (Option A), whose Pythons include them.

### The build is killed or the machine freezes

Compiling uses all CPU cores by default, which needs a few GB of RAM.  Limit
parallelism:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -e ".[test]"
```

### Something else went wrong during `pip install`

pip hides the compiler output by default.  Show it with `pip install -v -e .`
(or `uv pip install -v -e .`).  The first error printed is usually the one that
matters.

## Repository layout

```
include/tweedledum/   C++ headers (IR, operators, passes, synthesis, ...)
src/                  C++ implementation
python/tweedledum/    Python package; *.cpp files there are the pybind11 bindings
  bool_function_compiler/   Python -> logic network -> quantum circuit compiler
  qiskit/                   Converters to/from Qiskit
python/test/          Python tests (pytest)
tests/                C++ tests (Catch2)
examples/             Notebooks and examples
external/             Vendored C++ dependencies (mockturtle, kitty, fmt, Eigen, ...)
```

## Third-party libraries

tweedledum bundles or uses:
[abc](https://github.com/berkeley-abc/abc),
[bill](https://github.com/lsils/bill),
[Catch2](https://github.com/catchorg/Catch2),
[Eigen](https://gitlab.com/libeigen/eigen),
[{fmt}](https://github.com/fmtlib/fmt),
[kitty](https://github.com/msoeken/kitty),
[lorina](https://github.com/hriener/lorina),
[mockturtle](https://github.com/lsils/mockturtle),
[nlohmann/json](https://github.com/nlohmann/json),
[parallel-hashmap](https://github.com/greg7mdp/parallel-hashmap),
[percy](https://github.com/lsils/percy),
[pybind11](https://github.com/pybind/pybind11),
[rang](https://github.com/agauniyal/rang),
[PySAT](https://github.com/pysathq/pysat) and
[Qiskit](https://github.com/Qiskit/qiskit).

## License and credits

MIT licensed; see [LICENSE](LICENSE).  The original tweedledum was written by
Bruno Schmitt as part of the
[EPFL logic synthesis libraries](https://lsi.epfl.ch/page-138455-en.html)
([upstream repository](https://github.com/boschmitt/tweedledum)).
