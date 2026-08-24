# Fabrica Installation Guide

Supplementary install notes for issues not covered (or not fully covered) by
`README.md`. This file exists because the stock `pip install ./simulation`
step fails out of the box on a fresh Ubuntu 20.04 machine — the steps below
are the exact fixes that got it working.

Add new sections as new install issues get solved, following the same
`## Problem` / cause / fix format so this stays useful over time.

## Environment

Tested on Ubuntu 20.04 (focal), conda env `fabrica`, Python 3.10.

```bash
conda activate fabrica
```

## Quick start (`pip install ./simulation`)

Run these in order from the `Fabrica` repo root. Each one fixes a specific
build failure — see the sections below for why.

```bash
# 1. CMake itself — see "CMake: Permission denied" below before skipping this
conda install -y -c conda-forge cmake

# 2. C/C++ toolchain
sudo apt update && sudo apt install -y build-essential

# 3. OpenGL + X11 dev headers (needed by CMake's find_package(OpenGL) and by GLFW)
sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev libxrandr-dev \
    libxinerama-dev libxcursor-dev libxi-dev libx11-dev libxext-dev

# 4. Missing submodules (pybind11, libigl) — see "Broken submodules" below
cd simulation/externals
git clone https://github.com/pybind/pybind11.git pybind11
git -C pybind11 checkout ee2b5226295d67b690faddd446a329bb2840a1a8
git clone https://github.com/libigl/libigl.git libigl
git -C libigl checkout 87a550af22fc4af210af40f1d61d81594bbaf546
cd ../..

# 5. X11 XF86VidMode extension (link-time dependency of GLFW)
sudo apt install -y libxxf86vm-dev

# 6. Now the build succeeds
pip install ./simulation
```

## Problem: `PermissionError: [Errno 13] Permission denied: 'cmake'`

**Cause:** two things stack up on a fresh machine. First, `cmake` usually
isn't installed anywhere at all yet (not via apt/conda/pip — check with
`which cmake`, `conda list cmake`, `dpkg -l | grep cmake`). Second, on WSL,
`PATH` inherits Windows' `PATH`, which typically includes a
`.../AppData/Local/Microsoft/WindowsApps` directory — and if that directory
belongs to a *different* Windows user profile than the one WSL is running
under (check the username in the path), WSL can't access it at all. When
Python searches `PATH` for `cmake` and hits that inaccessible directory
first, the search aborts with `PermissionError` right there instead of
continuing on to report a plain "not found" — so this error shows up even
though the real problem is simply that cmake was never installed.

**Fix:** install cmake into the conda env. `<conda prefix>/bin` is always the
first entry in `PATH` when the env is activated, so this guarantees cmake is
found before the search ever reaches the broken WindowsApps directory:
```bash
conda install -y -c conda-forge cmake
```

## Problem: `No CMAKE_C_COMPILER could be found`

**Cause:** no C/C++ compiler installed at all (`gcc`/`g++` missing).

**Fix:**
```bash
sudo apt update && sudo apt install -y build-essential
```

## Problem: `Could NOT find OpenGL` / GLFW `find_package(X11 REQUIRED)` fails

**Cause:** CMake's `find_package(OpenGL REQUIRED)` (in `simulation/CMakeLists.txt`)
and GLFW's own `find_package(X11 REQUIRED)` (vendored under
`simulation/externals/glfw`) both need the corresponding `-dev` headers, not
just runtime libraries.

**Fix:**
```bash
sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev libxrandr-dev \
    libxinerama-dev libxcursor-dev libxi-dev libx11-dev libxext-dev
```

## Problem: `add_subdirectory given source "libigl"/"pybind11" which is not an existing directory`

**Cause:** this is an **upstream bug**, not a local setup issue. The repo's
`.gitmodules` declares `simulation/externals/pybind11` and
`simulation/externals/libigl` as git submodules, but the corresponding
commit pointers were never actually committed to the repo tree — confirmed
by checking `origin/main` directly, so a plain
`git submodule update --init --recursive` is a no-op and won't fix this.

**Fix:** clone the two dependencies manually. The exact commits below were
recovered from `yunshengtian/Assemble-Them-All` (the predecessor project,
whose `simulation/` tree is otherwise byte-identical to this one, submodule
pins included) — using these rather than "latest" avoids any
version-compatibility surprises with this specific, older CMake setup.

```bash
cd simulation/externals

git clone https://github.com/pybind/pybind11.git pybind11
git -C pybind11 checkout ee2b5226295d67b690faddd446a329bb2840a1a8

git clone https://github.com/libigl/libigl.git libigl
git -C libigl checkout 87a550af22fc4af210af40f1d61d81594bbaf546

cd ../..
```

## Problem: `/bin/ld: cannot find -lXxf86vm`

**Cause:** GLFW links against the X11 XF86VidMode extension at build time;
the dev package providing `libXxf86vm.so` was missing.

**Fix:**
```bash
sudo apt install -y libxxf86vm-dev
```

## Quick start (`pip install ./rendering`)

Run these in order from the `Fabrica` repo root. Requires the `simulation`
build's toolchain steps (1) to already be done.

```bash
# 1. eigen3 (pkg-config) + a newer compiler than the simulation build needs
sudo apt install -y libeigen3-dev gcc-10 g++-10

# 2. bpy has no cp310 wheel on plain PyPI anymore — get it from Blender's own index
pip install --extra-index-url https://download.blender.org/pypi/ bpy==4.0.0

# 3. mathutils' latest release needs a newer CPython than 3.10 — pin an older one
CC=gcc-10 CXX=g++-10 pip install mathutils==3.3.0

# 4. Now the build succeeds
CC=gcc-10 CXX=g++-10 pip install ./rendering
```

## Problem: `ERROR: No matching distribution found for bpy`

**Cause:** `rendering/setup.py` requires `bpy` unpinned. Current PyPI releases
of `bpy` (4.1.0 and up) only publish `cp311`/`cp313` wheels — Python 3.10
support was dropped. The `fabrica` conda env is Python 3.10, so pip finds no
installable wheel at all, regardless of network setup.

**Fix:** install from Blender's own wheel index instead, which still hosts
`cp310` builds up through `bpy==4.0.0`. Use `--extra-index-url` (not
`--index-url`) so pip still has plain PyPI available for `bpy`'s own
dependencies (`zstandard`, `cython`, `requests`):

```bash
pip install --extra-index-url https://download.blender.org/pypi/ bpy==4.0.0
```

## Problem: `Package eigen3 was not found in the pkg-config search path`

**Cause:** `mathutils` (a standalone PyPI package with its own C extension,
required by `rendering/setup.py`) needs Eigen's pkg-config file at build time.

**Fix:**
```bash
sudo apt install -y libeigen3-dev
```

## Problem: `g++: error: unrecognized command line option '-std=c++20'`

**Cause:** `mathutils`'s C++ sources are compiled with `-std=c++20`, but the
system default compiler is GCC 9 (installed earlier for the `simulation`
build, which only needs C++11) — GCC 9 doesn't recognize that flag spelling
(only `-std=c++2a`; the `c++20` alias arrived in GCC 10).

**Fix:** install GCC 10 alongside the existing GCC 9 (don't replace the
default — the `simulation` build already works with GCC 9) and scope it to
just this build via env vars:
```bash
sudo apt install -y gcc-10 g++-10
CC=gcc-10 CXX=g++-10 pip install mathutils==3.3.0   # see next entry for the version pin
```

## Problem: `'PyLong_AsInt' was not declared in this scope`

**Cause:** unpinned `mathutils` resolves to the latest release (5.1.0, tied
to Blender 5.1's source tree), whose C code calls the public CPython API
`PyLong_AsInt`. Python 3.10's headers only expose the private
`_PyLong_AsInt` — that public symbol doesn't exist yet in 3.10. Since `bpy`
was pinned to the `4.0` release train (the newest one with a `cp310` wheel),
`mathutils` needs to be pinned to a version from around the same era.

**Fix:** pin an older `mathutils` version compatible with Python 3.10.
`4.0.0` isn't published for `mathutils` on PyPI, so the closest available is
`3.3.0`:
```bash
CC=gcc-10 CXX=g++-10 pip install mathutils==3.3.0
```

## Other environment setup (from README, for reference)

Python dependencies, not part of the `simulation` C++ build above:

```bash
sudo apt-get install graphviz graphviz-dev   # needed by pygraphviz
pip install -r requirements.txt
```

## Verifying the install

```bash
python simulation/test/test_simple_sim.py --model box/box_stack
```

---

<!--
Template for adding a new entry:

## Problem: <short description / key error line>

**Cause:** <why it happens>

**Fix:**
```bash
<commands>
```
-->