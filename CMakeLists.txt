cmake_minimum_required(VERSION 3.15)
project(xmf LANGUAGES C CXX)

# C99, as the kernel was before (old recipe: gcc -std=c99).
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
# C1a: the hydro/source/scan loops ride the Kokkos interface; Kokkos 4.x
# needs C++17 and MSVC/MinGW must not fall back to a compiler default.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release when the backend/user does not pick a build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- C2 build switches ----------------------------------------------------
# XMF_CUDA=ON builds the single-GPU CUDA tree (WSL, RTX 5080 sm_120): the
# Kokkos DEFAULT execution space becomes Cuda, the Views land in device
# memory and every loop module rides the device. The default stays the
# host tree (Serial + OpenMP) -- the Windows tree must not regress.
# XMF_FMAD controls FMA contraction of the device code: OFF (default) adds
# --fmad=false, the correctness-tree contract; a speed-measurement build
# may flip it ON. The Kokkos version, submodule pin and the opcode set are
# untouched by both switches.
option(XMF_CUDA "Build the CUDA device tree (single GPU)" OFF)
option(XMF_FMAD "Allow FMA contraction in device code" OFF)

if(XMF_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # The card pins Kokkos_ARCH_BLACKWELL120; with Kokkos compiling as a
  # native CMake CUDA language the architecture travels via
  # CUDA_ARCHITECTURES (Kokkos sets it on its own targets; ours needs it
  # explicitly or CMake leaves the target without -gencode and nvcc falls
  # back to a pre-sm_75 default -- wrong binary for the RTX 5080 and the
  # wrong __CUDA_ARCH__ while compiling). sm_120 == architecture 120.
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# ---- Kokkos ----------------------------------------------------------------
# C1-sup: Kokkos is a git submodule under third_party/kokkos, pinned to the
# 4.7.04 commit by the gitlink SHA (never master/develop). Configure no
# longer needs network access -- the source tree is already in the repo.
# Host tree (C1b): Serial AND OpenMP enabled -> the default execution space
# is OpenMP, the loops ride it without ever naming an execution space;
# Serial stays the fallback space. CUDA tree (C2): CUDA on plus the
# BLACKWELL120 arch (sm_120) -> the default execution space is Cuda and
# the host space stays Serial (compute_dt / fill_bc are plain serial C).
# Kokkos_ENABLE_LIBDL is off on the MinGW toolchain only (it ships no
# dlfcn.h; the dlopen-based profiling hooks are not needed anyway).
# Kokkos' own tests/examples stay off. Kokkos derives the architecture of
# its own targets from Kokkos_ARCH_BLACKWELL120; xmf_core picks the same
# sm_120 through CMAKE_CUDA_ARCHITECTURES (set in the XMF_CUDA block
# above) because it compiles as a plain CUDA-language target.
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "C1a: Serial execution space" FORCE)
if(XMF_CUDA)
  set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL
      "C2 CUDA tree: the host space stays Serial" FORCE)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL
      "C2: CUDA device execution space" FORCE)
  set(Kokkos_ARCH_BLACKWELL120 ON CACHE BOOL
      "C2: RTX 5080 (sm_120)" FORCE)
  # Native CMake CUDA language for Kokkos itself (the very mode Kokkos'
  # own CUDA CI configures). The alternative launcher path would exec the
  # bin/kokkos_launch_compiler + nvcc_wrapper bash scripts, which carry
  # CRLF endings in this Windows-autocrlf checkout (env: 'bash\r') and
  # fail under WSL. With the native language Kokkos compiles its own
  # sources through CMAKE_CUDA_COMPILER, matching the xmf_core modules
  # below (LANGUAGE CUDA) -- the same compile language on both sides of
  # the Kokkos::kokkos link, as the upstream comment requires.
  set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL
      "C2 CUDA tree: no nvcc_wrapper bash launcher" FORCE)
else()
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL
      "C1b: OpenMP host execution space" FORCE)
  set(Kokkos_ENABLE_CUDA OFF CACHE BOOL
      "CUDA arrives with -DXMF_CUDA=ON" FORCE)
endif()
if(MINGW)
  set(Kokkos_ENABLE_LIBDL OFF CACHE BOOL
      "no dlfcn.h on this MinGW toolchain" FORCE)
endif()
set(Kokkos_ENABLE_TESTS    OFF CACHE BOOL "no Kokkos unit tests in this build" FORCE)
set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "no Kokkos examples in this build" FORCE)

# C1b: keep the DLL self-contained when OpenMP arrives. Kokkos'
# find_package(OpenMP) below resolves the gomp link to the SHARED import
# library (libgomp.dll.a) and splices that ABSOLUTE path into the link
# line, growing a libgomp-1.dll dependency -- ctypes loads the DLL with
# the default search directories only, so the artifact must carry its own
# OpenMP runtime like the C1a -static embeds libgcc/libstdc++. No link
# OPTION placement can outrank that mid-line operand (options land before
# the objects, and an archive only resolves symbols still unresolved at
# its scan point -- probes in .tmp/c1b_libgomp_link_notes.md), so the
# very same library lookup is pointed at the static libgomp.a. FindOpenMP
# rebuilds OpenMP_CXX_LIBRARIES and the OpenMP::OpenMP_CXX interface from
# this cache entry on every configure. Link-layer resolution only: flags,
# layout and formulas untouched. GNU/MinGW only.
if(MINGW AND Kokkos_ENABLE_OPENMP)
  find_library(XMF_GOMP_STATIC NAMES libgomp.a
               DOC "static libgomp for the self-contained xmf_core.dll")
  if(XMF_GOMP_STATIC)
    set(OpenMP_gomp_LIBRARY "${XMF_GOMP_STATIC}" CACHE FILEPATH
        "Path to the gomp library for OpenMP" FORCE)
  endif()
  # gcc's own libgomp.spec adds "-ldl" to static OpenMP links: the static
  # libgomp.a drags in offload/profiling members that want the dlfcn
  # symbols. Locate the static libdl.a (LoadLibrary-based) here; the
  # whole-archive link option below embeds it. Neither a raw -ldl nor a
  # plain archive operand can work: CMake sorts both ahead of Kokkos'
  # libraries, where no dlopen/dlsym symbol is unresolved yet, and an
  # archive only resolves symbols unresolved at its scan point. The
  # target lib directory (x86_64-w64-mingw32/lib) is not in find_library's
  # default search list, so derive it from the compiler's location.
  get_filename_component(_XMF_TC_ROOT "${CMAKE_C_COMPILER}" DIRECTORY)
  get_filename_component(_XMF_TC_ROOT "${_XMF_TC_ROOT}" DIRECTORY)
  find_library(XMF_DL_STATIC NAMES libdl.a
               PATHS "${_XMF_TC_ROOT}/x86_64-w64-mingw32/lib"
               DOC "static libdl needed by the static libgomp.a")
endif()

add_subdirectory(third_party/kokkos)

# Shared library, not a CPython extension: no PyInit_, no python3xx.lib link.
# xmf_core.h marks the four-step API with __declspec(dllexport) on Windows
# (and extern "C" everywhere -- the kernel mixes C translation units with
# C++ ones), so the xmf_* symbols are exported unmangled on all toolchains
# CMake may detect.
# Output name follows the platform default: xmf_core.dll on Windows,
# libxmf_core.so on Linux — exactly what xmf/_native.py loads.
# A4.5 split: the kernel is cut into module translation units sharing the
# internal header xmf_run.h (XmfRun definition, slot offsets, cidx/fld,
# internal prototypes). xmf_core.c keeps only the four-step API; the public
# surface stays xmf_core.h alone. A4.6 adds xmf_ir.c: the source-tape
# stack machine and the xmf_ir_load / xmf_ir_apply / xmf_ir_ntape helpers
# (numeric contract in xmf_ir.h); xmf_step never calls it. A4.7 switches
# the stack to dual numbers and adds xmf_ir_jac / xmf_ir_nstate /
# xmf_ir_eval_tape; the opcode set 0..9 stays frozen. A5 adds the
# radiation line scan (xmf_scan) and the semi-implicit source step
# (xmf_source) as their own translation units; xmf_step orchestrates
# scan -> hydro -> source per RK stage, and xmf_hydro gains the
# spherical x1 finite-volume sweep (cartesian dt/dx stays bit-for-bit).
# C1a: xmf_hydro / xmf_scan / xmf_source became .cpp -- their loops run on
# the Kokkos interface (parallel_for over transverse lines / active cells /
# independent rays; the scan's i-walk stays a sequential inward prefix
# product per ray). xmf_kokkos.cpp is the lifecycle base ONLY: the
# idempotent Kokkos::initialize guard plus the Host View helpers -- not a
# physics module. The formulas stay in their line/cell kernels verbatim.
add_library(xmf_core SHARED
  src/xmf/core/xmf_core.c
  src/xmf/core/xmf_mesh.c
  src/xmf/core/xmf_state.c
  src/xmf/core/xmf_bc.c
  src/xmf/core/xmf_mag.c
  src/xmf/core/xmf_ir.cpp
  src/xmf/core/xmf_kokkos.cpp
  src/xmf/core/xmf_device.cpp
  src/xmf/core/xmf_hydro.cpp
  src/xmf/core/xmf_scan.cpp
  src/xmf/core/xmf_source.cpp)
target_include_directories(xmf_core PRIVATE src/xmf/core)
if(XMF_CUDA)
  # C2: the .cpp modules carry device kernels -- compile them with nvcc
  # (CUDA separate compilation is NOT enabled, so every device-callable
  # helper is header-inline (xmf_ir_eval.hpp) or file-static; xmf_core.c
  # stays pure C and reaches the device only through the extern "C"
  # xmf_dev_* hooks). The correctness tree freezes FMA contraction.
  set_source_files_properties(
      src/xmf/core/xmf_ir.cpp
      src/xmf/core/xmf_kokkos.cpp
      src/xmf/core/xmf_device.cpp
      src/xmf/core/xmf_hydro.cpp
      src/xmf/core/xmf_scan.cpp
      src/xmf/core/xmf_source.cpp
      PROPERTIES LANGUAGE CUDA)
  if(NOT XMF_FMAD)
    target_compile_options(xmf_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endif()
endif()
# C1b: the static libdl.a via --whole-archive (see the Kokkos block: gcc's
# libgomp.spec "%{static: -ldl}" does the same for static OpenMP links).
# Whole-archive pulls every member regardless of position on the link line,
# so CMake sorting the option ahead of Kokkos' libraries does not matter;
# the dlfcn symbols are already defined when libgomp.a later needs them.
target_link_libraries(xmf_core PRIVATE Kokkos::kokkos)
if(MINGW AND Kokkos_ENABLE_OPENMP AND XMF_DL_STATIC)
  target_link_options(xmf_core PRIVATE
      "SHELL:-Wl,--whole-archive ${XMF_DL_STATIC} -Wl,--no-whole-archive")
endif()

# The GNU toolchain prefixes shared libraries with "lib" even on Windows;
# keep the canonical name xmf_core.dll there (MSVC has no prefix anyway) so
# xmf/_native.py finds it. Linux keeps the default libxmf_core.so.
if(WIN32)
  set_target_properties(xmf_core PROPERTIES PREFIX "")
endif()

# Optimization level of the old in-loop recipe (-O2); MSVC's /O2 matches its
# Release default. Restricted to the C/CXX sources: in the CUDA tree the
# .cpp modules compile as CUDA and keep nvcc's own Release level.
if(MSVC)
  target_compile_options(xmf_core PRIVATE /O2)
else()
  target_compile_options(xmf_core PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-O2>)
  if(MINGW)
    # C1a: the C++ translation units pull in the MinGW C++ runtime. ctypes
    # loads the DLL with the default search directories only (no PATH), so
    # libgcc/libstdc++/libwinpthread must not stay shared-library
    # dependencies -- embed them and keep the DLL self-contained like the
    # pre-Kokkos build.
    # C1b: with the gomp lookup pointed at the static archive above (see
    # the Kokkos block), the OpenMP runtime is embedded by the same
    # -static, and no shared libgomp-1.dll dependency leaks into the
    # artifact. The CUDA (Linux) tree keeps the shared runtime.
    target_link_options(xmf_core PRIVATE
        -static-libgcc -static-libstdc++ -static)
  endif()
endif()

# Install into the xmf package directory inside the wheel, next to
# _native.py. RUNTIME covers the Windows DLL, LIBRARY the Linux .so.
# The import library (ARCHIVE) is deliberately not installed.
install(TARGETS xmf_core
  LIBRARY DESTINATION xmf
  RUNTIME DESTINATION xmf)
