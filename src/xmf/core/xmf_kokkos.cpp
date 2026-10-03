/* Kokkos lifecycle base (C1a) -- NOT a physics module.

   Owns the one Kokkos::initialize call of the process behind an extern
   "C" idempotent guard: xmf_setup calls xmf_kokkos_ensure(), pytest's
   repeated setup/finalize cycles (test_abi runs 100 in one process) hit
   the guard and return immediately. xmf_finalize deliberately NEVER calls
   Kokkos::finalize -- entering and leaving the runtime every cycle is
   forbidden; the atexit handler registered on the first successful
   initialisation performs the single teardown at process exit -- except
   on Windows since C1b, where the OpenMP pool makes that exit window
   deadlock (see kokkos_finalize_at_exit below; Kokkos 4's finalize is
   optional and the OS reclaims the runtime).

   C1b: CMake pins Kokkos_ENABLE_SERIAL and Kokkos_ENABLE_OPENMP ON (CUDA
   OFF); with both host spaces enabled Kokkos' default execution space is
   OpenMP, so the loops ride OpenMP without ever naming a space (Serial
   stays available as the fallback). The lifecycle is unchanged except
   the Windows exit teardown (kokkos_finalize_at_exit below). */

#include "xmf_kokkos.h"

#include <cstdlib>

namespace {

/* Single teardown at process exit (registered by the first successful
   xmf_kokkos_ensure). Never called from xmf_finalize.

   C1b: on Windows the teardown is skipped. The process-exit CRT/DLL
   teardown window (loader lock held) deadlocks the moment the OpenMP
   pool exists -- Kokkos::finalize runs one last #pragma omp parallel
   region (Kokkos_OpenMP_Instance.cpp, OpenMPInternal::finalize) that the
   pool's worker threads can no longer safely join there, so every run
   with OMP_NUM_THREADS >= 2 would hang AFTER pytest finished. Kokkos 4
   declares Kokkos::finalize optional and the operating system reclaims
   the runtime at process exit anyway, so skipping is the safe behaviour.
   Serial-only runs had no pool and never noticed; other platforms keep
   the teardown. Probes: .tmp/c1b_exit_hang_notes.md. */
void kokkos_finalize_at_exit() {
#if defined(_WIN32)
  /* Deliberate no-op: see the C1b note above. */
#else
  if (Kokkos::is_initialized())
    Kokkos::finalize();
#endif
}

}  // namespace

extern "C" int xmf_kokkos_ensure(void) {
  if (Kokkos::is_initialized())
    return 0; /* idempotent: every setup after the first is a no-op */
  {
    Kokkos::InitializationSettings args; /* defaults: Serial + OpenMP build,
                                            OpenMP is the default execution
                                            space, no device */
    Kokkos::initialize(args);
  }
  if (!Kokkos::is_initialized())
    return -1;
  std::atexit(kokkos_finalize_at_exit);
  return 0;
}
