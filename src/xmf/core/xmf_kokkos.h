/* Kokkos lifecycle base (C1a) -- NOT a physics module.

   This header carries exactly two things:

   1. the extern "C" idempotent initialisation guard xmf_kokkos_ensure(),
      callable from the pure-C xmf_core.c (xmf_setup calls it once per
      setup; the guard makes repeated setup/finalize cycles in one process
      -- pytest does 100 of them -- a no-op after the first hit).
      xmf_finalize NEVER calls Kokkos::finalize; teardown happens at
      process exit (atexit handler registered by the first ensure) --
      skipped on Windows since C1b, where the OpenMP pool would deadlock
      the exit teardown (Kokkos 4 finalize is optional; xmf_kokkos.cpp).

   2. C++-only Host View helpers: thin factories for non-owning
      Kokkos::View<double*, HostSpace> wrapping the kernel's existing
      malloc buffers. Views built with these helpers live on the loop
      stack only -- they are never stored in XmfRun and never outlive the
      buffer they wrap.

   The loops themselves stay in their modules (xmf_hydro / xmf_source /
   xmf_scan); nothing here sweeps, scans or solves anything. */

#ifndef XMF_KOKKOS_H
#define XMF_KOKKOS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Idempotent Kokkos::initialize guard. Returns 0 once the Kokkos runtime
   is ready -- default execution space OpenMP since C1b, Serial stays
   enabled (already initialised counts as ready), -1 on initialisation
   failure. Safe to call from C. */
int xmf_kokkos_ensure(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace xmf {

/* Non-owning HostSpace views over caller-owned double buffers. The
   MemoryUnmanaged trait makes the wrapping explicit: the view never
   allocates and never frees -- the kernel's malloc blocks stay the only
   owners. */
using HostView1 =
    Kokkos::View<double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostView2 =
    Kokkos::View<double **, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

inline HostView1 host_view(double *p, std::size_t n0) {
  return HostView1(p, n0);
}

inline HostView2 host_view(double *p, std::size_t n0, std::size_t n1) {
  return HostView2(p, n0, n1);
}

}  // namespace xmf

#endif /* __cplusplus */

#endif /* XMF_KOKKOS_H */
