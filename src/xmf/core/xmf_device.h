/* Device-state owner of the xmf_core library (C2) -- NOT a physics
   module.

   Fields / P_e / the read-only tables (mass_code, charge, mesh metrics,
   beam tables, flattened source tapes) live here as Kokkos Views in the
   DEFAULT memory space of the default execution space: CUDA device
   memory in the WSL/Linux CUDA tree, plain HostSpace in the Windows
   OpenMP tree (there mirror and "device" share one space and every sync
   below reduces to a copy inside HostSpace -- the host tree never
   regresses). The host blocks of XmfRun stay the mirrors; host C only
   touches them at the xmf_dev_* sync points declared in xmf_run.h.

   Views are created by xmf_dev_attach (end of setup) and released by
   xmf_dev_detach (first act of xmf_finalize). They never enter XmfRun
   itself -- the run carries only the opaque void *dev handle -- and
   never outlive the run. This header is C++ only; pure-C translation
   units use the extern "C" hooks alone. */

#ifndef XMF_DEVICE_H
#define XMF_DEVICE_H

#ifdef __cplusplus

#include <Kokkos_Core.hpp>

#include <stddef.h>

#include "xmf_run.h"       /* XMF_SLOT_NVAR / XMF_KB must precede the hpp */
#include "xmf_ir_eval.hpp"

namespace xmf {

/* Owning device-side Views. Default template parameters -> memory space
   of the default execution space (CudaSpace on the CUDA tree, HostSpace
   on the host trees). Extent 0 marks "not loaded / not present". */
struct DeviceState {
  /* Evolved state (mirrors: run->fields, run->pe). */
  Kokkos::View<double *> fields; /* nfluid*5*ncells */
  Kokkos::View<double *> pe;     /* ncells; extent 0 when electrons off */
  /* M block 1 staggered magnetic state (mirrors: run->bf, run->ef);
     allocated/pushed by xmf_dev_mag_alloc. Extent 0 marks the closed
     magnetic gate (xmf_mag_load never landed). */
  Kokkos::View<double *> bf[3]; /* face-centred B, (ntot+1)*ntot*ntot */
  Kokkos::View<double *> ef[3]; /* edge-centred E, ntot*(ntot+1)^2 */
  /* M4 stage-start electron density snapshot (xmf_dev_ne_stage_fill),
     ncells: the charge shares w_s of every ion sweep of one RK stage
     read this frozen n_e instead of the live densities, so a fluid
     swept later never sees the densities of the fluids swept earlier
     in the same stage. Extent 0 marks the closed magnetic gate. */
  Kokkos::View<double *> ne_stage;
  /* Port D: device-side induction scratch (mag_up_snapshot + mag_induce).
     mag_up: charge-weighted ion velocity u_+ = sum q_s n_s u_s / sum q_s n_s
     (3 * ncells, layout [comp*ncells + c], mirrors run->mag_up). Allocated
     by xmf_dev_mag_alloc with the gate, extent 0 = gate closed. The u_+
     snapshot kernel writes this from dv->fields; mag_induce reads it.
     mag_g: cell-centred EMF G = u_+ x B scratch (3 * ncells, same layout).
     Written by the first half of mag_induce, consumed by the edge-average
     and curl passes. Both Views are owned by DeviceState and persist for
     the lifetime of the magnetic gate (no per-stage alloc/free). */
  Kokkos::View<double *> mag_up;
  Kokkos::View<double *> mag_g;
  /* Read-only tables (mirrors: run->mass_code, run->charge). */
  Kokkos::View<double *> mass_code;
  Kokkos::View<double *> charge;
  /* Mesh authority, per direction (mirrors: run->mesh[d]). x/xf/dx/dV
     length ntot (xf length ntot+1); A length ntot+1. */
  Kokkos::View<double *> x[3];
  Kokkos::View<double *> xf[3];
  Kokkos::View<double *> dx[3];
  Kokkos::View<double *> A[3];
  Kokkos::View<double *> dV[3];
  /* Radiation state (mirrors: run->rad members); allocated/pushed by
     xmf_dev_rad_alloc / xmf_dev_rad_sync_absorbers. */
  Kokkos::View<double *> sigma;
  Kokkos::View<double *> hnu;
  Kokkos::View<double *> finf;
  Kokkos::View<double *> flux;       /* nbeam*ncells */
  Kokkos::View<double *> dflux;      /* nbeam*ncells */
  Kokkos::View<double *> sigma_multi; /* nabs*nbeam */
  Kokkos::View<int *> absorbers;      /* nabs */
  /* Flattened source tape set (mirrors: run->tapes), rebuilt by
     xmf_dev_tapes_sync. tape_desc carries six ints per tape:
     dest_kind, dest_fluid, nops, nconst, ops_offset (ints, into
     tape_ops), consts_offset (doubles, into tape_consts). */
  Kokkos::View<int *> tape_desc;
  Kokkos::View<int *> tape_ops;
  Kokkos::View<double *> tape_consts;
  int ntape;
  int nbeam; /* device mirror of run->rad->nbeam (0 without rad) */
  int nabs;  /* device mirror of run->rad->nabs */
  /* Port B: device-side boundary condition state (mirrors: run->user_bc,
     run->bc_tapes, run->mag_user_bc). The BC fill kernels read these on
     the device instead of the host pointers; the host-side loaders
     (xmf_bc_set_user / xmf_bc_load_tapes / xmf_mag_set_user) push here
     after copying into the run. user_bc carries [nfluid*5*nztot*nytot]
     doubles (the floor buffer, NULL extent when no floor is loaded).
     bc_tape_* carry the flattened per-face assignment tapes:
     bc_tape_map[face*nslot + f*5+var] is the tape index or -1 (copy @1
     fallback); bc_tape_desc[t] carries four ints (dest_fluid, dest_var,
     nops, nconst). mag_user_bc carries [2*nztot*nytot] doubles (the
     tangential B floor on x1 faces, NULL extent when no floor loaded). */
  Kokkos::View<double *> user_bc;      /* extent 0 when no floor */
  Kokkos::View<int *> bc_tape_map;     /* [6*nslot], nslot = 5*nfluid */
  Kokkos::View<int *> bc_tape_desc;    /* [4*total_bc_tape] */
  Kokkos::View<int *> bc_tape_ops;    /* [2*total_ops] */
  Kokkos::View<double *> bc_tape_consts; /* [total_consts] */
  int bc_tape_total; /* total BC tapes across all six faces */
  int bc_nslot;      /* 5*nfluid */
  Kokkos::View<double *> mag_user_bc;  /* [2*nztot*nytot]; extent 0 = none */
  /* Port C: frozen-cell pin mask and values (mirrors: run->pin_mask,
     run->pin_vals). The pin-apply kernel reads these on the device
     instead of the host pointers; xmf_bc_set_pin pushes here after
     copying into the run. pin_mask carries ncells ints (0/1, ghosts
     included); pin_vals carries [nfluid*5*ncells] doubles (same flat
     layout as fields). Extent 0 = not loaded. P_e is never pinned;
     B is never pinned (CT path not touched by pin apply). */
  Kokkos::View<int *> pin_mask;
  Kokkos::View<double *> pin_vals;
};

/* The run's device state; NULL before attach / after detach. */
inline DeviceState *dev(const XmfRun *run) {
  return static_cast<DeviceState *>(run->dev);
}

/* Device-pointer IrCtx for the on-device source solve (mirrors the
   host-side builder in xmf_ir.cpp). */
IrCtx ir_ctx_device(const XmfRun *run);

}  // namespace xmf

#endif /* __cplusplus */

#endif /* XMF_DEVICE_H */
