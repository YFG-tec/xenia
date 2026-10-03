/* Device-state owner of the xmf_core library (C2) -- NOT a physics
   module. Implements the extern "C" xmf_dev_* hooks declared in
   xmf_run.h: attach/detach of the owning device Views, the
   SyncToDevice / SyncFromDevice mirror transfers at the locked sync
   points, the radiation / tape device mirrors and the device-side
   stage-buffer allocator (the device twin of dvec).

   The Views live in the DEFAULT memory space of the default execution
   space: CUDA device memory in the CUDA tree, HostSpace in the host
   trees (Windows OpenMP) -- there every sync reduces to a HostSpace
   copy and the host numerics stay bit-for-bit untouched. No sweep,
   scan or solve lives here; the kernels stay in their modules. */

#include "xmf_device.h"

#include <new>
#include <vector>

namespace {

using xmf::DeviceState;
using xmf::dev;

/* Unmanaged host views for the mirror transfers (const source on push,
   mutable destination on pull). */
using HostCV1 =
    Kokkos::View<const double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostV1 = Kokkos::View<double *, Kokkos::HostSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostCIV1 =
    Kokkos::View<const int *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

void push_d(const Kokkos::View<double *> &dv, const double *hp, size_t n) {
  if (n == 0 || dv.extent(0) != n)
    return;
  Kokkos::deep_copy(dv, HostCV1(hp, n));
}

void pull_d(const Kokkos::View<double *> &dv, double *hp, size_t n) {
  if (n == 0 || dv.extent(0) != n)
    return;
  Kokkos::deep_copy(HostV1(hp, n), dv);
}

void push_i(const Kokkos::View<int *> &dv, const int *hp, size_t n) {
  if (n == 0 || dv.extent(0) != n)
    return;
  Kokkos::deep_copy(dv, HostCIV1(hp, n));
}

/* One fluid/slot block offset in the flat [fluid][slot][cell] layout. */
size_t block_off(int f, int var, size_t ncells) {
  return (((size_t)f * XMF_SLOT_NVAR + (size_t)var) * ncells);
}

}  // namespace

extern "C" int xmf_dev_attach(XmfRun *run) {
  if (!run)
    return -1;
  if (run->dev)
    return 0; /* idempotent */
  try {
    DeviceState *dv = new DeviceState();
    size_t ncells = run->ncells;
    size_t nf5 = (size_t)run->nfluid * XMF_SLOT_NVAR;
    int d;
    dv->fields = Kokkos::View<double *>("xmf_fields", nf5 * ncells);
    if (run->pe)
      dv->pe = Kokkos::View<double *>("xmf_pe", ncells);
    dv->mass_code = Kokkos::View<double *>("xmf_mass_code", run->nfluid);
    dv->charge = Kokkos::View<double *>("xmf_charge", run->nfluid);
    for (d = 0; d < 3; d++) {
      int ntot = run->mesh[d].ntot;
      dv->x[d] = Kokkos::View<double *>("xmf_x", ntot);
      dv->xf[d] = Kokkos::View<double *>("xmf_xf", (size_t)ntot + 1);
      dv->dx[d] = Kokkos::View<double *>("xmf_dx", ntot);
      dv->A[d] = Kokkos::View<double *>("xmf_A", (size_t)ntot + 1);
      dv->dV[d] = Kokkos::View<double *>("xmf_dV", ntot);
    }
    dv->ntape = 0;
    dv->nbeam = 0;
    dv->nabs = 0;
    run->dev = dv;
  } catch (const std::exception &) {
    return -1;
  }
  /* Push the setup-time mirrors: fields + P_e (initial condition +
     fill_bc already applied), the default tables and the mesh arrays.
     Radiation and tapes arrive through their own load ports later. */
  xmf_dev_push_tables(run);
  xmf_dev_push(run);
  {
    DeviceState *dv = dev(run);
    for (int d = 0; d < 3; d++) {
      int ntot = run->mesh[d].ntot;
      push_d(dv->x[d], run->mesh[d].x, (size_t)ntot);
      push_d(dv->xf[d], run->mesh[d].xf, (size_t)ntot + 1);
      push_d(dv->dx[d], run->mesh[d].dx, (size_t)ntot);
      push_d(dv->A[d], run->mesh[d].A, (size_t)ntot + 1);
      push_d(dv->dV[d], run->mesh[d].dV, (size_t)ntot);
    }
  }
  return 0;
}

extern "C" void xmf_dev_detach(XmfRun *run) {
  if (!run || !run->dev)
    return;
  delete dev(run); /* the owning Views release their allocations; the
                      Kokkos runtime stays alive (finalize is the
                      process-exit handler's job, never xmf_finalize's) */
  run->dev = NULL;
}

extern "C" void xmf_dev_push(const XmfRun *run) {
  if (!run)
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  push_d(dv->fields, run->fields,
         (size_t)run->nfluid * XMF_SLOT_NVAR * run->ncells);
  if (run->pe)
    push_d(dv->pe, run->pe, run->ncells);
  /* M block 1: the magnetic mirrors ride the same locked sync points;
     a no-op while the gate is closed. */
  xmf_dev_push_b(run);
}

extern "C" void xmf_dev_pull(const XmfRun *run) {
  if (!run)
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  pull_d(dv->fields, run->fields,
         (size_t)run->nfluid * XMF_SLOT_NVAR * run->ncells);
  if (run->pe)
    pull_d(dv->pe, run->pe, run->ncells);
  xmf_dev_pull_b(run); /* M block 1; no-op while the gate is closed */
}

/* ---- M block 1 magnetic mirrors -----------------------------------------
   The six staggered blocks (bf/ef) sync as a group; the lengths come
   from xmf_mag_len_b / xmf_mag_len_e (xmf_mag.c). Every hook is a no-op
   with the gate closed (bf[0] NULL) or before attach / after detach. */

extern "C" int xmf_dev_mag_alloc(XmfRun *run) {
  if (!run)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0; /* host-only stage; nothing to mirror */
  if (!run->bf[0])
    return 0;
  try {
    for (int d = 0; d < 3; d++) {
      size_t nb = xmf_mag_len_b(run, d);
      size_t ne = xmf_mag_len_e(run, d);
      dv->bf[d] = Kokkos::View<double *>("xmf_bf", nb);
      dv->ef[d] = Kokkos::View<double *>("xmf_ef", ne);
      /* Push the host content: B arrives with its initial values, E
         calloc'd zero (the mirror contract stays honest). */
      push_d(dv->bf[d], run->bf[d], nb);
      push_d(dv->ef[d], run->ef[d], ne);
    }
    /* M4: the stage-start n_e snapshot beside the magnetic blocks (one
       double per cell, ghosts included); filled by
       xmf_dev_ne_stage_fill at every RK stage start. */
    dv->ne_stage = Kokkos::View<double *>("xmf_ne_stage", run->ncells);
    /* Port D: u_+ snapshot and G scratch for the device induction path.
       Both persist for the gate lifetime; the kernels write them every
       stage (no per-stage alloc/free). mag_up mirrors run->mag_up;
       mag_g is the cell-centred EMF scratch consumed by the edge average
       and curl passes. */
    dv->mag_up = Kokkos::View<double *>("xmf_mag_up", 3 * run->ncells);
    dv->mag_g = Kokkos::View<double *>("xmf_mag_g", 3 * run->ncells);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

/* M4: freeze the stage-start electron density n_e = sum_f q_s rho_s / m_s
   into dv->ne_stage. The ion MHD sweeps of one RK stage read this
   snapshot for their charge shares w_s = q_s n_s / n_e instead of the
   live densities -- the sweeps run sequentially in place, so without the
   freeze a fluid swept later would read the already-updated densities of
   the fluids swept earlier in the same stage (and two identically
   initialised ions would part ways). The formula and accumulation order
   are exactly hcell_ne's, so the snapshot is bit-for-bit the live value
   at the fill instant. Gate-closed / host-only / pre-attach: no-op. */
extern "C" int xmf_dev_ne_stage_fill(const XmfRun *run) {
  if (!run || !run->bf[0])
    return 0;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0;
  try {
    double *ne = dv->ne_stage.data();
    const double *fld_d = dv->fields.data();
    const double *charge = dv->charge.data();
    const double *mass = dv->mass_code.data();
    int nf = run->nfluid;
    size_t ncells = run->ncells;
    Kokkos::parallel_for(
        "xmf_dev_ne_stage_fill", ncells, KOKKOS_LAMBDA(const size_t c) {
          double s = 0.0;
          for (int f = 0; f < nf; f++)
            s += charge[f] *
                 fld_d[((size_t)f * XMF_SLOT_NVAR + XMF_SLOT_RHO) * ncells +
                       c] /
                 mass[f];
          ne[c] = s;
        });
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

extern "C" void xmf_dev_push_b(const XmfRun *run) {
  if (!run || !run->bf[0])
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  for (int d = 0; d < 3; d++) {
    push_d(dv->bf[d], run->bf[d], xmf_mag_len_b(run, d));
    push_d(dv->ef[d], run->ef[d], xmf_mag_len_e(run, d));
  }
}

extern "C" void xmf_dev_pull_b(const XmfRun *run) {
  if (!run || !run->bf[0])
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  for (int d = 0; d < 3; d++) {
    pull_d(dv->bf[d], run->bf[d], xmf_mag_len_b(run, d));
    pull_d(dv->ef[d], run->ef[d], xmf_mag_len_e(run, d));
  }
}

/* M4 B SSP stage hooks: the stage buffers live in the default memory
   space (xmf_dev_dvec), so the transfers use default-space unmanaged
   views -- NOT the HostSpace mirror helpers above (the buffers would
   live in CudaSpace on the CUDA tree). The fluid cons_combine never
   touches the magnetic blocks; these two hooks are the B twins of the
   fields_to_cons snapshot and cons_combine. */

extern "C" int xmf_dev_mag_stage_snapshot(const XmfRun *run, void **dst) {
  if (!run || !run->bf[0] || !dst)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0; /* host-only stage: nothing to mirror */
  try {
    for (int d = 0; d < 3; d++) {
      size_t n = xmf_mag_len_b(run, d);
      Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out(
          static_cast<double *>(dst[d]), n);
      Kokkos::deep_copy(out, dv->bf[d]);
    }
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

extern "C" int xmf_dev_mag_stage_combine(const XmfRun *run, double a,
                                         void **src0, double b) {
  if (!run || !run->bf[0] || !src0)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0; /* host-only stage: nothing to combine */
  try {
    for (int d = 0; d < 3; d++) {
      size_t n = xmf_mag_len_b(run, d);
      double *bd = dv->bf[d].data();
      const double *s0 = static_cast<const double *>(src0[d]);
      Kokkos::parallel_for(
          "xmf_mag_stage_combine", n, KOKKOS_LAMBDA(const size_t i) {
            bd[i] = a * s0[i] + b * bd[i];
          });
    }
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

extern "C" void xmf_dev_push_field(const XmfRun *run, int f, int var) {
  if (!run)
    return;
  DeviceState *dv = dev(run);
  if (!dv || f < 0 || f >= run->nfluid || var < 0 || var >= XMF_SLOT_NVAR)
    return;
  size_t off = block_off(f, var, run->ncells);
  auto blk =
      Kokkos::subview(dv->fields, Kokkos::make_pair(off, off + run->ncells));
  Kokkos::deep_copy(blk, HostCV1(run->fields + off, run->ncells));
}

extern "C" void xmf_dev_pull_field(const XmfRun *run, int f, int var) {
  if (!run)
    return;
  DeviceState *dv = dev(run);
  if (!dv || f < 0 || f >= run->nfluid || var < 0 || var >= XMF_SLOT_NVAR)
    return;
  size_t off = block_off(f, var, run->ncells);
  auto blk =
      Kokkos::subview(dv->fields, Kokkos::make_pair(off, off + run->ncells));
  Kokkos::deep_copy(HostV1(run->fields + off, run->ncells), blk);
}

extern "C" void xmf_dev_push_pe(const XmfRun *run) {
  if (!run || !run->pe)
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  push_d(dv->pe, run->pe, run->ncells);
}

extern "C" void xmf_dev_pull_pe(const XmfRun *run) {
  if (!run || !run->pe)
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  pull_d(dv->pe, run->pe, run->ncells);
}

extern "C" void xmf_dev_push_tables(const XmfRun *run) {
  if (!run)
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  push_d(dv->mass_code, run->mass_code, (size_t)run->nfluid);
  push_d(dv->charge, run->charge, (size_t)run->nfluid);
}

extern "C" void xmf_dev_pull_flux(const XmfRun *run) {
  if (!run || !run->rad)
    return;
  DeviceState *dv = dev(run);
  if (!dv)
    return;
  pull_d(dv->flux, run->rad->flux, (size_t)run->rad->nbeam * run->ncells);
}

extern "C" int xmf_dev_rad_alloc(XmfRun *run) {
  if (!run)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0; /* host-only stage; nothing to mirror */
  if (!run->rad)
    return 0;
  try {
    XmfRadiation *rad = run->rad;
    size_t nb = (size_t)rad->nbeam;
    size_t tot = nb * run->ncells;
    dv->sigma = Kokkos::View<double *>("xmf_sigma", nb);
    dv->hnu = Kokkos::View<double *>("xmf_hnu", nb);
    dv->finf = Kokkos::View<double *>("xmf_finf", nb);
    dv->flux = Kokkos::View<double *>("xmf_flux", tot);
    dv->dflux = Kokkos::View<double *>("xmf_dflux", tot);
    dv->sigma_multi = Kokkos::View<double *>(); /* dropped until B3 load */
    dv->absorbers = Kokkos::View<int *>();
    dv->nbeam = rad->nbeam;
    dv->nabs = rad->nabs;
    push_d(dv->sigma, rad->sigma, nb);
    push_d(dv->hnu, rad->hnu, nb);
    push_d(dv->finf, rad->finf, nb);
    /* flux/dflux arrive zero at xmf_rad_load time (dvec'd); pushing the
       host content keeps the mirror contract honest. */
    push_d(dv->flux, rad->flux, tot);
    push_d(dv->dflux, rad->dflux, tot);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

extern "C" int xmf_dev_rad_sync_absorbers(XmfRun *run) {
  if (!run)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0;
  if (!run->rad || run->rad->nabs < 2 || !run->rad->absorbers ||
      !run->rad->sigma_multi)
    return 0;
  try {
    XmfRadiation *rad = run->rad;
    size_t na = (size_t)rad->nabs;
    dv->sigma_multi =
        Kokkos::View<double *>("xmf_sigma_multi", na * (size_t)rad->nbeam);
    dv->absorbers = Kokkos::View<int *>("xmf_absorbers", na);
    dv->nabs = rad->nabs;
    push_d(dv->sigma_multi, rad->sigma_multi, na * (size_t)rad->nbeam);
    push_i(dv->absorbers, rad->absorbers, na);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

extern "C" int xmf_dev_tapes_sync(XmfRun *run) {
  if (!run)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0;
  try {
    int ntape = run->ntape;
    size_t tot_ops = 0, tot_cst = 0;
    int t;
    for (t = 0; t < ntape; t++) {
      tot_ops += 2 * (size_t)run->tapes[t].nops;
      tot_cst += (size_t)run->tapes[t].nconst;
    }
    dv->tape_desc = Kokkos::View<int *>("xmf_tape_desc", 6 * (size_t)ntape);
    dv->tape_ops = Kokkos::View<int *>("xmf_tape_ops", tot_ops);
    dv->tape_consts = Kokkos::View<double *>("xmf_tape_consts", tot_cst);
    dv->ntape = ntape;
    if (ntape == 0)
      return 0;
    /* Flatten the host tape set, then one transfer per array. */
    std::vector<int> desc(6 * (size_t)ntape);
    std::vector<int> opsflat(tot_ops);
    std::vector<double> cstflat(tot_cst);
    size_t off_ops = 0, off_cst = 0;
    for (t = 0; t < ntape; t++) {
      const XmfTape *tp = &run->tapes[t];
      size_t o = 6 * (size_t)t;
      desc[o + 0] = tp->dest_kind;
      desc[o + 1] = tp->dest_fluid;
      desc[o + 2] = tp->nops;
      desc[o + 3] = tp->nconst;
      desc[o + 4] = (int)off_ops;
      desc[o + 5] = (int)off_cst;
      for (int p = 0; p < tp->nops; p++) {
        opsflat[off_ops + 2 * (size_t)p] = tp->ops[2 * p];
        opsflat[off_ops + 2 * (size_t)p + 1] = tp->ops[2 * p + 1];
      }
      for (int q = 0; q < tp->nconst; q++)
        cstflat[off_cst + (size_t)q] = tp->consts[q];
      off_ops += 2 * (size_t)tp->nops;
      off_cst += (size_t)tp->nconst;
    }
    push_i(dv->tape_desc, desc.data(), 6 * (size_t)ntape);
    push_i(dv->tape_ops, opsflat.data(), tot_ops);
    push_d(dv->tape_consts, cstflat.data(), tot_cst);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

extern "C" void *xmf_dev_dvec(size_t n) {
  if (n == 0)
    n = 1;
  try {
    double *p = static_cast<double *>(
        Kokkos::kokkos_malloc(sizeof(double) * n));
    if (!p)
      return NULL;
    /* Zeroed, exactly the dvec contract (the heat scratch relies on it). */
    Kokkos::deep_copy(
        Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(p, n),
        0.0);
    return p;
  } catch (const std::exception &) {
    return NULL;
  }
}

extern "C" void xmf_dev_dfree(void *p) {
  if (p)
    Kokkos::kokkos_free(p);
}

/* ---- Port B: device-side boundary condition state & skin fill ----------- */

extern "C" int xmf_dev_bc_sync(XmfRun *run) {
  if (!run)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0;
  try {
    int nslot = 5 * run->nfluid;
    dv->bc_nslot = nslot;
    /* Push the user-BC floor buffer (extent 0 when no floor loaded). */
    if (run->user_bc) {
      size_t nt = (size_t)run->nytot * (size_t)run->nztot;
      size_t bytes = (size_t)run->nfluid * XMF_SLOT_NVAR * nt;
      dv->user_bc = Kokkos::View<double *>("xmf_user_bc", bytes);
      push_d(dv->user_bc, run->user_bc, bytes);
    } else {
      dv->user_bc = Kokkos::View<double *>();
    }
    /* Push the magnetic tangential floor (extent 0 when none loaded). */
    if (run->mag_user_bc) {
      size_t nt = (size_t)run->nytot * (size_t)run->nztot;
      size_t bytes = 2 * nt;
      dv->mag_user_bc = Kokkos::View<double *>("xmf_mag_user_bc", bytes);
      push_d(dv->mag_user_bc, run->mag_user_bc, bytes);
    } else {
      dv->mag_user_bc = Kokkos::View<double *>();
    }
    /* Flatten per-face BC tapes into device arrays. */
    int total = 0;
    size_t tot_ops = 0, tot_cst = 0;
    for (int face = 0; face < 6; face++)
      total += run->nbc_tape[face];
    dv->bc_tape_total = total;
    dv->bc_tape_map =
        Kokkos::View<int *>("xmf_bc_tape_map", 6 * (size_t)nslot);
    std::vector<int> mapv(6 * (size_t)nslot, -1);
    if (total > 0) {
      std::vector<int> descv(4 * (size_t)total);
      std::vector<int> opsv;
      std::vector<double> cstv;
      int toff = 0;
      for (int face = 0; face < 6; face++) {
        int nt = run->nbc_tape[face];
        const XmfTape *tps = run->bc_tapes[face];
        for (int t = 0; t < nt; t++) {
          int slot = tps[t].dest_fluid * 5 + tps[t].dest_kind;
          mapv[face * nslot + slot] = toff;
          descv[4 * toff + 0] = tps[t].dest_fluid;
          descv[4 * toff + 1] = tps[t].dest_kind;
          descv[4 * toff + 2] = tps[t].nops;
          descv[4 * toff + 3] = tps[t].nconst;
          for (int p = 0; p < tps[t].nops; p++) {
            opsv.push_back(tps[t].ops[2 * p]);
            opsv.push_back(tps[t].ops[2 * p + 1]);
          }
          for (int q = 0; q < tps[t].nconst; q++)
            cstv.push_back(tps[t].consts[q]);
          toff++;
        }
      }
      tot_ops = opsv.size();
      tot_cst = cstv.size();
      dv->bc_tape_desc =
          Kokkos::View<int *>("xmf_bc_tape_desc", 4 * (size_t)total);
      dv->bc_tape_ops =
          Kokkos::View<int *>("xmf_bc_tape_ops", tot_ops > 0 ? tot_ops : 1);
      dv->bc_tape_consts = Kokkos::View<double *>(
          "xmf_bc_tape_consts", tot_cst > 0 ? tot_cst : 1);
      push_i(dv->bc_tape_desc, descv.data(), 4 * (size_t)total);
      if (tot_ops > 0)
        push_i(dv->bc_tape_ops, opsv.data(), tot_ops);
      if (tot_cst > 0)
        push_d(dv->bc_tape_consts, cstv.data(), tot_cst);
    } else {
      dv->bc_tape_desc = Kokkos::View<int *>();
      dv->bc_tape_ops = Kokkos::View<int *>();
      dv->bc_tape_consts = Kokkos::View<double *>();
    }
    push_i(dv->bc_tape_map, mapv.data(), 6 * (size_t)nslot);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

/* ---- Port B device BC fill kernels -------------------------------------- */

namespace {

/* Device-side BC parameter bundle (value type, copied into kernels). */
struct BcP {
  double *fields;
  double *pe;
  const double *user_bc;
  const int *bc_tape_map;
  const int *bc_tape_desc;
  const int *bc_tape_ops;
  const double *bc_tape_consts;
  const double *x1, *x2, *x3;
  const double *mass_code;
  int nfluid, ng, nxtot, nytot, nztot, nx, ny, nz;
  int bc[6];
  size_t ncells;
  int nslot;
  int bc_tape_total;
  double unit_P;
};

KOKKOS_INLINE_FUNCTION
size_t bc_cidx(const BcP &bp, int k, int j, int i) {
  return (((size_t)k * (size_t)bp.nytot) + (size_t)j) * (size_t)bp.nxtot +
         (size_t)i;
}

KOKKOS_INLINE_FUNCTION
double *bc_slot(const BcP &bp, int f, int var) {
  return bp.fields +
         (((size_t)f * XMF_SLOT_NVAR + (size_t)var) * bp.ncells);
}

KOKKOS_INLINE_FUNCTION
int bc_nb_value_dev(const BcP &bp, int face, int code, int k, int j, int i,
                    double *out) {
  int a = (code >> 8) & 0xFF;
  int b = code & 0xFF;
  int layer = (b >> 4) & 0x0F;
  int var = b & 0x0F;
  int d = face / 2, side = face % 2;
  int nint = (d == 0) ? bp.nx : (d == 1) ? bp.ny : bp.nz;
  int nb;
  int kk, jj, ii;
  size_t c;
  if (a >= bp.nfluid || layer < 1 || layer > 2 || var > XMF_IR_T)
    return -1;
  if (side == 0)
    nb = bp.ng + (layer - 1);
  else
    nb = bp.ng + nint - layer;
  kk = (d == 2) ? nb : k;
  jj = (d == 1) ? nb : j;
  ii = (d == 0) ? nb : i;
  c = bc_cidx(bp, kk, jj, ii);
  if (var == XMF_IR_T) {
    double rho = bc_slot(bp, a, XMF_SLOT_RHO)[c];
    double pc = bc_slot(bp, a, XMF_SLOT_P)[c];
    double n = rho / bp.mass_code[a];
    double n_safe = n > 1e-30 ? n : 1e-30;
    *out = pc * bp.unit_P / (n_safe * XMF_KB);
    return 0;
  }
  if (var == XMF_IR_N) {
    *out = bc_slot(bp, a, XMF_SLOT_RHO)[c] / bp.mass_code[a];
    return 0;
  }
  *out = bc_slot(bp, a, var)[c];
  return 0;
}

KOKKOS_INLINE_FUNCTION
int bc_eval_tape_dev(const BcP &bp, int face, int tape_idx, int k, int j,
                     int i, double *st, int st_cap, double *out) {
  int sp = 0, p;
  int nops = bp.bc_tape_desc[4 * tape_idx + 2];
  int nconst = bp.bc_tape_desc[4 * tape_idx + 3];
  int off_ops = 0, off_cst = 0;
  for (int t = 0; t < tape_idx; t++) {
    off_ops += 2 * bp.bc_tape_desc[4 * t + 2];
    off_cst += bp.bc_tape_desc[4 * t + 3];
  }
  if (nops + 1 > st_cap)
    return -1;
  for (p = 0; p < nops; p++) {
    int op = bp.bc_tape_ops[off_ops + 2 * p];
    int arg = bp.bc_tape_ops[off_ops + 2 * p + 1];
    switch (op) {
    case XMF_OP_LOAD_CONST:
      if (sp >= st_cap || arg < 0 || arg >= nconst)
        return -1;
      st[sp++] = bp.bc_tape_consts[off_cst + arg];
      break;
    case XMF_OP_LOAD_SLOT: {
      int kind = (arg >> 16) & 0xFF;
      int ax = (arg >> 8) & 0xFF;
      double v;
      if (sp >= st_cap)
        return -1;
      if (kind == XMF_IR_GRID && ax <= XMF_IR_X3 && (arg & 0xFF) == 0) {
        v = (ax == XMF_IR_X1)   ? bp.x1[i]
            : (ax == XMF_IR_X2) ? bp.x2[j]
                                : bp.x3[k];
      } else if (kind == XMF_IR_NB) {
        if (bc_nb_value_dev(bp, face, arg, k, j, i, &v))
          return -1;
      } else
        return -1;
      st[sp++] = v;
      break;
    }
    case XMF_OP_NEG:
      if (sp < 1) return -1;
      st[sp - 1] = -st[sp - 1];
      break;
    case XMF_OP_EXP:
      if (sp < 1) return -1;
      st[sp - 1] = exp(st[sp - 1]);
      break;
    case XMF_OP_LOG:
      if (sp < 1) return -1;
      st[sp - 1] = log(st[sp - 1]);
      break;
    default: {
      double bv, av, v;
      if (sp < 2) return -1;
      bv = st[--sp];
      av = st[--sp];
      if (op == XMF_OP_ADD) v = av + bv;
      else if (op == XMF_OP_SUB) v = av - bv;
      else if (op == XMF_OP_MUL) v = av * bv;
      else if (op == XMF_OP_DIV) v = av / bv;
      else if (op == XMF_OP_POW) v = pow(av, bv);
      else return -1;
      st[sp++] = v;
    }
    }
  }
  if (sp != 1) return -1;
  *out = st[0];
  return 0;
}

/* Per-face fluid BC fill kernel. Each iteration fills one ghost cell of
   one face. Handles copy / reflect / user (floor + tape) / periodic.
   Periodic: ghost wraps from the opposite active domain (no sign flip). */
KOKKOS_INLINE_FUNCTION
void bc_fill_one_face(const BcP &bp, int face, int a, int b, int m) {
  static const int normvar[3] = {XMF_SLOT_VX, XMF_SLOT_VY, XMF_SLOT_VZ};
  int d = face / 2, side = face % 2;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int is_reflect = (bp.bc[face] == XMF_BC_REFLECT);
  int is_user = (bp.bc[face] == XMF_BC_USER);
  int is_periodic = (bp.bc[face] == XMF_BC_PERIODIC);
  int ti[3];
  ti[t1] = a;
  ti[t2] = b;
  int nint_d = (d == 0) ? bp.nx : (d == 1) ? bp.ny : bp.nz;
  int gnorm, snorm;
  if (is_periodic) {
    /* Periodic: ghost wraps from opposite active domain. beg ghost m
       wraps from end interior (ng + nint - 1 - m); end ghost m wraps
       from beg interior (ng + m). No sign flip. */
    if (side == 0) {
      gnorm = bp.ng - 1 - m;
      snorm = bp.ng + nint_d - 1 - m;
    } else {
      gnorm = bp.ng + nint_d + m;
      snorm = bp.ng + m;
    }
  } else if (side == 0) {
    gnorm = bp.ng - 1 - m;
    snorm = is_reflect ? bp.ng + m : bp.ng;
  } else {
    gnorm = bp.ng + nint_d + m;
    snorm = is_reflect ? bp.ng + nint_d - 1 - m : bp.ng + nint_d - 1;
  }
  ti[d] = gnorm;
  int k = ti[2], j = ti[1], i = ti[0];
  size_t cg = bc_cidx(bp, k, j, i);
  int si[3] = {i, j, k};
  si[d] = snorm;
  size_t csrc = bc_cidx(bp, si[2], si[1], si[0]);
  /* User face: try floor buffer, then tape, then copy @1. */
  if (is_user) {
    if (bp.user_bc && d == 0) {
      size_t nt = (size_t)bp.nytot * (size_t)bp.nztot;
      size_t tidx = (size_t)k * (size_t)bp.nytot + (size_t)j;
      int i_inner = (side == 0) ? bp.ng : bp.ng + bp.nx - 1;
      size_t c_inner = bc_cidx(bp, k, j, i_inner);
      for (int f = 0; f < bp.nfluid; f++)
        for (int var = 0; var < XMF_SLOT_NVAR; var++) {
          double v0 =
              bp.user_bc[((size_t)f * XMF_SLOT_NVAR + (size_t)var) * nt +
                         tidx];
          bc_slot(bp, f, var)[cg] = v0;
          bc_slot(bp, f, var)[c_inner] = v0;
        }
      return;
    }
    if (bp.bc_tape_total > 0) {
      double st[64];
      for (int f = 0; f < bp.nfluid; f++)
        for (int var = 0; var < XMF_SLOT_NVAR; var++) {
          int slot = f * 5 + var;
          int tidx = bp.bc_tape_map[face * bp.nslot + slot];
          if (tidx >= 0) {
            double val;
            if (bc_eval_tape_dev(bp, face, tidx, k, j, i, st, 64, &val) ==
                0) {
              bc_slot(bp, f, var)[cg] =
                  (var == XMF_SLOT_RHO) ? bp.mass_code[f] * val : val;
            } else {
              int sn2[3] = {i, j, k};
              sn2[d] = (side == 0) ? bp.ng : bp.ng + nint_d - 1;
              size_t c2 = bc_cidx(bp, sn2[2], sn2[1], sn2[0]);
              bc_slot(bp, f, var)[cg] = bc_slot(bp, f, var)[c2];
            }
          } else {
            int sn2[3] = {i, j, k};
            sn2[d] = (side == 0) ? bp.ng : bp.ng + nint_d - 1;
            size_t c2 = bc_cidx(bp, sn2[2], sn2[1], sn2[0]);
            bc_slot(bp, f, var)[cg] = bc_slot(bp, f, var)[c2];
          }
        }
      return;
    }
  }
  /* copy / outflow / reflect / periodic / user-without-tapes */
  for (int f = 0; f < bp.nfluid; f++)
    for (int var = 0; var < XMF_SLOT_NVAR; var++) {
      double v = bc_slot(bp, f, var)[csrc];
      if (is_reflect && var == normvar[d])
        v = -v;
      bc_slot(bp, f, var)[cg] = v;
    }
}

/* P_e auxiliary-block fill kernel: scalar, never flipped. */
KOKKOS_INLINE_FUNCTION
void bc_fill_pe_face(const BcP &bp, int face, int a, int b, int m) {
  int d = face / 2, side = face % 2;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int is_periodic = (bp.bc[face] == XMF_BC_PERIODIC);
  int is_reflect = (bp.bc[face] == XMF_BC_REFLECT);
  int ti[3];
  ti[t1] = a;
  ti[t2] = b;
  int nint_d = (d == 0) ? bp.nx : (d == 1) ? bp.ny : bp.nz;
  int gnorm, snorm;
  if (is_periodic) {
    if (side == 0) {
      gnorm = bp.ng - 1 - m;
      snorm = bp.ng + nint_d - 1 - m;
    } else {
      gnorm = bp.ng + nint_d + m;
      snorm = bp.ng + m;
    }
  } else if (side == 0) {
    gnorm = bp.ng - 1 - m;
    snorm = is_reflect ? bp.ng + m : bp.ng;
  } else {
    gnorm = bp.ng + nint_d + m;
    snorm = is_reflect ? bp.ng + nint_d - 1 - m : bp.ng + nint_d - 1;
  }
  ti[d] = gnorm;
  int k = ti[2], j = ti[1], i = ti[0];
  size_t cg = bc_cidx(bp, k, j, i);
  int si[3] = {i, j, k};
  si[d] = snorm;
  size_t csrc = bc_cidx(bp, si[2], si[1], si[0]);
  bp.pe[cg] = bp.pe[csrc];
}

/* ---- Port B: B face-centred magnetic ghost parameter bundle ------------- */

struct BcB {
  double *bf[3];        /* face-centred B blocks */
  const double *Ad[3];  /* face area factors per direction */
  const double *dVd[3]; /* volume factors per direction */
  const double *mag_user_bc; /* [2*nt] tangential B floor or NULL */
  int ng, nxtot, nytot, nztot, nx, ny, nz;
  int bc[6];
  int mag_reflect;      /* 0 = wall, 1 = symmetric */
};

/* Flat index into bf[comp] (same formula as mag_bidx in xmf_mag.c). */
KOKKOS_INLINE_FUNCTION
size_t bc_mag_bidx(const BcB &bb, int comp, const int c[3]) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  int nt_d = (comp == 0) ? bb.nxtot : (comp == 1) ? bb.nytot : bb.nztot;
  int nt_d1 = (d1 == 0) ? bb.nxtot : (d1 == 1) ? bb.nytot : bb.nztot;
  return (((size_t)c[d2] * (size_t)nt_d1) + (size_t)c[d1]) *
             (size_t)(nt_d + 1) +
         (size_t)c[comp];
}

KOKKOS_INLINE_FUNCTION
int bc_mag_ntot(const BcB &bb, int d) {
  return (d == 0) ? bb.nxtot : (d == 1) ? bb.nytot : bb.nztot;
}

/* Reflect fill for one face of the B blocks (device). Matches the host
   mag_fill_reflect semantics: wall tier flips B_n and pins wall face to 0,
   copies B_t; symmetric tier copies B_n, flips B_t. */
KOKKOS_INLINE_FUNCTION
void bc_fill_bface_reflect(const BcB &bb, int face, int a, int b, int m) {
  int d = face / 2, side = face % 2;
  int ng = bb.ng;
  int nint = (d == 0) ? bb.nx : (d == 1) ? bb.ny : bb.nz;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int wall = (bb.mag_reflect == 0);
  int w = (side == 0) ? ng : ng + nint;
  double sgn_n = wall ? -1.0 : 1.0;
  double sgn_t = wall ? 1.0 : -1.0;
  int delta = m + 1; /* ghost layer 1..ng */
  /* Normal component (face-indexed along d, cell-indexed across). */
  if (a < bc_mag_ntot(bb, t1) && b < bc_mag_ntot(bb, t2)) {
    int c[3];
    c[t1] = a;
    c[t2] = b;
    if (wall && delta == 1) {
      c[d] = w;
      bb.bf[d][bc_mag_bidx(bb, d, c)] = 0.0;
    }
    int f = (side == 0) ? w - delta : w + delta;
    int src = (side == 0) ? w + delta : w - delta;
    double v;
    c[d] = src;
    v = bb.bf[d][bc_mag_bidx(bb, d, c)];
    c[d] = f;
    bb.bf[d][bc_mag_bidx(bb, d, c)] = sgn_n * v;
  }
  /* Tangential components (cell-indexed along d). */
  for (int t = 0; t < 3; t++) {
    int u, nf, nc;
    if (t == d) continue;
    u = 3 - d - t;
    nf = bc_mag_ntot(bb, t) + 1;
    nc = bc_mag_ntot(bb, u);
    /* a iterates t1, b iterates t2; map to (t, u) face/cell indices. */
    int fa = (t == t1) ? a : b;
    int fb = (t == t1) ? b : a;
    if (fa >= nf || fb >= nc) continue;
    int c[3];
    c[t] = fa;
    c[u] = fb;
    int g = (side == 0) ? ng - delta : ng + nint - 1 + delta;
    int src = (side == 0) ? ng + delta - 1 : ng + nint - delta;
    double v;
    c[d] = src;
    v = bb.bf[t][bc_mag_bidx(bb, t, c)];
    c[d] = g;
    bb.bf[t][bc_mag_bidx(bb, t, c)] = sgn_t * v;
  }
}

/* Copy/outflow/none fill for one face of the B blocks (device). */
KOKKOS_INLINE_FUNCTION
void bc_fill_bface_copy(const BcB &bb, int face, int a, int b, int m) {
  int d = face / 2, side = face % 2;
  int ng = bb.ng;
  int nint = (d == 0) ? bb.nx : (d == 1) ? bb.ny : bb.nz;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int delta = m + 1;
  /* Normal component: ghost face = boundary face value. */
  if (a < bc_mag_ntot(bb, t1) && b < bc_mag_ntot(bb, t2)) {
    int c[3];
    int w = (side == 0) ? ng : ng + nint;
    c[t1] = a;
    c[t2] = b;
    c[d] = w;
    double bw = bb.bf[d][bc_mag_bidx(bb, d, c)];
    int f = (side == 0) ? ng - delta : ng + nint + delta;
    c[d] = f;
    bb.bf[d][bc_mag_bidx(bb, d, c)] = bw;
  }
  /* Tangential components: ghost = outermost active cell. */
  for (int t = 0; t < 3; t++) {
    int u, nf, nc;
    if (t == d) continue;
    u = 3 - d - t;
    nf = bc_mag_ntot(bb, t) + 1;
    nc = bc_mag_ntot(bb, u);
    int fa = (t == t1) ? a : b;
    int fb = (t == t1) ? b : a;
    if (fa >= nf || fb >= nc) continue;
    int c[3];
    int src;
    double v0;
    c[t] = fa;
    c[u] = fb;
    src = (side == 0) ? ng : ng + nint - 1;
    c[d] = src;
    v0 = bb.bf[t][bc_mag_bidx(bb, t, c)];
    int g = (side == 0) ? ng - delta : ng + nint - 1 + delta;
    c[d] = g;
    bb.bf[t][bc_mag_bidx(bb, t, c)] = v0;
  }
}

/* Periodic fill for one face of the B blocks (device). */
KOKKOS_INLINE_FUNCTION
void bc_fill_bface_periodic(const BcB &bb, int face, int a, int b, int m) {
  int d = face / 2;
  int nint_d = (d == 0) ? bb.nx : (d == 1) ? bb.ny : bb.nz;
  int ng = bb.ng;
  int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int delta = m + 1;
  /* Remap launch indices (a, b) from the (t1, t2) launch convention to
     the (d1, d2) storage axes. The launch iterates a over the t1 face
     range and b over the t2 face range; when (d1, d2) is a permutation
     of (t1, t2) that is not the identity (d == 1 swaps them), a direct
     assignment cc[d1]=a, cc[d2]=b would put x-positions into z-slots
     and vice versa. Match the sibling reflect / copy / user_tangential
     kernels which all do this remap. */
  int id1 = (d1 == t1) ? a : b;
  int id2 = (d2 == t1) ? a : b;
  /* Walk all three components. For each, fill the ghost entry at this
     transverse position and ghost layer. */
  for (int comp = 0; comp < 3; comp++) {
    int nt_d1 = bc_mag_ntot(bb, d1);
    int nt_d2 = bc_mag_ntot(bb, d2);
    /* Transverse ranges include face rows (ntot[d_k]+1) for the
       component's own direction, matching host mag_fill_periodic. */
    int f1_max = (comp == d1) ? nt_d1 + 1 : nt_d1;
    int f2_max = (comp == d2) ? nt_d2 + 1 : nt_d2;
    if (id1 >= f1_max || id2 >= f2_max) continue;
    int cc[3], cc_src[3];
    cc[d1] = id1;
    cc[d2] = id2;
    cc_src[d1] = id1;
    cc_src[d2] = id2;
    if (comp == d) {
      /* B_d: face-centred along d. Active faces: ng..ng+nint_d.
         Beg ghosts: 0..ng-1 (f_d = ng - delta).
         End ghosts: ng+nint_d+1..ntot[d] (f_d = ng + nint_d + delta). */
      int side = face % 2;
      if (side == 0) {
        cc[d] = ng - delta;
        cc_src[d] = ng - delta + nint_d;
      } else {
        cc[d] = ng + nint_d + delta;
        cc_src[d] = ng + delta;
      }
      bb.bf[comp][bc_mag_bidx(bb, comp, cc)] =
          bb.bf[comp][bc_mag_bidx(bb, comp, cc_src)];
    } else {
      /* B_d1 or B_d2: cell-centred along d. Active cells: ng..ng+nint_d-1.
         Beg ghosts: 0..ng-1 (f_d = ng - delta).
         End ghosts: ng+nint_d..ntot[d]-1 (f_d = ng + nint_d + delta - 1). */
      int side = face % 2;
      if (side == 0) {
        cc[d] = ng - delta;
        cc_src[d] = ng - delta + nint_d;
      } else {
        cc[d] = ng + nint_d + delta - 1;
        cc_src[d] = ng + delta - 1;
      }
      bb.bf[comp][bc_mag_bidx(bb, comp, cc)] =
          bb.bf[comp][bc_mag_bidx(bb, comp, cc_src)];
    }
  }
}

/* User face B fill: tangential from floor (x1 only) or copy @1, one
   ghost layer per iteration. The normal reconstruction (divB = 0 march)
   is done in a separate kernel after all tangential layers are filled. */
KOKKOS_INLINE_FUNCTION
void bc_fill_bface_user_tangential(const BcB &bb, int face, int a, int b,
                                   int m) {
  int d = face / 2, side = face % 2;
  int ng = bb.ng;
  int nint = (d == 0) ? bb.nx : (d == 1) ? bb.ny : bb.nz;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int delta = m + 1;
  int has_floor = (d == 0) && bb.mag_user_bc;
  size_t nt = (size_t)bb.nytot * (size_t)bb.nztot;
  for (int t = 0; t < 3; t++) {
    int u, nf, nc;
    if (t == d) continue;
    u = 3 - d - t;
    nf = bc_mag_ntot(bb, t) + 1;
    nc = bc_mag_ntot(bb, u);
    int fa = (t == t1) ? a : b;
    int fb = (t == t1) ? b : a;
    if (fa >= nf || fb >= nc) continue;
    int c[3];
    double v0;
    c[t] = fa;
    c[u] = fb;
    if (has_floor && fa < bc_mag_ntot(bb, t)) {
      int tcomp = (t == 1) ? 0 : 1;
      v0 = bb.mag_user_bc[(size_t)tcomp * nt +
                          (size_t)c[2] * (size_t)bb.nytot +
                          (size_t)c[1]];
    } else {
      int src = (side == 0) ? ng : ng + nint - 1;
      c[d] = src;
      v0 = bb.bf[t][bc_mag_bidx(bb, t, c)];
    }
    int g = (side == 0) ? ng - delta : ng + nint - 1 + delta;
    c[d] = g;
    bb.bf[t][bc_mag_bidx(bb, t, c)] = v0;
  }
}

/* User face B normal reconstruction (divB = 0 march). One thread per
   transverse position; marches outward through all ng ghost layers.
   Must be launched AFTER the tangential fill kernel completes. */
KOKKOS_INLINE_FUNCTION
void bc_fill_bface_user_normal(const BcB &bb, int face, int a, int b) {
  int d = face / 2, side = face % 2;
  int ng = bb.ng;
  int nint = (d == 0) ? bb.nx : (d == 1) ? bb.ny : bb.nz;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  const double *Ad = bb.Ad[d];
  const double *dVd = bb.dVd[d];
  int c[3];
  c[d] = 0;
  c[t1] = a;
  c[t2] = b;
  for (int m = 1; m <= ng; m++) {
    int i = (side == 0) ? ng - m : ng + nint - 1 + m;
    int fk, fu;
    double s = 0.0;
    c[d] = i;
    for (int e = 0; e < 3; e++) {
      const double *Ae, *dVe;
      int ce;
      double vin, vout;
      if (e == d) continue;
      Ae = bb.Ad[e];
      dVe = bb.dVd[e];
      ce = c[e];
      if (dVe[ce] == 0.0) continue;
      c[e] = ce + 1;
      vout = bb.bf[e][bc_mag_bidx(bb, e, c)];
      c[e] = ce;
      vin = bb.bf[e][bc_mag_bidx(bb, e, c)];
      s += (Ae[ce + 1] * vout - Ae[ce] * vin) / dVe[ce];
    }
    if (side == 0) { fk = i + 1; fu = i; }
    else { fk = i; fu = i + 1; }
    c[d] = fk;
    double bk = bb.bf[d][bc_mag_bidx(bb, d, c)];
    c[d] = fu;
    if (Ad[fu] != 0.0) {
      double num = Ad[fk] * bk + (side == 0 ? dVd[i] * s : -dVd[i] * s);
      bb.bf[d][bc_mag_bidx(bb, d, c)] = num / Ad[fu];
    } else {
      bb.bf[d][bc_mag_bidx(bb, d, c)] = bk;
    }
  }
}

/* ---- Port B: E edge-centred ghost fill (device) -------------------------
   Non-periodic ghost edges: zero-order copy from nearest interior edge.
   Periodic ghost edges are already filled by the host interior loop in
   mag_induce (the wrapped 4-cell average); this kernel only fills the
   non-periodic ones. The three-pass dependency order (d1 edges → d2 edges
   → corners) is maintained by launching three separate kernels. */

struct BcE {
  double *ef[3];
  int ng, nxtot, nytot, nztot, nx, ny, nz;
  int bc[6];
};

KOKKOS_INLINE_FUNCTION
size_t bc_mag_eidx(const BcE &be, int comp, const int c[3]) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  int nt_d = (comp == 0) ? be.nxtot : (comp == 1) ? be.nytot : be.nztot;
  int nt_d1 = (d1 == 0) ? be.nxtot : (d1 == 1) ? be.nytot : be.nztot;
  return (((size_t)c[d2] * (size_t)(nt_d1 + 1)) + (size_t)c[d1]) *
             (size_t)nt_d +
         (size_t)c[comp];
}

/* Pass 1: d1 ghost edges over interior d2 range (1..nd2-1). */
KOKKOS_INLINE_FUNCTION
void bc_fill_edge_d1(const BcE &be, int d, int k, int i) {
  int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
  int nd1 = (d1 == 0) ? be.nxtot : (d1 == 1) ? be.nytot : be.nztot;
  int nd2 = (d2 == 0) ? be.nxtot : (d2 == 1) ? be.nytot : be.nztot;
  int beg_per = (be.bc[2 * d1] == XMF_BC_PERIODIC);
  int end_per = (be.bc[2 * d1 + 1] == XMF_BC_PERIODIC);
  int cc_ghost[3], cc_src[3];
  cc_ghost[d] = k;
  cc_src[d] = k;
  cc_ghost[d2] = i;
  cc_src[d2] = i;
  if (!beg_per) {
    cc_ghost[d1] = 0;
    cc_src[d1] = 1;
    be.ef[d][bc_mag_eidx(be, d, cc_ghost)] =
        be.ef[d][bc_mag_eidx(be, d, cc_src)];
  }
  if (!end_per) {
    cc_ghost[d1] = nd1;
    cc_src[d1] = nd1 - 1;
    be.ef[d][bc_mag_eidx(be, d, cc_ghost)] =
        be.ef[d][bc_mag_eidx(be, d, cc_src)];
  }
}

/* Pass 2: d2 ghost edges over interior d1 range (1..nd1-1). */
KOKKOS_INLINE_FUNCTION
void bc_fill_edge_d2(const BcE &be, int d, int k, int j) {
  int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
  int nd1 = (d1 == 0) ? be.nxtot : (d1 == 1) ? be.nytot : be.nztot;
  int nd2 = (d2 == 0) ? be.nxtot : (d2 == 1) ? be.nytot : be.nztot;
  int beg_per_d2 = (be.bc[2 * d2] == XMF_BC_PERIODIC);
  int end_per_d2 = (be.bc[2 * d2 + 1] == XMF_BC_PERIODIC);
  int cc_ghost[3], cc_src[3];
  cc_ghost[d] = k;
  cc_src[d] = k;
  cc_ghost[d1] = j;
  cc_src[d1] = j;
  if (!beg_per_d2) {
    cc_ghost[d2] = 0;
    cc_src[d2] = 1;
    be.ef[d][bc_mag_eidx(be, d, cc_ghost)] =
        be.ef[d][bc_mag_eidx(be, d, cc_src)];
  }
  if (!end_per_d2) {
    cc_ghost[d2] = nd2;
    cc_src[d2] = nd2 - 1;
    be.ef[d][bc_mag_eidx(be, d, cc_ghost)] =
        be.ef[d][bc_mag_eidx(be, d, cc_src)];
  }
}

/* Pass 3: corner ghost edges (both d1 and d2 at ghost positions). */
KOKKOS_INLINE_FUNCTION
void bc_fill_edge_corner(const BcE &be, int d, int k, int di, int dj) {
  int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
  int nd1 = (d1 == 0) ? be.nxtot : (d1 == 1) ? be.nytot : be.nztot;
  int nd2 = (d2 == 0) ? be.nxtot : (d2 == 1) ? be.nytot : be.nztot;
  int d1_ghost[2] = {0, nd1};
  int d2_ghost[2] = {0, nd2};
  int d1_is_beg = (d1_ghost[di] == 0);
  int d2_is_beg = (d2_ghost[dj] == 0);
  int d1_per = d1_is_beg ? (be.bc[2 * d1] == XMF_BC_PERIODIC)
                         : (be.bc[2 * d1 + 1] == XMF_BC_PERIODIC);
  int d2_per = d2_is_beg ? (be.bc[2 * d2] == XMF_BC_PERIODIC)
                         : (be.bc[2 * d2 + 1] == XMF_BC_PERIODIC);
  if (d1_per && d2_per) return;
  int cc_ghost[3], cc_src[3];
  cc_ghost[d] = k;
  cc_src[d] = k;
  cc_ghost[d1] = d1_ghost[di];
  cc_ghost[d2] = d2_ghost[dj];
  cc_src[d1] = d1_is_beg ? 1 : nd1 - 1;
  cc_src[d2] = d2_is_beg ? 1 : nd2 - 1;
  be.ef[d][bc_mag_eidx(be, d, cc_ghost)] =
      be.ef[d][bc_mag_eidx(be, d, cc_src)];
}

}  // anonymous namespace

extern "C" void xmf_bc_fill_device(XmfRun *run) {
  if (!run || !run->dev)
    return;
  DeviceState *dv = dev(run);
  BcP bp;
  bp.fields = dv->fields.data();
  bp.pe = run->pe ? dv->pe.data() : nullptr;
  bp.user_bc = dv->user_bc.extent(0) > 0 ? dv->user_bc.data() : nullptr;
  bp.bc_tape_map =
      dv->bc_tape_map.extent(0) > 0 ? dv->bc_tape_map.data() : nullptr;
  bp.bc_tape_desc =
      dv->bc_tape_desc.extent(0) > 0 ? dv->bc_tape_desc.data() : nullptr;
  bp.bc_tape_ops =
      dv->bc_tape_ops.extent(0) > 0 ? dv->bc_tape_ops.data() : nullptr;
  bp.bc_tape_consts =
      dv->bc_tape_consts.extent(0) > 0 ? dv->bc_tape_consts.data() : nullptr;
  bp.x1 = dv->x[0].data();
  bp.x2 = dv->x[1].data();
  bp.x3 = dv->x[2].data();
  bp.mass_code = dv->mass_code.data();
  bp.nfluid = run->nfluid;
  bp.ng = run->ng;
  bp.nxtot = run->nxtot;
  bp.nytot = run->nytot;
  bp.nztot = run->nztot;
  bp.nx = run->nx;
  bp.ny = run->ny;
  bp.nz = run->nz;
  for (int f = 0; f < 6; f++)
    bp.bc[f] = run->bc[f];
  bp.ncells = run->ncells;
  bp.nslot = dv->bc_nslot;
  bp.bc_tape_total = dv->bc_tape_total;
  bp.unit_P = run->unit_P;
  int ele = elec_closure_active(run);
  int mag = run->bf[0] != NULL;
  int nd[3] = {run->nx, run->ny, run->nz};
  for (int d = 0; d < 3; d++) {
    /* Port B: process ALL faces including degenerate directions. The fluid
       ghost fill for degenerate y/z faces copies the active slice to the
       ghost slices, keeping the state transversely uniform (needed for
       the induction curl's transverse derivatives to be zero in 1D). The
       B fill also runs for degenerate directions: the CT transverse
       stencil reads degenerate ghost cells (see .tmp/port_b_degenerate_caliber.md). */
    int t1_ext = (d == 0) ? bp.nytot : bp.nxtot;
    int t2_ext = (d == 2) ? bp.nytot : bp.nztot;
    for (int side = 0; side < 2; side++) {
      int face = 2 * d + side;
      try {
        /* Fluid + P_e ghost fill (one kernel per face). */
        Kokkos::parallel_for(
            "xmf_bc_fill_fluid_face",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0, 0, 0}, {t1_ext, t2_ext, bp.ng}),
            KOKKOS_LAMBDA(const int a, const int b, const int m) {
              bc_fill_one_face(bp, face, a, b, m);
            });
        if (ele) {
          Kokkos::parallel_for(
              "xmf_bc_fill_pe_face",
              Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                  {0, 0, 0}, {t1_ext, t2_ext, bp.ng}),
              KOKKOS_LAMBDA(const int a, const int b, const int m) {
                bc_fill_pe_face(bp, face, a, b, m);
              });
        }
        /* B face-centred ghost fill (gate open only). Port B: fill ALL
           directions including degenerate. */
        if (mag) {
          BcB bb;
          for (int dd = 0; dd < 3; dd++) {
            bb.bf[dd] = dv->bf[dd].data();
            bb.Ad[dd] = dv->A[dd].data();
            bb.dVd[dd] = dv->dV[dd].data();
          }
          bb.mag_user_bc =
              dv->mag_user_bc.extent(0) > 0 ? dv->mag_user_bc.data()
                                            : nullptr;
          bb.ng = run->ng;
          bb.nxtot = run->nxtot;
          bb.nytot = run->nytot;
          bb.nztot = run->nztot;
          bb.nx = run->nx;
          bb.ny = run->ny;
          bb.nz = run->nz;
          for (int f = 0; f < 6; f++)
            bb.bc[f] = run->bc[f];
          bb.mag_reflect = run->mag_reflect;
          int is_reflect = (run->bc[face] == XMF_BC_REFLECT);
          int is_periodic = (run->bc[face] == XMF_BC_PERIODIC);
          int is_user = (run->bc[face] == XMF_BC_USER);
          if (is_reflect) {
            /* Reflect: one kernel covers normal + tangential ghosts.
               The transverse extents differ per component; the kernel
               internally bounds-checks. For the normal component,
               a iterates t1 cells (ntot[t1]), b iterates t2 cells
               (ntot[t2]). For tangential, the ranges are larger
               (ntot[t]+1 face rows). Launch over the max extent. */
            int t1 = (d == 0) ? 1 : 0;
            int t2 = (d == 2) ? 1 : 2;
            int a_max = bc_mag_ntot(bb, t1) + 1;
            int b_max = bc_mag_ntot(bb, t2) + 1;
            Kokkos::parallel_for(
                "xmf_bc_fill_bface_reflect",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0, 0, 0}, {a_max, b_max, bp.ng}),
                KOKKOS_LAMBDA(const int a, const int b, const int m) {
                  bc_fill_bface_reflect(bb, face, a, b, m);
                });
          } else if (is_user) {
            /* User: tangential first, then normal march (sequential). */
            int t1 = (d == 0) ? 1 : 0;
            int t2 = (d == 2) ? 1 : 2;
            int a_max = bc_mag_ntot(bb, t1) + 1;
            int b_max = bc_mag_ntot(bb, t2) + 1;
            Kokkos::parallel_for(
                "xmf_bc_fill_bface_user_tan",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0, 0, 0}, {a_max, b_max, bp.ng}),
                KOKKOS_LAMBDA(const int a, const int b, const int m) {
                  bc_fill_bface_user_tangential(bb, face, a, b, m);
                });
            /* Normal march: one thread per transverse position. */
            Kokkos::parallel_for(
                "xmf_bc_fill_bface_user_norm",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {0, 0}, {bc_mag_ntot(bb, t1), bc_mag_ntot(bb, t2)}),
                KOKKOS_LAMBDA(const int a, const int b) {
                  bc_fill_bface_user_normal(bb, face, a, b);
                });
          } else if (is_periodic) {
            int t1 = (d == 0) ? 1 : 0;
            int t2 = (d == 2) ? 1 : 2;
            int a_max = bc_mag_ntot(bb, t1) + 1;
            int b_max = bc_mag_ntot(bb, t2) + 1;
            Kokkos::parallel_for(
                "xmf_bc_fill_bface_periodic",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0, 0, 0}, {a_max, b_max, bp.ng}),
                KOKKOS_LAMBDA(const int a, const int b, const int m) {
                  bc_fill_bface_periodic(bb, face, a, b, m);
                });
          } else {
            /* copy / outflow / none */
            int t1 = (d == 0) ? 1 : 0;
            int t2 = (d == 2) ? 1 : 2;
            int a_max = bc_mag_ntot(bb, t1) + 1;
            int b_max = bc_mag_ntot(bb, t2) + 1;
            Kokkos::parallel_for(
                "xmf_bc_fill_bface_copy",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0, 0, 0}, {a_max, b_max, bp.ng}),
                KOKKOS_LAMBDA(const int a, const int b, const int m) {
                  bc_fill_bface_copy(bb, face, a, b, m);
                });
          }
        }
      } catch (const std::exception &e) {
        fprintf(stderr, "xmf_bc_fill_device: face %d exception: %s\n", face,
                e.what());
        return;
      } catch (...) {
        fprintf(stderr, "xmf_bc_fill_device: face %d unknown exception\n",
                face);
        return;
      }
    }
  }
}

/* Port B: E edge-centred ghost fill on device. Called from mag_induce
   after the interior E fill. Three separate kernel launches maintain
   the d1 → d2 → corner dependency order. */
extern "C" void xmf_bc_fill_e_ghosts_device(XmfRun *run) {
  if (!run || !run->dev || !run->bf[0])
    return;
  DeviceState *dv = dev(run);
  BcE be;
  for (int d = 0; d < 3; d++)
    be.ef[d] = dv->ef[d].data();
  be.ng = run->ng;
  be.nxtot = run->nxtot;
  be.nytot = run->nytot;
  be.nztot = run->nztot;
  be.nx = run->nx;
  be.ny = run->ny;
  be.nz = run->nz;
  for (int f = 0; f < 6; f++)
    be.bc[f] = run->bc[f];
  for (int d = 0; d < 3; d++) {
    int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
    int nd = (d == 0) ? be.nxtot : (d == 1) ? be.nytot : be.nztot;
    int nd1 = (d1 == 0) ? be.nxtot : (d1 == 1) ? be.nytot : be.nztot;
    int nd2 = (d2 == 0) ? be.nxtot : (d2 == 1) ? be.nytot : be.nztot;
    try {
      /* Pass 1: d1 ghost edges over interior d2 range (1..nd2-1). */
      Kokkos::parallel_for(
          "xmf_bc_fill_edge_d1",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {nd, nd2}),
          KOKKOS_LAMBDA(const int k, const int i) {
            bc_fill_edge_d1(be, d, k, i);
          });
      /* Pass 2: d2 ghost edges over interior d1 range (1..nd1-1). */
      Kokkos::parallel_for(
          "xmf_bc_fill_edge_d2",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {nd, nd1}),
          KOKKOS_LAMBDA(const int k, const int j) {
            bc_fill_edge_d2(be, d, k, j);
          });
      /* Pass 3: corner ghost edges. */
      Kokkos::parallel_for(
          "xmf_bc_fill_edge_corner",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
              {0, 0, 0}, {nd, 2, 2}),
          KOKKOS_LAMBDA(const int k, const int di, const int dj) {
            bc_fill_edge_corner(be, d, k, di, dj);
          });
    } catch (...) {
      return;
    }
  }
}

/* ---- Port D: device-side induction (mag_up_snapshot + mag_induce) --------
   The two device hooks mirror the host loops of xmf_mag.c expression by
   expression (target: bit-for-bit under --fmad=false). The host path
   (run->dev == NULL) is unchanged and kept as the fallback. The device
   path reads dv->fields / dv->bf / dv->mag_up / dv->mag_g exclusively;
   no host pointer is dereferenced inside a kernel. The mirror contract:
   the host run->bf / run->ef / run->mag_up are "last-pulled" state --
   unpack helpers pull on read, xmf_divb_max pulls on entry, and the host
   fallback path still does pull → compute → push every stage. */

namespace {

struct IndP {
  double *fields;
  double *mag_up;
  double *mag_g;
  double *bf[3];
  double *ef[3];
  const double *charge;
  const double *mass;
  const double *A[3];
  const double *dV[3];
  int nfluid;
  int ng, nx, ny, nz;
  int nxtot, nytot, nztot;
  size_t ncells;
  int bc[6];
};

KOKKOS_INLINE_FUNCTION
size_t ind_cidx(const IndP &p, int k, int j, int i) {
  return (((size_t)k * (size_t)p.nytot) + (size_t)j) * (size_t)p.nxtot +
         (size_t)i;
}

KOKKOS_INLINE_FUNCTION
int ind_ntot(const IndP &p, int d) {
  return (d == 0) ? p.nxtot : (d == 1) ? p.nytot : p.nztot;
}

KOKKOS_INLINE_FUNCTION
size_t ind_bidx(const IndP &p, int comp, const int c[3]) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  int nt_d = ind_ntot(p, comp);
  int nt_d1 = ind_ntot(p, d1);
  return (((size_t)c[d2] * (size_t)nt_d1) + (size_t)c[d1]) *
             (size_t)(nt_d + 1) +
         (size_t)c[comp];
}

KOKKOS_INLINE_FUNCTION
size_t ind_eidx(const IndP &p, int comp, const int c[3]) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  int nt_d = ind_ntot(p, comp);
  int nt_d1 = ind_ntot(p, d1);
  return (((size_t)c[d2] * (size_t)(nt_d1 + 1)) + (size_t)c[d1]) *
             (size_t)nt_d +
         (size_t)c[comp];
}

/* mag_g_edge_p with periodic wrapping (M6 fix, host :782-819). The edge
   E at a periodic ghost position is the wrapped 4-cell average of G, not
   a copy from an interior edge (which misses the cell-index wrap and
   introduces an O(dx) bias). */
KOKKOS_INLINE_FUNCTION
double ind_g_edge_p(const IndP &p, int d, const int cc[3], int de, int comp,
                    int per0, int per1, int per2) {
  int d3 = 3 - d - de;
  int ntot_de = ind_ntot(p, de);
  int ntot_d = ind_ntot(p, d);
  int ntot_d3 = ind_ntot(p, d3);
  int per_d = (d == 0) ? per0 : (d == 1) ? per1 : per2;
  int per_de = (de == 0) ? per0 : (de == 1) ? per1 : per2;
  double s = 0.0;
  for (int a = 0; a < 2; a++) {
    int id = cc[d] - 1 + a;
    if (per_d && id < 0)
      id += ntot_d;
    else if (per_d && id >= ntot_d)
      id -= ntot_d;
    for (int b = 0; b < 2; b++) {
      int ide = cc[de] - 1 + b;
      int lo[3];
      int id3;
      if (per_de && ide < 0)
        ide += ntot_de;
      else if (per_de && ide >= ntot_de)
        ide -= ntot_de;
      id3 = cc[d3];
      if (id3 >= ntot_d3)
        id3 = ntot_d3 - 1;
      lo[d] = id;
      lo[de] = ide;
      lo[d3] = id3;
      s += p.mag_g[(size_t)comp * p.ncells +
                   ind_cidx(p, lo[2], lo[1], lo[0])];
    }
  }
  return 0.25 * s;
}

/* mag_g_edge without wrapping (host :676-690). Used by the curl pass
   where the face-range guards ensure all four cells are in range. */
KOKKOS_INLINE_FUNCTION
double ind_g_edge(const IndP &p, int d, const int cc[3], int de, int comp) {
  int lo[3] = {cc[0], cc[1], cc[2]};
  double s = 0.0;
  for (int a = 0; a < 2; a++) {
    lo[d] = cc[d] - 1 + a;
    for (int b = 0; b < 2; b++) {
      lo[de] = cc[de] - 1 + b;
      s += p.mag_g[(size_t)comp * p.ncells +
                   ind_cidx(p, lo[2], lo[1], lo[0])];
    }
  }
  return 0.25 * s;
}

}  // anonymous namespace

/* Phase 1: u_+ snapshot (host :607-659). One thread per cell, same
   degenerate-axis remap, same density floor, same ionized fallback. */
extern "C" int xmf_dev_mag_up_snapshot(const XmfRun *run) {
  if (!run || !run->bf[0])
    return 0;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0;
  if (dv->mag_up.extent(0) == 0)
    return 0;
  try {
    IndP p;
    p.fields = dv->fields.data();
    p.mag_up = dv->mag_up.data();
    p.mag_g = nullptr;
    for (int d = 0; d < 3; d++) {
      p.bf[d] = nullptr;
      p.ef[d] = nullptr;
      p.A[d] = nullptr;
      p.dV[d] = nullptr;
    }
    p.charge = dv->charge.data();
    p.mass = dv->mass_code.data();
    p.nfluid = run->nfluid;
    p.ng = run->ng;
    p.nx = run->nx;
    p.ny = run->ny;
    p.nz = run->nz;
    p.nxtot = run->nxtot;
    p.nytot = run->nytot;
    p.nztot = run->nztot;
    p.ncells = run->ncells;
    for (int f = 0; f < 6; f++)
      p.bc[f] = 0;
    size_t ncells = run->ncells;
    Kokkos::parallel_for(
        "xmf_dev_mag_up_snapshot", ncells,
        KOKKOS_LAMBDA(const size_t c) {
          /* Degenerate-axis active-slice remap (host :620-635). */
          int i = (int)(c % (size_t)p.nxtot);
          int j = (int)((c / (size_t)p.nxtot) % (size_t)p.nytot);
          int k = (int)(c / ((size_t)p.nxtot * (size_t)p.nytot));
          int ir = (p.nx == 1) ? p.ng : i;
          int jr = (p.ny == 1) ? p.ng : j;
          int kr = (p.nz == 1) ? p.ng : k;
          size_t cr = ind_cidx(p, kr, jr, ir);
          double ne = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
          int ionized = 0;
          for (int f = 0; f < p.nfluid; f++) {
            double ns, rho;
            if (p.charge[f] <= 0.0)
              continue;
            rho = p.fields[((size_t)f * XMF_SLOT_NVAR + XMF_SLOT_RHO) *
                               p.ncells +
                           cr];
            if (rho > XMF_RHO_FLOOR)
              ionized = 1;
            ns = p.charge[f] * rho / p.mass[f];
            ne += ns;
            mx += ns *
                  p.fields[((size_t)f * XMF_SLOT_NVAR + XMF_SLOT_VX) *
                               p.ncells +
                           cr];
            my += ns *
                  p.fields[((size_t)f * XMF_SLOT_NVAR + XMF_SLOT_VY) *
                               p.ncells +
                           cr];
            mz += ns *
                  p.fields[((size_t)f * XMF_SLOT_NVAR + XMF_SLOT_VZ) *
                               p.ncells +
                           cr];
          }
          if (ionized && ne > 0.0) {
            p.mag_up[c] = mx / ne;
            p.mag_up[p.ncells + c] = my / ne;
            p.mag_up[2 * p.ncells + c] = mz / ne;
          } else {
            p.mag_up[c] = 0.0;
            p.mag_up[p.ncells + c] = 0.0;
            p.mag_up[2 * p.ncells + c] = 0.0;
          }
        });
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

/* Phase 2: full CT induction on device (host :839-1024). Four sub-phases:
   A: cell-centred B and G = u_+ x B (one kernel over all cells).
   B: edge E = -G (interior 4-cell average + periodic ghost wrap).
   C: discrete curl update bf += dt*(D1-D2).
   D: non-periodic E ghost fill (xmf_bc_fill_e_ghosts_device).
   Each phase is a separate kernel to enforce the dependency chain. */
extern "C" int xmf_dev_mag_induce(XmfRun *run, double dt) {
  if (!run || !run->bf[0])
    return 0;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0;
  if (dv->mag_up.extent(0) == 0 || dv->mag_g.extent(0) == 0)
    return 0;
  if (dt == 0.0)
    return 0;
  try {
    IndP p;
    p.fields = dv->fields.data();
    p.mag_up = dv->mag_up.data();
    p.mag_g = dv->mag_g.data();
    for (int d = 0; d < 3; d++) {
      p.bf[d] = dv->bf[d].data();
      p.ef[d] = dv->ef[d].data();
      p.A[d] = dv->A[d].data();
      p.dV[d] = dv->dV[d].data();
    }
    p.charge = dv->charge.data();
    p.mass = dv->mass_code.data();
    p.nfluid = run->nfluid;
    p.ng = run->ng;
    p.nx = run->nx;
    p.ny = run->ny;
    p.nz = run->nz;
    p.nxtot = run->nxtot;
    p.nytot = run->nytot;
    p.nztot = run->nztot;
    p.ncells = run->ncells;
    for (int f = 0; f < 6; f++)
      p.bc[f] = run->bc[f];

    int per0 = (run->bc[0] == XMF_BC_PERIODIC);
    int per1 = (run->bc[2] == XMF_BC_PERIODIC);
    int per2 = (run->bc[4] == XMF_BC_PERIODIC);

    /* Phase A: cell-centred B and G = u_+ x B (host :852-884). */
    {
      size_t nwork =
          (size_t)p.nztot * (size_t)p.nytot * (size_t)p.nxtot;
      Kokkos::parallel_for(
          "xmf_ind_phaseA_G", nwork,
          KOKKOS_LAMBDA(const size_t idx) {
            int i = (int)(idx % (size_t)p.nxtot);
            int j = (int)((idx / (size_t)p.nxtot) % (size_t)p.nytot);
            int k = (int)(idx / ((size_t)p.nxtot * (size_t)p.nytot));
            size_t c = ind_cidx(p, k, j, i);
            int cc[3] = {i, j, k};
            double bcx = 0.0, bcy = 0.0, bcz = 0.0;
            for (int dd = 0; dd < 3; dd++) {
              int hi[3] = {cc[0], cc[1], cc[2]};
              double v;
              hi[dd] += 1;
              v = 0.5 * (p.bf[dd][ind_bidx(p, dd, cc)] +
                         p.bf[dd][ind_bidx(p, dd, hi)]);
              if (dd == 0)
                bcx = v;
              else if (dd == 1)
                bcy = v;
              else
                bcz = v;
            }
            {
              double ux = p.mag_up[c];
              double uy = p.mag_up[p.ncells + c];
              double uz = p.mag_up[2 * p.ncells + c];
              p.mag_g[c] = uy * bcz - uz * bcy;
              p.mag_g[p.ncells + c] = uz * bcx - ux * bcz;
              p.mag_g[2 * p.ncells + c] = ux * bcy - uy * bcx;
            }
          });
    }

    /* Phase B: edge E = -G (host :888-975). Per-component launches. */
    for (int d = 0; d < 3; d++) {
      int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
      int nd = ind_ntot(p, d);
      int nd1 = ind_ntot(p, d1);
      int nd2 = ind_ntot(p, d2);
      /* Interior E fill: k in [0,nd), j in [1,nd1), i in [1,nd2).
         Host :907-916. */
      Kokkos::parallel_for(
          "xmf_ind_E_interior",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 1, 1}, {nd, nd1, nd2}),
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            int cc[3];
            cc[d] = k;
            cc[d1] = j;
            cc[d2] = i;
            p.ef[d][ind_eidx(p, d, cc)] =
                -ind_g_edge_p(p, d, cc, d1, d, per0, per1, per2);
          });
      /* Periodic ghost edges (host :922-968). */
      if (per0 || per1 || per2) {
        int per_d1 = (d1 == 0) ? per0 : (d1 == 1) ? per1 : per2;
        int per_d2 = (d2 == 0) ? per0 : (d2 == 1) ? per1 : per2;
        /* d1 ghost edges over interior d2 range (host :932-941). */
        if (per_d1) {
          Kokkos::parallel_for(
              "xmf_ind_E_per_d1",
              Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                  {0, 1, 0}, {nd, nd2, 2}),
              KOKKOS_LAMBDA(const int k, const int i, const int side) {
                int cc[3];
                cc[d] = k;
                cc[d2] = i;
                cc[d1] = side * nd1;
                p.ef[d][ind_eidx(p, d, cc)] =
                    -ind_g_edge_p(p, d, cc, d1, d, per0, per1, per2);
              });
        }
        /* d2 ghost edges over interior d1 range (host :944-953). */
        if (per_d2) {
          Kokkos::parallel_for(
              "xmf_ind_E_per_d2",
              Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                  {0, 1, 0}, {nd, nd1, 2}),
              KOKKOS_LAMBDA(const int k, const int j, const int side) {
                int cc[3];
                cc[d] = k;
                cc[d1] = j;
                cc[d2] = side * nd2;
                p.ef[d][ind_eidx(p, d, cc)] =
                    -ind_g_edge_p(p, d, cc, d1, d, per0, per1, per2);
              });
        }
        /* Corner ghost edges (host :956-966). */
        if (per_d1 && per_d2) {
          Kokkos::parallel_for(
              "xmf_ind_E_per_corner",
              Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                  {0, 0, 0}, {nd, 2, 2}),
              KOKKOS_LAMBDA(const int k, const int jd, const int id) {
                int cc[3];
                cc[d] = k;
                cc[d1] = jd * nd1;
                cc[d2] = id * nd2;
                p.ef[d][ind_eidx(p, d, cc)] =
                    -ind_g_edge_p(p, d, cc, d1, d, per0, per1, per2);
              });
        }
      }
    }

    /* Phase C: discrete curl update bf += dt*(D1-D2) (host :982-1017).
       Per-component launches. The face-range guards and A/dV geometry
       match the host code exactly. */
    for (int d = 0; d < 3; d++) {
      int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
      int nd = ind_ntot(p, d);
      int nd1 = ind_ntot(p, d1);
      int nd2 = ind_ntot(p, d2);
      size_t nwork =
          (size_t)p.nztot * (size_t)p.nytot * (size_t)p.nxtot;
      Kokkos::parallel_for(
          "xmf_ind_curl", nwork,
          KOKKOS_LAMBDA(const size_t idx) {
            int i = (int)(idx % (size_t)p.nxtot);
            int j = (int)((idx / (size_t)p.nxtot) % (size_t)p.nytot);
            int k = (int)(idx / ((size_t)p.nxtot * (size_t)p.nytot));
            int cc[3] = {i, j, k};
            double D1 = 0.0, D2 = 0.0;
            /* Face range guards (host :993-998). */
            if (cc[d] < 1 || cc[d] > nd - 1)
              return;
            if (cc[d1] < 1 || cc[d1] > nd1 - 2)
              return;
            if (cc[d2] < 1 || cc[d2] > nd2 - 2)
              return;
            /* d1 derivative (host :999-1006). */
            if (p.dV[d1][cc[d1]] != 0.0) {
              int up[3] = {cc[0], cc[1], cc[2]};
              double g_lo = ind_g_edge(p, d, cc, d1, d2);
              up[d1] += 1;
              D1 = (p.A[d1][cc[d1] + 1] *
                        ind_g_edge(p, d, up, d1, d2) -
                    p.A[d1][cc[d1]] * g_lo) /
                   p.dV[d1][cc[d1]];
            }
            /* d2 derivative (host :1007-1014). */
            if (p.dV[d2][cc[d2]] != 0.0) {
              int up[3] = {cc[0], cc[1], cc[2]};
              double g_lo = ind_g_edge(p, d, cc, d2, d1);
              up[d2] += 1;
              D2 = (p.A[d2][cc[d2] + 1] *
                        ind_g_edge(p, d, up, d2, d1) -
                    p.A[d2][cc[d2]] * g_lo) /
                   p.dV[d2][cc[d2]];
            }
            p.bf[d][ind_bidx(p, d, cc)] += dt * (D1 - D2);
          });
    }

    /* Phase D: non-periodic E ghost fill (existing Port B kernel). */
    xmf_bc_fill_e_ghosts_device(run);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

/* ---- Port C: device pin sync & apply ------------------------------------- */

extern "C" int xmf_dev_pin_sync(XmfRun *run) {
  if (!run)
    return -1;
  DeviceState *dv = dev(run);
  if (!dv)
    return 0;
  try {
    /* Push the pin mask and values to the device Views (extent 0 when no
       pin is loaded). */
    if (run->pin_mask && run->pin_vals) {
      size_t ncells = run->ncells;
      size_t val_n = (size_t)run->nfluid * XMF_SLOT_NVAR * ncells;
      dv->pin_mask = Kokkos::View<int *>("xmf_pin_mask", ncells);
      dv->pin_vals = Kokkos::View<double *>("xmf_pin_vals", val_n);
      push_i(dv->pin_mask, run->pin_mask, ncells);
      push_d(dv->pin_vals, run->pin_vals, val_n);
    } else {
      dv->pin_mask = Kokkos::View<int *>();
      dv->pin_vals = Kokkos::View<double *>();
    }
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

/* Device pin apply: overwrite masked cells' fluid five slots with the
   pinned values. P_e is never touched. B is never touched -- the CT
   path's B is NOT pinned here; extending pin to B is a future slice
   that needs a hook after mag_induce, not a drive-by addition here. */
extern "C" void xmf_pin_apply_device(XmfRun *run) {
  if (!run || !run->dev)
    return;
  DeviceState *dv = dev(run);
  if (dv->pin_mask.extent(0) == 0 || dv->pin_vals.extent(0) == 0)
    return;
  double *fld_d = dv->fields.data();
  double *pv_d = dv->pin_vals.data();
  int *pm_d = dv->pin_mask.data();
  size_t ncells = run->ncells;
  size_t nwork = (size_t)run->nfluid * XMF_SLOT_NVAR * ncells;
  try {
    Kokkos::parallel_for(
        "xmf_pin_apply", nwork, KOKKOS_LAMBDA(const size_t idx) {
          size_t c = idx % ncells;
          if (pm_d[c])
            fld_d[idx] = pv_d[idx];
        });
  } catch (...) {
    return; /* kernel launch failure: fields keep their previous values */
  }
}

namespace xmf {

IrCtx ir_ctx_device(const XmfRun *run) {
  DeviceState *dv = dev(run);
  IrCtx g;
  g.fields = dv->fields.data();
  g.pe = run->pe ? dv->pe.data() : NULL;
  g.mass_code = dv->mass_code.data();
  g.charge = dv->charge.data();
  g.x1 = dv->x[0].data();
  g.x2 = dv->x[1].data();
  g.x3 = dv->x[2].data();
  g.flux = dv->flux.extent(0) > 0 ? dv->flux.data() : NULL;
  g.finf = dv->finf.extent(0) > 0 ? dv->finf.data() : NULL;
  g.nfluid = run->nfluid;
  g.nbeam = dv->nbeam;
  g.rad_enabled = run->rad ? run->rad->enabled : 0;
  g.nxtot = run->nxtot;
  g.nytot = run->nytot;
  g.ncells = run->ncells;
  g.unit_P = run->unit_P;
  return g;
}

}  // namespace xmf
