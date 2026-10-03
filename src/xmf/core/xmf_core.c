/* Minimal ideal-Euler kernel for the Python-package demo, on the real
   (nz,ny,nx) field layout. Direction-split first-order Godunov (PCM +
   RK1) with three switchable Riemann solvers (hll/hllc/tvdlf) and a
   three-direction CFL, per-face boundary ghosts (A3). Not the PLUTO tree.

   A4.5 split: this file holds only the four-step public API
   (setup / step / fields / finalize) plus the one-line xmf_time / xmf_dt /
   xmf_ncell accessors. This file stays pure C (C1a). The rest of the
   kernel lives in sibling translation units sharing the internal header
   xmf_run.h:
       xmf_mesh.c   -- axis_* mesh construction and xmf_mesh_* queries;
       xmf_state.c  -- xmf_state_* pack/unpack and P_e helpers;
       xmf_bc.c     -- fill_bc family, xmf_bc_type / xmf_fill_bc,
                       xmf_bc_set_user (A5 floor), xmf_bc_load_tapes
                       (A3 supplement boundary assignment tapes);
       xmf_mag.c    -- magnetic state: the staggered B/E storage (M2),
                       the xmf_mag_* load/pack/unpack ports, the
                       face-centred B ghost fills (M3), the M5
                       constrained transport (u_+ snapshot, edge E,
                       mag_induce, divB diagnostic) and the B SSP
                       stage helpers;
       xmf_hydro.cpp -- face states + Riemann + sweeps + CFL + the
                       whole-step L_f and SSP stage helpers xmf_step calls;
                       C1a/C1b: its transverse-line loops ride Kokkos
                       parallel_for on the default host execution space
                       (OpenMP since C1b, Serial the C1a baseline).
       xmf_ir.cpp   -- source-tape stack machine (dual numbers, A4.7):
                       xmf_ir_load / apply / ntape / jac / nstate /
                       eval_tape helpers; xmf_step never calls it. C2:
                       the evaluator lives in xmf_ir_eval.hpp (device
                       callable) and the host module wraps it.
       xmf_scan.cpp -- radiation line scan (A5), own module.
       xmf_source.cpp -- implicit source step (A5), own module.
       xmf_device.cpp -- C2 device-state owner: the owning device Views
                       and the xmf_dev_* mirror hooks; not a physics
                       module.
       xmf_kokkos.cpp -- C1a lifecycle base ONLY: the idempotent
                       Kokkos::initialize guard (xmf_kokkos_ensure) plus
                       the Host View helpers; not a physics module.

   ---- Layout & index convention ----------------------------------------
   Cells are indexed (k,j,i) == (x3,x2,x1); i is the fastest-running index.
   Every direction, degenerate or not, carries ng ghost cells on each side:
       nxtot = n[0] + 2*ng,  nytot = n[1] + 2*ng,  nztot = n[2] + 2*ng
   Interior cells are k in [ng, ng+nz), j in [ng, ng+ny), i in [ng, ng+nx).
   A 1D run is the degeneration (nz,ny,nx) = (1,1,nx); the loops below still
   walk all three indices, they simply have extent 1 in k and j.

   Fields: one flat buffer laid out [fluid][slot][k][j][i]. Each fluid (the
   count comes from XmfConfig.nfluid; the kernel never maps names to
   indices) owns XMF_SLOT_NVAR slots, in the fixed order (n, vx, vy, vz, P).
   In 1D the idle transverse velocities vy,vz stay 0.

   Public surface stays the four-step API. The xmf_mesh_* symbols are
   read-only queries in the xmf_ncell/xmf_time helper style: they copy one
   direction's mesh arrays into caller-owned storage and change nothing.
   ---------------------------------------------------------------------- */
#include "xmf_run.h"

#include <stdlib.h>

#include "xmf_kokkos.h"

XmfRun *xmf_setup(const XmfConfig *cfg, const double *xf1, const double *xf2,
                  const double *xf3) {
  XmfRun *run;
  int i, j, k, d, nx, ny, nz, ng, nfluid;
  int n[3];
  const double *xfa[3];
  if (!cfg || cfg->schema_version != XMF_SCHEMA)
    return NULL;
  /* The per-direction active face lattices (n[d]+1 absolute interface
     positions each) are mandatory setup input -- never part of the flat
     blob. The config checks above run first, so a bad blob still fails
     before any face pointer is touched. */
  if (!xf1 || !xf2 || !xf3)
    return NULL;
  if (cfg->n[0] < 1) /* bad grid size, negatives included */
    return NULL;
  nx = cfg->n[0];
  ny = cfg->n[1];
  nz = cfg->n[2];
  ng = cfg->ng;
  nfluid = cfg->nfluid;
  /* Existing sanity checks are only ever tightened, never removed. */
  if (nx < 8 || ny < 1 || nz < 1 || ng < 1 || nfluid < 1)
    return NULL;
  if (cfg->gamma <= 1.0 || cfg->cfl <= 0.0)
    return NULL;
  if (cfg->riemann < XMF_RIEMANN_HLL || cfg->riemann > XMF_RIEMANN_HLLD)
    return NULL; /* only hll/hllc/tvdlf/hlld are implemented (no
                    roe/exact). M4: hlld is the Miyoshi-Kusano MHD
                    solver -- gate-open ion sweeps dispatch to the
                    8-component MHD branch, gate-closed runs and
                    neutrals ride its zero-field limit. The magnetic
                    gate's own refusal of hllc lives in xmf_mag_load:
                    hllc's wave fan carries no magnetic waves and is
                    never silently swapped for an MHD solver */
  if (cfg->rk < XMF_RK1 || cfg->rk > XMF_RK3)
    return NULL; /* only rk1/rk2/rk3 are implemented; rk4 is enumerated
                    (XMF_RK4) on purpose but refused here, and there is no
                    silent fallback to RK1 for any unknown value */
  if (cfg->recon < XMF_RECON_PCM || cfg->recon > XMF_RECON_WENO3)
    return NULL; /* only pcm/plm/weno3 exist; an unknown value fails
                    setup instead of silently running as PCM */
  if (cfg->recon >= XMF_RECON_PLM && ng < 2)
    return NULL; /* plm/weno3 stencils reach i-1..i+2 across each face,
                    which needs two ghost layers on every side */
  if (cfg->electron_mode != XMF_ELEC_OFF &&
      cfg->electron_mode != XMF_ELEC_PRESSURE_ONLY)
    return NULL;
  if (cfg->shock_heat < XMF_SHOCK_HEAT_IONS_ONLY ||
      cfg->shock_heat > XMF_SHOCK_HEAT_ELECTRON)
    return NULL; /* only the three tiers exist; an unknown value fails
                    setup instead of silently running as ions-only */
  if (cfg->x1max <= cfg->x1min)
    return NULL;
  if (cfg->x2max <= cfg->x2min || cfg->x3max <= cfg->x3min)
    return NULL;
  for (i = 0; i < 6; i++)
    if (cfg->bc[i] < XMF_BC_REFLECT || cfg->bc[i] > XMF_BC_PERIODIC)
      return NULL; /* out-of-range per-face boundary enum */

  /* C1a lifecycle: the hydro/source/scan loops ride the Kokkos interface,
     so the execution space must be ready before the first step. The guard
     is idempotent -- every setup after the first in a process is a no-op
     (pytest runs 100 setup/finalize cycles in one process). xmf_finalize
     never calls Kokkos::finalize; teardown is the process-exit handler's
     job. A failed initialisation fails setup. */
  if (xmf_kokkos_ensure())
    return NULL;

  run = (XmfRun *)calloc(1, sizeof(XmfRun));
  if (!run)
    return NULL;
  run->nx = nx;
  run->ny = ny;
  run->nz = nz;
  run->ng = ng;
  run->nfluid = nfluid;
  run->geometry = cfg->geometry;
  run->nxtot = nx + 2 * ng;
  run->nytot = ny + 2 * ng;
  run->nztot = nz + 2 * ng;
  run->ncells = (size_t)run->nxtot * (size_t)run->nytot * (size_t)run->nztot;
  run->gamma = cfg->gamma;
  run->cfl = cfg->cfl;
  run->riemann = cfg->riemann;
  run->rk = cfg->rk; /* validated above: rk1/rk2/rk3 dispatched in xmf_step */
  run->recon = cfg->recon; /* validated above; sweep_line dispatches on it
                              for every face: pcm keeps the adjacent cell
                              centres, plm (van Leer) and weno3 (Jiang-Shu)
                              reconstruct the face states. No silent PCM. */
  run->shock_heat = cfg->shock_heat; /* validated above (enum XmfShockHeat);
                                        the former reserved0 blob slot --
                                        byte position frozen, name moved.
                                        Read only by the hydro sweeps, and
                                        only when the electron closure gate
                                        is on. */
  run->dt_max = cfg->dt_max;
  /* No run-wide scalar spacings any more (A1 stretch supplement): the
     per-cell mesh dx/A/dV arrays built below are the only grid authority
     the hydro operator and CFL ever read. */
  run->t = 0.0;
  run->dt = 0.0;
  /* Per-face boundary types are owned by the run from here on; the
     config blob does not need to stay alive past setup. */
  for (i = 0; i < 6; i++)
    run->bc[i] = cfg->bc[i];

  n[0] = nx;
  n[1] = ny;
  n[2] = nz;
  xfa[0] = xf1;
  xfa[1] = xf2;
  xfa[2] = xf3;

  for (d = 0; d < 3; d++) {
    if (!axis_alloc(&run->mesh[d], n[d] + 2 * ng)) {
      /* Every pointer is NULL-initialised, so the shared cleanup path is
         safe no matter where allocation stopped. */
      xmf_finalize(run);
      return NULL;
    }
  }
  run->fields = dvec((size_t)nfluid * XMF_SLOT_NVAR * run->ncells);
  if (!run->fields) {
    xmf_finalize(run);
    return NULL;
  }
  /* A5 physics defaults: every mass_code and every unit starts at 1.0,
     so a run that never calls xmf_phys_load keeps the pre-A5 numerics
     bit for bit (LOAD n returns slot 0 unchanged). The helper replaces
     these values; xmf_finalize frees the block. */
  run->mass_code = dvec((size_t)nfluid);
  if (!run->mass_code) {
    xmf_finalize(run);
    return NULL;
  }
  for (i = 0; i < nfluid; i++)
    run->mass_code[i] = 1.0;
  run->unit_density = 1.0;
  run->unit_velocity = 1.0;
  run->unit_length = 1.0;
  run->unit_P = 1.0;
  /* B1 charge defaults: every q_s starts at 0.0 (dvec zero-fills), so the
     electron closure gate stays off until xmf_charge_load lands -- the
     pre-B1 hydro path, bit for bit. The helper replaces these values;
     xmf_finalize frees the block. */
  run->charge = dvec((size_t)nfluid);
  if (!run->charge) {
    xmf_finalize(run);
    return NULL;
  }
  /* Optional electron-pressure slot (A2): one separate block, same length
     as every fluid slot (ghosts included), calloc'd to zero. Not allocated
     at all when electrons are off -- run->pe stays NULL from the calloc of
     the run itself. */
  if (cfg->electron_mode == XMF_ELEC_PRESSURE_ONLY) {
    run->pe = dvec(run->ncells);
    if (!run->pe) {
      xmf_finalize(run);
      return NULL;
    }
  }

  for (d = 0; d < 3; d++) {
    /* Active faces in, ghosts padded with the adjacent active width,
       centres/widths derived; the authoritative monotonicity check lives
       inside (strictly increasing, else setup fails). */
    if (!axis_fill_faces(&run->mesh[d], n[d], ng, xfa[d])) {
      xmf_finalize(run);
      return NULL;
    }
    axis_fill_metrics(&run->mesh[d], cfg->geometry, d);
  }
  /* Spherical x1: every interface, ghosts included, must stay strictly
     positive -- an r <= 0 face anywhere on the lattice fails setup. */
  if (cfg->geometry == XMF_GEOM_SPHERICAL) {
    const XmfAxis *ax1 = &run->mesh[0];
    for (i = 0; i <= ax1->ntot; i++)
      if (ax1->xf[i] <= 0.0) {
        xmf_finalize(run);
        return NULL;
      }
  }

  /* Initial Riemann states into fluid 0; the discontinuity is located with
     the 1-D x1 centre array (no 3-D coordinate field exists). Idle slots
     and extra fluids stay zero from calloc. */
  {
    const double *x1 = run->mesh[0].x;
    double *fn = fld(run, 0, XMF_SLOT_RHO);
    double *fv = fld(run, 0, XMF_SLOT_VX);
    double *fp = fld(run, 0, XMF_SLOT_P);
    for (k = 0; k < run->nztot; k++)
      for (j = 0; j < run->nytot; j++)
        for (i = 0; i < run->nxtot; i++) {
          size_t c = cidx(run, k, j, i);
          if (x1[i] < cfg->x_disc) {
            fn[c] = cfg->rho_l;
            fv[c] = cfg->v_l;
            fp[c] = cfg->p_l;
          } else {
            fn[c] = cfg->rho_r;
            fv[c] = cfg->v_r;
            fp[c] = cfg->p_r;
          }
        }
  }

  fill_bc(run);
  run->dt = compute_dt(run);
  /* C2: the device state (owning Views of fields / P_e / tables / mesh)
     is attached once, after the initial fill_bc + compute_dt, and pushed
     from the finished mirrors. Allocation failure fails setup. Idempotent
     inside the hook; xmf_finalize detaches it again. */
  if (xmf_dev_attach(run)) {
    xmf_finalize(run);
    return NULL;
  }
  /* Port B: push any setup-time user-BC floor / tapes / magnetic floor to
     the device Views (usually empty at this point; loaders call
     xmf_dev_bc_sync themselves after setup returns). The host fill_bc
     above already wrote the mirrors; now fill the device ghosts so the
     first step sees a consistent device state. */
  xmf_dev_bc_sync(run);
  xmf_bc_fill_device(run);
  return run;
}

int xmf_step(XmfRun *run) {
  int ele;
  double dt;
  /* C2: the RK stage buffers U0/Us and the B2 heat scratch are DEVICE
     allocations (xmf_dev_dvec / xmf_dev_dfree). They stay opaque void*
     here -- the pure-C orchestration never dereferences a device
     pointer; the hydro module's kernels do. */
  void *heat = NULL, *U0 = NULL, *Us = NULL;
  /* M4: separate stage buffers for the face-centred B (three blocks,
     ghosts included). The fluid U0/Us length stays untouched; B rides
     the very same Heun / Shu-Osher coefficients beside every
     cons_combine. Never allocated while the magnetic gate is closed;
     RK1 stays in-place (no buffers at all). */
  void *B0[3] = {NULL, NULL, NULL};
  if (!run)
    return -1;
  /* A5 orchestration: every RK stage runs the full three-step sequence
     scan -> hydro -> source around the whole-step operator L_f. The
     radiation scan fills the transmitted beam fluxes F_b from the
     current absorber density (a no-op without a loaded, enabled beam
     state); the semi-implicit source step advances the fields by one
     linearised backward-Euler source increment (a no-op with zero tapes
     loaded, so the pure-hydro path stays bit-for-bit). F_b and the grid
     coordinates stay frozen inside the source solve. With nothing
     loaded, scan + source contribute nothing and the step below is
     bit-for-bit the pre-A5 hydro step. */
  /* dt is computed once per full step from U^n (dt_max still caps it) and
     shared by every RK stage; t advances once, after the successful step.
     The CFL is the per-cell form of the A1 stretch supplement --
     cfl / max_cell sum_d (s_{d,cell} / dx_{d,cell}) over active
     directions (n_d > 1), widths from mesh[d].dx -- and stays ahead of
     the hydro operator. Port C: compute_dt dispatches on run->dev; with
     a device attached the reduction runs on the device Views (no pull
     needed -- the previous step left the device as the fresh copy);
     without a device the serial HOST loop runs over the host mirror. */
  dt = compute_dt(run);
  run->dt = dt;
  /* C1a: the per-line sweep workspaces (q / qe / Uc / Un) moved into the
     hydro module itself -- one row per transverse line, allocated and
     freed inside the Kokkos loop scope. Nothing step-wide is preallocated
     for them any more. */
  ele = elec_closure_active(run);
  /* B2 shock-heating share: the by-charge/electron tiers need one
     ncells-long scratch for the per-direction f_s*H share destined for
     P_e. ions-only keeps heat NULL on purpose -- the sweeps then run the
     B1 two-pass path bit for bit. */
  if (ele && run->shock_heat != XMF_SHOCK_HEAT_IONS_ONLY) {
    heat = xmf_dev_dvec(run->ncells);
    if (!heat)
      return -1;
  }

  if (run->rk == XMF_RK1) {
    /* U^{n+1} = U^n + dt*L(U^n): one in-place whole-step L_f wrapped by
       the A5 scan/source pair. This is bit-for-bit the operator the
       kernel ran before the RK slice when no beam state and no tapes are
       loaded. M4/M5 (magnetic gate open; every hook below is a gate-
       closed no-op): u_+ is snapped from the STAGE-START ion velocities
       before lf_all_dirs; after the sweeps and BEFORE the source step
       one CT step (mag_induce: edge E + curl-E update) advances B
       (B stays frozen DURING the sweeps, sources never touch it). RK1
       keeps B in-place. */
    xmf_rad_scan(run);
    mag_up_snapshot(run);
    if (xmf_dev_ne_stage_fill(run))
      goto fail;
    if (lf_all_dirs(run, dt, heat))
      goto fail;
    if (mag_induce(run, dt))
      goto fail;
    if (xmf_source_step(run, dt))
      goto fail;
  } else {
    /* Stage buffers hold the conservative state of every fluid slot over
       all cells plus -- when the P_e block exists -- one trailing P_e
       segment (fields_to_cons / cons_to_fields / cons_combine all carry
       it), so the SSP combinations cover the electron pressure too. */
    size_t ntot = (size_t)run->nfluid * XMF_SLOT_NVAR * run->ncells +
                  (run->pe ? run->ncells : 0);
    U0 = xmf_dev_dvec(ntot);
    Us = xmf_dev_dvec(ntot);
    if (!U0 || !Us)
      goto fail;
    /* M4: the B stage buffers beside U0 (gate-closed no-op: nothing
       allocated, every snapshot/combine below refuses to nothing). */
    if (mag_b_stage_alloc(run, B0))
      goto fail;
    fields_to_cons(run, U0); /* U^n, kept for every convex combination */
    mag_b_snapshot(run, B0); /* M4: B^n beside U^n */

    /* Stage 1: fields hold U^n, so one whole-step L_f gives
       U^(1) = U^n + dt*L(U^n) (both RK2 and RK3 start here). M4: u_+
       from the stage-start ions, one induction after the sweeps. */
    xmf_rad_scan(run);
    mag_up_snapshot(run);
    if (xmf_dev_ne_stage_fill(run))
      goto fail;
    if (lf_all_dirs(run, dt, heat))
      goto fail;
    if (mag_induce(run, dt))
      goto fail;
    if (xmf_source_step(run, dt))
      goto fail;
    fields_to_cons(run, Us);

    if (run->rk == XMF_RK3) {
      /* Stage 2: U^(2) = 3/4 U^n + 1/4 U^(1) + 1/4 dt*L(U^(1))
                        = 3/4 U^n + 1/4 (U^(1) + dt*L(U^(1))).
         The stage sweeps from U^(1) itself (Shu-Osher form), so B
         already holds B^(1) at the stage start -- its convex combine
         lands beside the fluid one, on the stage-completed B. */
      cons_to_fields(run, Us);
      xmf_rad_scan(run);
      mag_up_snapshot(run);
      if (xmf_dev_ne_stage_fill(run))
        goto fail;
      if (lf_all_dirs(run, dt, heat))
        goto fail;
      if (mag_induce(run, dt))
        goto fail;
      if (xmf_source_step(run, dt))
        goto fail;
      fields_to_cons(run, Us);
      cons_combine(run, Us, 0.75, U0, 0.25);
      mag_b_combine(run, 0.75, B0, 0.25); /* B^(2) = 3/4 B^n + 1/4 B^(1) */
    }

    /* Final stage: sweep from the combined stage state, then average. */
    cons_to_fields(run, Us);
    xmf_rad_scan(run);
    mag_up_snapshot(run);
    if (xmf_dev_ne_stage_fill(run))
      goto fail;
    if (lf_all_dirs(run, dt, heat))
      goto fail;
    if (mag_induce(run, dt))
      goto fail;
    if (xmf_source_step(run, dt))
      goto fail;
    fields_to_cons(run, Us);
    if (run->rk == XMF_RK2) {
      /* Heun: U^{n+1} = 1/2 U^n + 1/2 U^(1) + 1/2 dt*L(U^(1))
                       = 1/2 U^n + 1/2 (U^(1) + dt*L(U^(1))). */
      cons_combine(run, Us, 0.5, U0, 0.5);
      mag_b_combine(run, 0.5, B0, 0.5);
    } else { /* XMF_RK3 */
      /* Shu-Osher: U^{n+1} = 1/3 U^n + 2/3 U^(2) + 2/3 dt*L(U^(2))
                            = 1/3 U^n + 2/3 (U^(2) + dt*L(U^(2))). */
      cons_combine(run, Us, 1.0 / 3.0, U0, 2.0 / 3.0);
      mag_b_combine(run, 1.0 / 3.0, B0, 2.0 / 3.0);
    }
    cons_to_fields(run, Us);
  }

  xmf_dev_dfree(heat); /* NULL-safe: only the by-charge/electron share
                          allocates it */
  xmf_dev_dfree(U0); /* NULL-safe: rk1 never allocates the stage buffers */
  xmf_dev_dfree(Us);
  mag_b_stage_free(B0); /* NULL-safe: gate closed / rk1 keep all NULL */
  run->t += dt;
  return 0;

fail:
  /* A failed stage (a sweep workspace allocation, the induction scratch
     or the implicit source solve) leaves the run untouched in time: t
     does not advance. */
  xmf_dev_dfree(heat);
  xmf_dev_dfree(U0);
  xmf_dev_dfree(Us);
  mag_b_stage_free(B0);
  return -1;
}

double xmf_time(const XmfRun *run) { return run ? run->t : 0.0; }
double xmf_dt(const XmfRun *run) { return run ? run->dt : 0.0; }
int xmf_ncell(const XmfRun *run) { return run ? run->nx : 0; }

void xmf_fields(const XmfRun *run, double *x, double *rho, double *v, double *p) {
  int i;
  size_t base;
  const double *fn, *fv, *fp, *x1;
  if (!run)
    return;
  /* C2 sync point: the host mirror may lag the device after a step; pull
     the fields back before reading them out. */
  xmf_dev_pull(run);
  /* Interior line along x1 at the first interior (k,j); for a 1D run that
     is the single (ng,ng) cell in the degenerate directions. Coordinates
     come from the 1-D x1 centre array. */
  base = cidx(run, run->ng, run->ng, run->ng);
  x1 = run->mesh[0].x;
  fn = fld(run, 0, XMF_SLOT_RHO);
  fv = fld(run, 0, XMF_SLOT_VX);
  fp = fld(run, 0, XMF_SLOT_P);
  for (i = 0; i < run->nx; i++) {
    x[i] = x1[run->ng + i];
    rho[i] = fn[base + i];
    v[i] = fv[base + i];
    p[i] = fp[base + i];
  }
}

void xmf_finalize(XmfRun *run) {
  int d;
  if (!run)
    return;
  /* C2: release the device state FIRST (the owning Views; NULL-safe
     before attach / after detach). The Kokkos runtime itself stays
     alive -- Kokkos::finalize is the process-exit handler's job, never
     xmf_finalize's. */
  xmf_dev_detach(run);
  for (d = 0; d < 3; d++) {
    free(run->mesh[d].x);
    free(run->mesh[d].xf);
    free(run->mesh[d].dx);
    free(run->mesh[d].A);
    free(run->mesh[d].dV);
  }
  free(run->fields);
  free(run->pe); /* NULL-safe: electrons off never allocate it */
  free(run->mass_code); /* A5 physics block (default 1.0 per fluid) */
  free(run->charge);    /* B1 charge table (default 0.0 per fluid) */
  xmf_rad_clear(run);   /* beam state (A5); NULL-safe when never loaded */
  free(run->user_bc);   /* user boundary floor (A5); NULL-safe */
  free(run->pin_mask);  /* Port C: frozen-cell pin mask; NULL-safe */
  free(run->pin_vals);  /* Port C: frozen-cell pin values; NULL-safe */
  xmf_mag_clear(run);   /* M block 1 magnetic blocks + tangential floor;
                           NULL-safe when the gate never opened */
  xmf_bc_clear_tapes(run); /* per-face boundary tapes (A3 supplement);
                              NULL-safe, kept apart from run->tapes */
  xmf_ir_clear(run); /* source tapes (A4.6); NULL-safe when none loaded */
  free(run);
}
