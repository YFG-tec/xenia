/* Source-tape stack machine (A4.6 split module, A4.7 forward AD --
   stays in its own translation unit, never merged back into
   xmf_core.c). C2: the file is C++ so the evaluation hot path can be
   the shared KOKKOS_INLINE implementation in xmf_ir_eval.hpp; the host
   helpers below (apply / jac / eval_tape) now wrap that very code path
   through a host IrCtx, and the load port mirrors every accepted set
   onto the device (xmf_dev_tapes_sync). The opcode set 0..9 stays
   frozen.

   Python (xmf.source_ir) compiles every sources[] expression into
   integer opcodes and packed slots (numeric contract in xmf_ir.h); this
   module is the only place in the kernel that evaluates tapes. There is
   no formula string here, no `if (gravity)` branch, no component name
   and no per-cell Python callback -- only the integer opcode loop.

   A4.7: the stack carries dual numbers {v, vdot}. The opcode set 0..9
   stays frozen -- the arithmetic is only the dual extension of A4.6, so
   with every seed at zero the .v path is bit-for-bit the A4.6 value.

   Public helpers (declared in xmf_core.h, helper level like
   xmf_state_pack -- NOT a fifth public verb; the public surface stays
   setup / step / fields / finalize):
       xmf_ir_load      -- copy compiled tapes into the run after setup;
       xmf_ir_apply     -- add every tape's value to its write target;
       xmf_ir_ntape     -- how many tapes are currently loaded;
       xmf_ir_jac       -- dense dL_c/du of one interior cell;
       xmf_ir_nstate    -- N = 5*nfluid independent variables;
       xmf_ir_eval_tape -- one tape, one seeded slot direction
                           (self-check seam, not an implicit column).

   ---- C2 sync points -------------------------------------------------------
   apply / jac / eval_tape are host-C "read interior -> evaluate -> write
   interior" ports: before the evaluation the device state is pulled into
   the mirrors (xmf_dev_pull + xmf_dev_pull_flux, since tapes may load
   F_b), after a successful apply the mirrors are pushed back. The
   internal evaluator never touches a device View directly.
   --------------------------------------------------------------------------

   ---- Locked convention (apply) -------------------------------------------
   One xmf_ir_apply call evaluates every loaded tape on every interior
   cell (ghosts are not touched) and ADDS the tape's instantaneous value
   directly to the write target (dvx -> vx += expr, dn -> n += expr,
   dP -> P += expr). It does NOT multiply by dt and does NOT run hydro;
   the gravity self-check is therefore dvx = -G*M/r^2 after exactly one
   apply. Whether a tape expresses a rate or an increment, and how dt
   enters the time stepping, is A5 semantics and is deliberately not
   invented here. xmf_step never calls apply and never forms J; with
   zero tapes loaded apply is a no-op and xmf_step stays bit-for-bit the
   pure-hydro step.

   ---- Locked convention (Jacobian, A4.7) ----------------------------------
   The independent variables u of the implicit Jacobian are exactly the
   stored per-fluid slots (n, vx, vy, vz, P), indexed f*5 + slot with
   N = 5*nfluid. Grid coordinates r/x1/x2/x3, the beam flux F_b and
   constants{} are never seeded by xmf_ir_jac: loading them always
   carries vdot == 0 and they get no column of J. xmf_ir_jac seeds each
   state slot with udot = 1 (all others 0), runs the very tapes already
   loaded, and ADDS each tape's vdot to its write-target row -- a bare
   dvx/dvy/dvz writes into every fluid's vx/vy/vz row, matching apply's
   ADDS convention. A derivative with respect to r stays reachable
   through xmf_ir_eval_tape (e.g. d(-GM/r^2)/dr = 2GM/r^3) without ever
   becoming a jac column.

   B2 extension: with the P_e block present N = 5*nfluid + 1 and the
   electron pressure P_e is the LAST unknown -- one extra column seeded
   with the packed ELEC P_e code (so n_e / T_e readers pick up their
   P_e derivatives automatically) and one extra row for dP_e tapes.
   P_e never enters the fluid five slots; with electrons off N stays
   5*nfluid and dP_e tapes are refused at load.
   ---------------------------------------------------------------------- */

#include "xmf_run.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "xmf_ir_eval.hpp"

/* Maximum stack depth of one tape evaluation; compiled expressions stay
   far below this. The device evaluator carries the same lock
   (XMF_IR_EVAL_STACK in xmf_ir_eval.hpp). */
#define XMF_IR_STACK 64

/* Host-side IrCtx over the run's mirrors -- every host evaluation
   (apply / jac / eval_tape / load-time slot checks) runs through the
   same shared stack machine as the device source step. The FLUX read
   contract: flux non-NULL iff a radiation state is loaded (flux is
   always allocated with it); rad_enabled selects the scanned flux vs
   finf exactly as the A5 slot read did. */
static xmf::IrCtx ir_ctx_host(const XmfRun *run) {
  xmf::IrCtx g;
  g.fields = run->fields;
  g.pe = run->pe;
  g.mass_code = run->mass_code;
  g.charge = run->charge;
  g.x1 = run->mesh[0].x;
  g.x2 = run->mesh[1].x;
  g.x3 = run->mesh[2].x;
  g.flux = run->rad ? run->rad->flux : NULL;
  g.finf = run->rad ? run->rad->finf : NULL;
  g.nfluid = run->nfluid;
  g.nbeam = run->rad ? run->rad->nbeam : 0;
  g.rad_enabled = run->rad ? run->rad->enabled : 0;
  g.nxtot = run->nxtot;
  g.nytot = run->nytot;
  g.ncells = run->ncells;
  g.unit_P = run->unit_P;
  return g;
}

static void free_tape_set(XmfTape *set, int ntape) {
  int t;
  if (!set)
    return;
  for (t = 0; t < ntape; t++) {
    free(set[t].ops);
    free(set[t].consts);
  }
  free(set);
}

void xmf_ir_clear(XmfRun *run) {
  if (!run)
    return;
  free_tape_set(run->tapes, run->ntape);
  run->tapes = NULL;
  run->ntape = 0;
}

/* Load a whole tape set, replacing any previously loaded one.
   header: four ints per tape -- dest_kind (enum XmfIrWrite), dest_fluid
   (fluid index, or -1 for bare dvx/dvy/dvz), nops (>= 1), nconst.
   ops: nops (opcode, argument) pairs per tape, concatenated.
   consts: nconst doubles per tape, concatenated.
   Returns ntape on success; -1 on invalid arguments or allocation
   failure (the previous set stays untouched then). ntape == 0 clears
   the set. See the file header for the apply convention this set is
   evaluated under.

   C2: a successful swap rebuilds the flattened device tape set
   (xmf_dev_tapes_sync) so the device source step evaluates exactly the
   loaded tapes. A device-sync failure fails the load with the previous
   set restored. */
int xmf_ir_load(XmfRun *run, int ntape, const int *header, const int *ops,
                const double *consts) {
  XmfTape *set;
  xmf::IrCtx g;
  int t, p;
  size_t off_ops = 0, off_cst = 0;
  if (!run || ntape < 0)
    return -1;
  if (ntape == 0) { /* empty table: clear and stay on the pure-hydro path */
    xmf_ir_clear(run);
    if (xmf_dev_tapes_sync(run))
      return -1;
    return 0;
  }
  if (!header || !ops)
    return -1;
  g = ir_ctx_host(run);
  /* consts may be NULL when the whole set carries zero constants; it is
     checked per tape below, only when actually dereferenced. */
  set = (XmfTape *)calloc((size_t)ntape, sizeof(XmfTape));
  if (!set)
    return -1;
  for (t = 0; t < ntape; t++) {
    int kind = header[4 * t + 0];
    int fl = header[4 * t + 1];
    int nops = header[4 * t + 2];
    int ncst = header[4 * t + 3];
    XmfTape *tp = &set[t];
    tp->dest_kind = kind;
    tp->dest_fluid = fl;
    tp->nops = nops;
    tp->nconst = ncst;
    /* Write kinds are the frozen range DN..DPE. The M1-appended magnetic
       names DBX/DBY/DBZ (6..8) fall OUTSIDE it on purpose: this slice
       wires no magnetic increment target, so their tapes are refused at
       load -- exactly the rk4 precedent (enumerated, never silently
       runnable). The range check below is the refusal. */
    if (kind < XMF_IR_DN || kind > XMF_IR_DPE)
      goto fail;
    if (fl == -1) {
      /* bare dvx/dvy/dvz: the same acceleration for every fluid; dP_e
         (B2) writes the electron block, which is not a fluid either. */
      if (kind == XMF_IR_DPE) {
        if (!run->pe)
          goto fail; /* electrons off: the write name is refused at load */
      } else if (kind != XMF_IR_DVX && kind != XMF_IR_DVY &&
                 kind != XMF_IR_DVZ) {
        goto fail;
      }
    } else if (kind == XMF_IR_DPE || fl < 0 || fl >= run->nfluid) {
      /* dP_e never targets a fluid index */
      goto fail;
    }
    if (nops < 1 || ncst < 0)
      goto fail;
    tp->ops = (int *)malloc(sizeof(int) * 2 * (size_t)nops);
    if (!tp->ops)
      goto fail;
    if (ncst > 0) {
      if (!consts)
        goto fail; /* only dereferenced when a tape really has constants */
      tp->consts = (double *)malloc(sizeof(double) * (size_t)ncst);
      if (!tp->consts)
        goto fail;
      memcpy(tp->consts, consts + off_cst, sizeof(double) * (size_t)ncst);
    }
    for (p = 0; p < nops; p++) {
      int op = ops[off_ops + 2 * (size_t)p];
      int arg = ops[off_ops + 2 * (size_t)p + 1];
      if (op < XMF_OP_LOAD_CONST || op > XMF_OP_LOG)
        goto fail;
      if (op == XMF_OP_LOAD_CONST) {
        if (arg < 0 || arg >= ncst)
          goto fail;
      } else if (op == XMF_OP_LOAD_SLOT) {
        if (!xmf::ir_slot_loadable(g, arg))
          goto fail;
      }
      tp->ops[2 * p] = op;
      tp->ops[2 * p + 1] = arg;
    }
    off_ops += 2 * (size_t)nops;
    off_cst += (size_t)ncst;
  }
  /* New set complete: swap it in, drop the old one, mirror to device. */
  free_tape_set(run->tapes, run->ntape);
  run->tapes = set;
  run->ntape = ntape;
  if (xmf_dev_tapes_sync(run)) {
    /* Device mirror failed: restore the previous set contractually. The
       previous set was already freed above -- the only honest state now
       is a cleared host set plus an empty device set. */
    xmf_ir_clear(run);
    xmf_dev_tapes_sync(run);
    return -1;
  }
  return ntape;

fail:
  free_tape_set(set, ntape); /* NULL pointers from calloc are free-safe */
  return -1;
}

/* One tape at cell (k,j,i) on dual numbers {v, vdot} -- the host seam.
   C2: a thin wrapper over the shared stack machine in xmf_ir_eval.hpp
   with the host IrCtx; the arithmetic and the failure contract are the
   verbatim A4.7 ones (see xmf_ir.h). */
int xmf_ir_eval(const XmfRun *run, const XmfTape *tp, int k, int j, int i,
                int seed_slot, double *out_v, double *out_vdot) {
  if (!run || !tp || !out_v)
    return -1;
  return xmf::ir_eval(ir_ctx_host(run), tp->ops, tp->consts, tp->nops,
                      tp->nconst, k, j, i, seed_slot, out_v, out_vdot);
}

/* Map a write kind (enum XmfIrWrite) to the field slot it increments.
   Written out explicitly even though the numbers currently coincide:
   the IR contract and the field layout are two different locks. Shared
   with xmf_source.c through the declaration in xmf_ir.h. */
int xmf_ir_write_slot(int kind) {
  switch (kind) {
  case XMF_IR_DN:
    return XMF_SLOT_RHO;
  case XMF_IR_DVX:
    return XMF_SLOT_VX;
  case XMF_IR_DVY:
    return XMF_SLOT_VY;
  case XMF_IR_DVZ:
    return XMF_SLOT_VZ;
  case XMF_IR_DP:
    return XMF_SLOT_P;
  default:
    return -1;
  }
}

/* One apply: every tape's instantaneous value added to its write target
   on every interior cell. No dt factor, no hydro, ghosts untouched --
   see the locked convention in the file header. Runs the dual evaluator
   with no seeded variable (seed_slot = -1, derivative not requested),
   so the .v path is bit-for-bit the A4.6 computation.

   C2 sync point: device -> mirror before the first evaluation,
   mirror -> device after the last successful one. */
int xmf_ir_apply(XmfRun *run) {
  int t, f, k, j, i;
  if (!run)
    return -1;
  xmf_dev_pull(run);       /* fields + P_e mirrors fresh from the device */
  xmf_dev_pull_flux(run);  /* tapes may read F_b */
  for (t = 0; t < run->ntape; t++) {
    const XmfTape *tp = &run->tapes[t];
    int slot;
    int f0, f1;
    if (tp->dest_kind == XMF_IR_DPE) {
      /* B2: the increment enters the electron pressure block DIRECTLY --
         no fluid slot, no projection through an ion P. xmf_ir_load only
         accepts dP_e with the P_e block present; the NULL check is the
         load-time contract repeated defensively. */
      if (!run->pe)
        return -1;
      for (k = run->ng; k < run->ng + run->nz; k++)
        for (j = run->ng; j < run->ng + run->ny; j++)
          for (i = run->ng; i < run->ng + run->nx; i++) {
            size_t c = cidx(run, k, j, i);
            double val;
            if (xmf_ir_eval(run, tp, k, j, i, -1, &val, NULL))
              return -1;
            run->pe[c] += val;
          }
      continue;
    }
    slot = xmf_ir_write_slot(tp->dest_kind);
    if (slot < 0)
      return -1;
    if (tp->dest_fluid < 0) { /* bare dvx/dvy/dvz: every fluid */
      f0 = 0;
      f1 = run->nfluid;
    } else {
      f0 = tp->dest_fluid;
      f1 = tp->dest_fluid + 1;
    }
    for (f = f0; f < f1; f++) {
      double *dst = fld(run, f, slot);
      /* Density never rides the wall (A5): a dn tape writes the number
         increment into the mass-density slot as rho += mass_code*dn.
         With the default mass_code = 1 this is bit-for-bit the pre-A5
         add, so old tests stay green. Every other slot adds unchanged. */
      double factor = (slot == XMF_SLOT_RHO) ? run->mass_code[f] : 1.0;
      for (k = run->ng; k < run->ng + run->nz; k++)
        for (j = run->ng; j < run->ng + run->ny; j++)
          for (i = run->ng; i < run->ng + run->nx; i++) {
            size_t c = cidx(run, k, j, i);
            double val;
            if (xmf_ir_eval(run, tp, k, j, i, -1, &val, NULL))
              return -1;
            dst[c] += factor * val;
          }
    }
  }
  xmf_dev_push(run); /* mirror edits land on the device */
  return 0;
}

int xmf_ir_ntape(const XmfRun *run) { return run ? run->ntape : 0; }

int xmf_ir_nstate(const XmfRun *run) {
  /* Independent variables of the implicit Jacobian: the five stored
     slots (n, vx, vy, vz, P) per fluid, indexed f*5 + slot. Grid
     coordinates r/x1/x2/x3, F_b and constants{} are never independent
     variables. B2: with the P_e block present the electron pressure is
     ONE extra Newton unknown, last (index N-1 = 5*nfluid) -- it never
     enters the fluid five slots; with electrons off N stays 5*nfluid. */
  if (!run)
    return 0;
  return 5 * run->nfluid + (run->pe ? 1 : 0);
}

/* Dense J = dL_c/du of one interior cell (see the locked convention in
   the file header and xmf_core.h). N = xmf_ir_nstate(run): for every
   fluid state column the seeded slot is the packed FLUID code of
   (f = col/5, s = col%5); B2 adds -- exactly when the P_e block exists --
   one last column seeded with the packed ELEC P_e code. Every loaded tape
   runs on that seed and its vdot is ADDED to the write-target row,
   matching apply's ADDS convention (bare dvx/dvy/dvz -> every fluid's
   vx/vy/vz row; dP_e -> the last row). Zero tapes leave the memset-zero
   J untouched.

   C2 sync point: read-only port -- device -> mirror before evaluating
   (tapes may read F_b); nothing is pushed back. */
int xmf_ir_jac(const XmfRun *run, int k, int j, int i, double *J) {
  int n, nf5, col, t, f;
  if (!run || !J)
    return -1;
  if (k < run->ng || k >= run->ng + run->nz || j < run->ng ||
      j >= run->ng + run->ny || i < run->ng || i >= run->ng + run->nx)
    return -1;
  xmf_dev_pull(run);      /* fields + P_e mirrors fresh from the device */
  xmf_dev_pull_flux(run); /* tapes may read F_b */
  n = xmf_ir_nstate(run);
  nf5 = 5 * run->nfluid;
  memset(J, 0, sizeof(double) * (size_t)n * (size_t)n);
  for (col = 0; col < n; col++) {
    int seed = col < nf5
                   ? (((int)XMF_IR_FLUID << 16) | ((col / 5) << 8) |
                      (col % 5))
                   : (((int)XMF_IR_ELEC << 16) | ((int)XMF_IR_PE << 8));
    for (t = 0; t < run->ntape; t++) {
      const XmfTape *tp = &run->tapes[t];
      int slot;
      double v, d;
      if (xmf_ir_eval(run, tp, k, j, i, seed, &v, &d))
        return -1;
      if (tp->dest_kind == XMF_IR_DPE) { /* P_e row is always the last */
        J[(size_t)(n - 1) * (size_t)n + (size_t)col] += d;
        continue;
      }
      slot = xmf_ir_write_slot(tp->dest_kind);
      if (slot < 0)
        return -1;
      if (tp->dest_fluid < 0) { /* bare dvx/dvy/dvz: every fluid's row */
        for (f = 0; f < run->nfluid; f++)
          J[(size_t)(f * 5 + slot) * (size_t)n + (size_t)col] += d;
      } else {
        J[(size_t)(tp->dest_fluid * 5 + slot) * (size_t)n + (size_t)col] +=
            d;
      }
    }
  }
  return 0;
}

/* One loaded tape at one interior cell, evaluated with a single seeded
   slot direction -- a self-check seam, not an implicit column (r seeded
   here never becomes a column of xmf_ir_jac). seed_slot is -1 (no seed)
   or any loadable packed slot code, seeded with derivative 1.

   C2 sync point: read-only port, same pull discipline as xmf_ir_jac. */
int xmf_ir_eval_tape(const XmfRun *run, int t, int k, int j, int i,
                     int seed_slot, double *out_v, double *out_vdot) {
  if (!run || t < 0 || t >= run->ntape || !out_v || !out_vdot)
    return -1;
  if (k < run->ng || k >= run->ng + run->nz || j < run->ng ||
      j >= run->ng + run->ny || i < run->ng || i >= run->ng + run->nx)
    return -1;
  xmf_dev_pull(run);
  xmf_dev_pull_flux(run);
  return xmf_ir_eval(run, &run->tapes[t], k, j, i, seed_slot, out_v,
                     out_vdot);
}
