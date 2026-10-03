/* Per-face ghost fill of the ideal-Euler kernel (A4.5 split out of
   xmf_core.c; the public surface stays xmf_core.h).

   Boundary conditions (A3): six faces in the fixed order 0=x1beg,
   1=x1end, 2=x2beg, 3=x2end, 4=x3beg, 5=x3end, one enum XmfBcType each.
   The six integers arrive in XmfConfig.bc and are copied into the run at
   setup (the config blob is not kept). reflect mirrors the interior
   (normal velocity slot odd, every other slot even); copy replicates the
   outermost interior layer; user/outflow/none are padded exactly like
   copy this slice -- defined values, never uninitialised, and user is
   never a function pointer or a per-cell callback. Every face writes the
   whole transverse plane and all five fluid slots. The optional P_e
   block is ghost-filled with it (copy/reflect of the adjacent interior
   layer, P_e is a scalar so never flipped) ONLY when the B1 electron
   closure gate is on; with the gate off the block is left untouched
   (placeholder-step contract: no charges loaded, no Pe ghost fill).
   Ghosts are refilled after the initial condition and at the start of
   every step. xmf_bc_type/xmf_fill_bc are helpers on the xmf_state_pack
   level, not a fifth public verb.

   A3 supplement: a user face may additionally carry compiled assignment
   tapes (xmf_bc_load_tapes). Every ghost cell of the face is then
   evaluated independently and ASSIGNED (not incremented): a slot with a
   tape gets the tape's value, an unwritten slot falls back to copy @1.
   Reads resolve the ghost's own cell centre (grid slots) and the two
   innermost active layers of the same face (KIND_NB slots, face-relative
   @1/@2). Boundary tapes never enter run->tapes and never touch the
   Newton solve -- the evaluation here is value-only. The floor path
   (fill_user_x1) is checked first and stays untouched. Auxiliary blocks
   (the optional P_e and the face-centred B of the M block) are
   ghost-filled on user faces by bc_fill_aux_blocks right after the
   fluid fill -- every such block must be registered there, or its
   user-face ghosts would silently stay stale while sweeps and Riemann
   reads consume them. */
#include "xmf_run.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- Per-face ghost fill (A3) -------------------------------------------
   Face indices are fixed: 0=x1beg, 1=x1end, 2=x2beg, 3=x2end, 4=x3beg,
   5=x3end -- dir = face/2 (0=x1, 1=x2, 2=x3), side = face%2 (0=beg,
   1=end). Each face carries one enum XmfBcType from XmfConfig.bc, copied
   into the run at setup; the kernel never reads boundary names.

   reflect mirrors the interior: the velocity slot normal to the face is
   odd (sign-flipped), every other slot is even. copy replicates the
   outermost interior layer (zero gradient, nothing flipped). user,
   outflow and none are padded exactly like copy this slice -- defined
   values rather than uninitialised ghosts; user is never a function
   pointer or a per-cell callback, the face simply keeps its recorded
   type for queries.

   Ghosts are written for the whole transverse plane, so 1D (where the
   transverse directions are degenerate) stays self-consistent. Every
   fluid's five slots are filled. The optional P_e block is ghost-filled
   with them (B1) only when the electron closure gate is on; with the
   gate off it stays untouched (A2 contract: a placeholder step never
   changes P_e). Only the face's own ghost cells are written; interior
   cells are never touched. */

/* Pad one face of a single ncells-long block. reflect: ghost layer m
   mirrors interior layer m (flip negates the normal velocity slot);
   copy: every ghost layer replicates the outermost interior layer. */
static void fill_face_block(const XmfRun *run, double *F, int face,
                            int reflect, int flip) {
  int nint[3], ntot[3];
  size_t stride[3];
  int d = face / 2, side = face % 2;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int ng = run->ng;
  int a, b, m;
  nint[0] = run->nx;
  nint[1] = run->ny;
  nint[2] = run->nz;
  ntot[0] = run->nxtot;
  ntot[1] = run->nytot;
  ntot[2] = run->nztot;
  stride[0] = 1;
  stride[1] = (size_t)run->nxtot;
  stride[2] = (size_t)run->nxtot * (size_t)run->nytot;
  for (a = 0; a < ntot[t1]; a++)
    for (b = 0; b < ntot[t2]; b++) {
      size_t base = (size_t)a * stride[t1] + (size_t)b * stride[t2];
      for (m = 0; m < ng; m++) {
        size_t g, src;
        if (side == 0) { /* beg face */
          g = base + (size_t)(ng - 1 - m) * stride[d];
          src = base + (size_t)(reflect ? ng + m : ng) * stride[d];
        } else { /* end face */
          g = base + (size_t)(ng + nint[d] + m) * stride[d];
          src = base +
                (size_t)(reflect ? ng + nint[d] - 1 - m : ng + nint[d] - 1) *
                    stride[d];
        }
        F[g] = flip ? -F[src] : F[src];
      }
    }
}

/* User floor fill (A5): every fill writes the pinned base state back to
   the face-side ghost layers AND the innermost active cell, for every
   fluid and slot, over the whole transverse plane. side 0 = x1beg,
   side 1 = x1end. The buffer layout is [fluid][slot][k*nytot+j]; no
   function pointers, no per-cell callbacks. */
static void fill_user_x1(const XmfRun *run, int side) {
  int ng = run->ng;
  int f, var, k, j, m;
  size_t nt = (size_t)run->nytot * (size_t)run->nztot;
  int i_inner = (side == 0) ? ng : ng + run->nx - 1;
  for (f = 0; f < run->nfluid; f++)
    for (var = 0; var < XMF_SLOT_NVAR; var++) {
      double *F = fld(run, f, var);
      const double *buf =
          run->user_bc +
          (((size_t)f * XMF_SLOT_NVAR + (size_t)var) * nt);
      for (k = 0; k < run->nztot; k++)
        for (j = 0; j < run->nytot; j++) {
          double v0 = buf[(size_t)k * (size_t)run->nytot + (size_t)j];
          for (m = 0; m < ng; m++) {
            int i = (side == 0) ? ng - 1 - m : ng + run->nx + m;
            F[cidx(run, k, j, i)] = v0;
          }
          F[cidx(run, k, j, i_inner)] = v0;
        }
    }
}

/* Periodic ghost fill (M5 supplement): ghost cells wrap around to the
   opposite side of the active domain. For a beg face, ghost layer m
   takes the value of interior layer (nint - ng + m); for an end face,
   ghost layer m takes interior layer (ng + m). No sign flips -- periodic
   is strict wrap, not a mirror. Both beg and end faces must be periodic
   (enforced by config validation); this function fills one face at a
   time so the dispatch stays uniform with the other BC types. */
static void fill_periodic_block(const XmfRun *run, double *F, int face) {
  int nint[3], ntot[3];
  size_t stride[3];
  int d = face / 2, side = face % 2;
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int ng = run->ng;
  int a, b, m;
  nint[0] = run->nx;
  nint[1] = run->ny;
  nint[2] = run->nz;
  ntot[0] = run->nxtot;
  ntot[1] = run->nytot;
  ntot[2] = run->nztot;
  stride[0] = 1;
  stride[1] = (size_t)run->nxtot;
  stride[2] = (size_t)run->nxtot * (size_t)run->nytot;
  for (a = 0; a < ntot[t1]; a++)
    for (b = 0; b < ntot[t2]; b++) {
      size_t base = (size_t)a * stride[t1] + (size_t)b * stride[t2];
      for (m = 0; m < ng; m++) {
        size_t g, src;
        if (side == 0) { /* beg face: ghost wraps from end interior */
          g = base + (size_t)(ng - 1 - m) * stride[d];
          src = base + (size_t)(ng + nint[d] - 1 - m) * stride[d];
        } else { /* end face: ghost wraps from beg interior */
          g = base + (size_t)(ng + nint[d] + m) * stride[d];
          src = base + (size_t)(ng + m) * stride[d];
        }
        F[g] = F[src];
      }
    }
}

/* ---- Analytic user boundary tapes (A3 supplement) -------------------------
   Assignment semantics, face-relative reads, value-only evaluation. The
   tapes live on the face (run->bc_tapes[face]), never in run->tapes --
   the source set feeding xmf_ir_apply / xmf_ir_jac / the implicit source
   solve stays exactly the loaded sources. Opcodes are the frozen table
   0..9 of xmf_ir.h; the only slot kinds a boundary tape may load are the
   ghost's own grid centre (GRID) and the two innermost active layers of
   this face (NB). */

/* Free one per-face tape set (NULL-safe). */
static void bc_free_tape_set(XmfTape *set, int ntape) {
  int t;
  if (!set)
    return;
  for (t = 0; t < ntape; t++) {
    free(set[t].ops);
    free(set[t].consts);
  }
  free(set);
}

void xmf_bc_clear_tapes(XmfRun *run) {
  int face;
  if (!run)
    return;
  for (face = 0; face < 6; face++) {
    bc_free_tape_set(run->bc_tapes[face], run->nbc_tape[face]);
    run->bc_tapes[face] = NULL;
    run->nbc_tape[face] = 0;
  }
}

/* Slot codes a boundary tape may load, checked at xmf_bc_load_tapes so
   the fill below can never fail on an accepted set: the ghost's own grid
   centre (GRID), or a neighbour read (NB) with fluid in range, layer
   1..2, var n..T, and enough active cells behind the face for the layer.
   Fluid self-reads, electron slots and beam fluxes are refused. */
static int bc_slot_loadable(const XmfRun *run, int face, int code) {
  int kind, a, b, d, nint;
  if (code < 0)
    return 0;
  kind = (code >> 16) & 0xFF;
  a = (code >> 8) & 0xFF;
  b = code & 0xFF;
  switch (kind) {
  case XMF_IR_GRID:
    return a <= XMF_IR_X3 && b == 0;
  case XMF_IR_NB: {
    int layer = (b >> 4) & 0x0F;
    int var = b & 0x0F;
    if (a >= run->nfluid)
      return 0;
    if (layer < 1 || layer > 2)
      return 0;
    if (var > XMF_IR_T)
      return 0;
    d = face / 2;
    nint = (d == 0) ? run->nx : (d == 1) ? run->ny : run->nz;
    return nint >= layer; /* @2 needs two active cells behind the face */
  }
  default:
    return 0;
  }
}

/* Value of one NB slot on the transverse row (k,j,i) of the ghost being
   filled: the neighbour cell is layer @1/@2 counted inward from the face
   along its normal direction (face-relative; works for beg and end and
   for any axis). Density reads return the number density rho/mass_code;
   T is derived exactly like the source IR does (P*unit_P/(n*kB), n
   floored at 1e-30); every other slot is the stored code value. */
static int bc_nb_value(const XmfRun *run, int face, int code, int k, int j,
                       int i, double *out) {
  int a = (code >> 8) & 0xFF;
  int b = code & 0xFF;
  int layer = (b >> 4) & 0x0F;
  int var = b & 0x0F;
  int d = face / 2, side = face % 2;
  int nint = (d == 0) ? run->nx : (d == 1) ? run->ny : run->nz;
  int nb; /* normal index of the neighbour cell */
  int kk, jj, ii;
  size_t c;
  if (a >= run->nfluid || layer < 1 || layer > 2 || var > XMF_IR_T)
    return -1;
  if (side == 0)
    nb = run->ng + (layer - 1); /* @1 = ng, @2 = ng+1 */
  else
    nb = run->ng + nint - layer; /* @1 = ng+nint-1, @2 = ng+nint-2 */
  kk = (d == 2) ? nb : k;
  jj = (d == 1) ? nb : j;
  ii = (d == 0) ? nb : i;
  c = cidx(run, kk, jj, ii);
  if (var == XMF_IR_T) {
    double rho = fld(run, a, XMF_SLOT_RHO)[c];
    double pc = fld(run, a, XMF_SLOT_P)[c];
    double n = rho / run->mass_code[a];
    double n_safe = n > 1e-30 ? n : 1e-30;
    *out = pc * run->unit_P / (n_safe * XMF_KB);
    return 0;
  }
  if (var == XMF_IR_N) {
    *out = fld(run, a, XMF_SLOT_RHO)[c] / run->mass_code[a];
    return 0;
  }
  *out = fld(run, a, var)[c];
  return 0;
}

/* Value-only stack machine for one boundary tape at ghost cell (k,j,i)
   of face. Same frozen opcode table as xmf_ir.c, but no dual numbers --
   boundary tapes assign ghosts and never enter the Newton solve. st is
   caller scratch of at least nops+1 doubles. Returns 0, or -1 on a
   malformed tape (cannot happen for sets accepted by
   xmf_bc_load_tapes). */
static int bc_eval_tape(const XmfRun *run, int face, const XmfTape *tp,
                        int k, int j, int i, double *st, double *out) {
  int sp = 0, p;
  for (p = 0; p < tp->nops; p++) {
    int op = tp->ops[2 * p];
    int arg = tp->ops[2 * p + 1];
    switch (op) {
    case XMF_OP_LOAD_CONST:
      if (sp >= tp->nops + 1 || arg < 0 || arg >= tp->nconst)
        return -1;
      st[sp++] = tp->consts[arg];
      break;
    case XMF_OP_LOAD_SLOT: {
      double v;
      int kind = (arg >> 16) & 0xFF;
      int ax = (arg >> 8) & 0xFF;
      if (sp >= tp->nops + 1)
        return -1;
      if (kind == XMF_IR_GRID && ax <= XMF_IR_X3 && (arg & 0xFF) == 0) {
        v = (ax == XMF_IR_X1)   ? run->mesh[0].x[i]
            : (ax == XMF_IR_X2) ? run->mesh[1].x[j]
                                : run->mesh[2].x[k];
      } else if (kind == XMF_IR_NB) {
        if (bc_nb_value(run, face, arg, k, j, i, &v))
          return -1;
      } else
        return -1;
      st[sp++] = v;
      break;
    }
    case XMF_OP_NEG:
      if (sp < 1)
        return -1;
      st[sp - 1] = -st[sp - 1];
      break;
    case XMF_OP_EXP:
      if (sp < 1)
        return -1;
      st[sp - 1] = exp(st[sp - 1]);
      break;
    case XMF_OP_LOG:
      if (sp < 1)
        return -1;
      st[sp - 1] = log(st[sp - 1]);
      break;
    default: {
      double bv, av, v;
      if (sp < 2)
        return -1;
      bv = st[--sp];
      av = st[--sp];
      if (op == XMF_OP_ADD)
        v = av + bv;
      else if (op == XMF_OP_SUB)
        v = av - bv;
      else if (op == XMF_OP_MUL)
        v = av * bv;
      else if (op == XMF_OP_DIV)
        v = av / bv;
      else if (op == XMF_OP_POW)
        v = pow(av, bv); /* value-only path, exactly the xmf_ir_apply
                            convention; boundary tapes are never seeded */
      else
        return -1;
      st[sp++] = v;
    }
    }
  }
  if (sp != 1)
    return -1;
  *out = st[0];
  return 0;
}

/* Tape fill of one user face (A3 supplement): every ghost cell of the
   face's whole slab (all ghost layers, the whole transverse plane --
   loop k,j,i over the full block and keep the face's own ghost rows) is
   evaluated independently and ASSIGNED. A slot with a tape gets the
   tape's value; an unwritten slot falls back to copy @1 (the face's
   outermost active cell on the same transverse row). The density slot
   stores rho = mass_code*n, mirroring the source-IR write convention;
   every other slot is assigned unchanged. The P_e block is never a tape
   target: with the electron closure gate on, bc_fill_aux_blocks pads its
   ghost layers right after this fill (zero-order copy of the outermost
   active layer); with the gate off it stays untouched. On any
   (defensive) failure the ghosts keep their previous defined values. */
static void fill_user_tape_face(XmfRun *run, int face) {
  int d = face / 2, side = face % 2;
  int ng = run->ng;
  int ntape = run->nbc_tape[face];
  const XmfTape *tapes = run->bc_tapes[face];
  int nint = (d == 0) ? run->nx : (d == 1) ? run->ny : run->nz;
  int nslot = 5 * run->nfluid;
  int *which; /* slot f*5+var -> tape index, -1 = unwritten (copy @1) */
  double *vals, *st;
  int maxops = 0;
  int f, var, t, k, j, i;
  if (ntape <= 0)
    return;
  for (t = 0; t < ntape; t++)
    if (tapes[t].nops > maxops)
      maxops = tapes[t].nops;
  which = (int *)malloc(sizeof(int) * (size_t)nslot);
  vals = (double *)malloc(sizeof(double) * (size_t)ntape);
  st = (double *)malloc(sizeof(double) * (size_t)(maxops + 1));
  if (!which || !vals || !st) {
    free(which);
    free(vals);
    free(st);
    return;
  }
  for (t = 0; t < nslot; t++)
    which[t] = -1;
  for (t = 0; t < ntape; t++)
    /* For boundary tapes XmfTape carries dest_kind = the assigned
       field slot var (0..4, same numbers as enum XmfIrWrite) and
       dest_fluid = the fluid index. */
    which[tapes[t].dest_fluid * 5 + tapes[t].dest_kind] = t;
  for (k = 0; k < run->nztot; k++)
    for (j = 0; j < run->nytot; j++)
      for (i = 0; i < run->nxtot; i++) {
        int gd = (d == 0) ? i : (d == 1) ? j : k;
        size_t cg;
        if (side == 0 ? (gd >= ng) : (gd < ng + nint))
          continue; /* outside this face's ghost slab */
        for (t = 0; t < ntape; t++)
          if (bc_eval_tape(run, face, &tapes[t], k, j, i, st, &vals[t])) {
            free(which);
            free(vals);
            free(st);
            return; /* validated sets cannot fail; keep old ghosts */
          }
        cg = cidx(run, k, j, i);
        for (f = 0; f < run->nfluid; f++)
          for (var = 0; var < XMF_SLOT_NVAR; var++) {
            double *F = fld(run, f, var);
            int tt = which[f * 5 + var];
            if (tt >= 0) {
              F[cg] = (var == XMF_SLOT_RHO) ? run->mass_code[f] * vals[tt]
                                            : vals[tt];
            } else {
              int ks = k, js = j, is = i;
              if (d == 0)
                is = (side == 0) ? ng : ng + nint - 1;
              else if (d == 1)
                js = (side == 0) ? ng : ng + nint - 1;
              else
                ks = (side == 0) ? ng : ng + nint - 1;
              F[cg] = F[cidx(run, ks, js, is)]; /* copy @1 fallback */
            }
          }
      }
  free(which);
  free(vals);
  free(st);
}

/* ---- Auxiliary-block ghost port (B1 supplement) --------------------------
   The fluid fills of bc_fill_face (floor buffer, assignment tapes, and
   the generic copy/reflect pad) cover only the five fluid slots. Every
   auxiliary block that lives on the run -- the optional P_e block and
   the face-centred B of the M block -- MUST be registered in this one
   helper: the floor buffer layout has no slot for it and the tape slot
   check refuses non-fluid slots, so a block missing here would silently
   keep stale ghost cells on user faces while the sweeps and Riemann
   reads consume them.

   Padding semantics per face kind:
   - non-user, non-periodic faces: the same pad as the fluid fill via
     fill_face_block -- reflect mirrors (reflect passed through from the
     face's bc type; a scalar pressure never flips), copy / outflow /
     none replicate the outermost active layer;
   - user faces (never reflect): zero-order copy -- every ghost layer of
     the whole transverse plane takes the outermost active layer's value
     on the same row. P_e has no prescribed boundary value: the floor
     buffer carries no P_e slot and the tape loader refuses it, so it
     stays an evolved quantity and the innermost active cell is never
     overwritten. If a prescribed boundary value is ever wanted, change
     it HERE -- never leave a silent 0/stale ghost.
   - periodic faces (M5 supplement): ghost wraps from the opposite
     active domain via fill_periodic_block (P_e is a scalar, no flip).

   With the electron closure gate off (electrons off, or no charges
   loaded) the P_e entry below writes nothing, bit for bit (A2 contract:
   a step or tape fill never touches P_e). */
static void bc_fill_aux_blocks(XmfRun *run, int face, int reflect) {
  int periodic = (run->bc[face] == XMF_BC_PERIODIC);
  /* Auxiliary-block registry -- every new block goes here: */
  if (elec_closure_active(run)) { /* P_e (B1): scalar, never flipped */
    if (periodic)
      fill_periodic_block(run, run->pe, face);
    else
      fill_face_block(run, run->pe, face, reflect, 0);
  }
  /* Face-centred B (M3): the whole staggered fill lives in xmf_mag.c
     (reflect wall/symmetric tiers, user-face tangential floor +
     divB = 0 normal reconstruction, periodic wrap, zero-order copy
     otherwise); the dispatch is a no-op while the magnetic gate is
     closed, so the pre-M path stays bit for bit. */
  mag_fill_face(run, face, reflect);
}

static void bc_fill_face(XmfRun *run, int face) {
  static const int normvar[3] = {XMF_SLOT_VX, XMF_SLOT_VY, XMF_SLOT_VZ};
  int d = face / 2;
  int reflect = (run->bc[face] == XMF_BC_REFLECT);
  int periodic = (run->bc[face] == XMF_BC_PERIODIC);
  int f, var;
  /* x1 user face with a loaded floor buffer (A5): pinned ghosts +
     innermost active cell. Without a buffer a user face keeps the
     pre-A5 copy padding, so old tests stay green. */
  if (run->bc[face] == XMF_BC_USER && run->user_bc && d == 0) {
    fill_user_x1(run, face % 2);
    bc_fill_aux_blocks(run, face, 0); /* user faces are never reflect */
    return;
  }
  /* User face with loaded assignment tapes (A3 supplement): assign every
     ghost slot (unwritten slots copy @1). Configs keep the floor path
     and the tapes mutually exclusive on x1; should both ever be loaded
     the floor above wins, unchanged. */
  if (run->bc[face] == XMF_BC_USER && run->nbc_tape[face] > 0) {
    fill_user_tape_face(run, face);
    bc_fill_aux_blocks(run, face, 0); /* user faces are never reflect */
    return;
  }
  /* Periodic face (M5 supplement): ghost cells wrap from the opposite
     active domain. No sign flips -- periodic is strict wrap, not mirror.
     Both beg and end faces must be periodic (config validation enforces
     this); the auxiliary blocks (P_e, B) go through their own periodic
     fill in bc_fill_aux_blocks with reflect=0. */
  if (periodic) {
    for (f = 0; f < run->nfluid; f++)
      for (var = 0; var < XMF_SLOT_NVAR; var++)
        fill_periodic_block(run, fld(run, f, var), face);
    bc_fill_aux_blocks(run, face, 0); /* periodic faces are never reflect */
    return;
  }
  for (f = 0; f < run->nfluid; f++)
    for (var = 0; var < XMF_SLOT_NVAR; var++)
      fill_face_block(run, fld(run, f, var), face, reflect,
                      reflect && (var == normvar[d]));
  /* B1: when the electron closure gate is on the P_e block is swept
     every direction, so its ghosts need the same padding as any fluid
     slot. All auxiliary blocks go through the single registration point
     bc_fill_aux_blocks -- here with the face's own reflect flag, on the
     two user early returns above with zero-order copy. With the gate off
     (electrons off, or no charges loaded) it writes nothing, exactly as
     pre-B1, so the A2 contract (test_state / test_bc_user: a step or
     tape fill never touches P_e) stays bit-for-bit green. */
  bc_fill_aux_blocks(run, face, reflect);
}

void fill_bc(XmfRun *run) {
  int face;
  for (face = 0; face < 6; face++)
    bc_fill_face(run, face);
}

/* ---- Per-face boundary helpers (A3, helper level like xmf_state_pack) ---
   Face indices are fixed: 0=x1beg, 1=x1end, 2=x2beg, 3=x2end, 4=x3beg,
   5=x3end. xmf_bc_type reports the enum XmfBcType stored for one face
   (reflect=0, copy=1, outflow=2, user=3, none=4), or -1 on invalid
   arguments. xmf_fill_bc immediately re-pads every face's ghost cells
   according to the run's current six face types -- the same fill that
   setup performs after the initial condition and every step performs at
   its start. It is not a fifth public verb. Returns 0, or -1 on invalid
   arguments. */

int xmf_bc_type(const XmfRun *run, int face) {
  if (!run || face < 0 || face > 5)
    return -1;
  return run->bc[face];
}

int xmf_fill_bc(XmfRun *run) {
  if (!run)
    return -1;
  /* Port B: fill the ghost skin on the device (no host roundtrip). Falls
     back to the host path when no device state is attached (before setup
     completes, after finalize, or on allocation failure). */
  if (run->dev) {
    xmf_bc_fill_device(run);
    /* Port C: re-pin frozen cells after ghost fill (device path). B is
       NOT pinned -- the CT path's B is not touched here. */
    xmf_pin_apply_device(run);
    return 0;
  }
  /* Host fallback: the fill logic (floor / copy / reflect / boundary
     tapes) runs on the host mirror, bracketed by one device<->mirror
     roundtrip. The internal fill_bc stays mirror-only; lf_all_dirs
     brackets its own per-direction calls the same way. */
  xmf_dev_pull(run);
  fill_bc(run);
  xmf_dev_push(run);
  /* Port C: re-pin frozen cells after ghost fill (host path). B is NOT
     pinned -- the CT path's B is not touched here. */
  xmf_pin_apply_host(run);
  return 0;
}

/* ---- Port C: frozen-cell pin mask ----------------------------------------
   The pin loader copies a whole-block mask (ncells ints, 0/1, ghosts
   included) and matching pin values ([nfluid][XMF_SLOT_NVAR][ncells]
   doubles, same flat layout as run->fields) into the run. The step path
   applies them automatically: after every active sweep in lf_all_dirs
   and after every ghost fill in xmf_fill_bc. Only masked cells are
   overwritten; P_e is never touched; B is never touched (the CT path's
   B is not pinned by this port -- extending pin to B is a future slice
   and needs a hook after mag_induce, not here). Calling the same entry
   point with a new mask/values replaces the previous pin. NULL mask or
   all-zero mask clears the pin. Returns 0, or -1 on invalid arguments /
   allocation failure (the previous pin stays loaded). */

/* Host pin apply: iterate every cell, write pin_vals where pin_mask is
   set. No-op when no pin is loaded. P_e is never touched; B is never
   touched. Layout is [fluid][slot][cell] matching run->fields /
   fld(run,f,var). */
void xmf_pin_apply_host(XmfRun *run) {
  size_t c;
  int f, var;
  if (!run || !run->pin_mask || !run->pin_vals)
    return;
  for (c = 0; c < run->ncells; c++) {
    if (!run->pin_mask[c])
      continue;
    for (f = 0; f < run->nfluid; f++)
      for (var = 0; var < XMF_SLOT_NVAR; var++) {
        size_t idx =
            ((size_t)f * XMF_SLOT_NVAR + (size_t)var) * run->ncells + c;
        run->fields[idx] = run->pin_vals[idx];
      }
  }
}

int xmf_bc_set_pin(XmfRun *run, const int *mask, const double *vals) {
  int *new_mask = NULL;
  double *new_vals = NULL;
  size_t mask_bytes, val_bytes;
  size_t c;
  int any;
  if (!run)
    return -1;
  /* NULL mask or NULL vals: clear the pin. */
  if (!mask || !vals) {
    free(run->pin_mask);
    free(run->pin_vals);
    run->pin_mask = NULL;
    run->pin_vals = NULL;
    xmf_dev_pin_sync(run);
    return 0;
  }
  /* Check if the mask is all-zero: treat as clear. */
  any = 0;
  for (c = 0; c < run->ncells; c++) {
    if (mask[c]) {
      any = 1;
      break;
    }
  }
  if (!any) {
    free(run->pin_mask);
    free(run->pin_vals);
    run->pin_mask = NULL;
    run->pin_vals = NULL;
    xmf_dev_pin_sync(run);
    return 0;
  }
  mask_bytes = run->ncells * sizeof(int);
  val_bytes = (size_t)run->nfluid * XMF_SLOT_NVAR * run->ncells *
              sizeof(double);
  new_mask = (int *)malloc(mask_bytes);
  new_vals = (double *)malloc(val_bytes);
  if (!new_mask || !new_vals) {
    free(new_mask);
    free(new_vals);
    return -1;
  }
  memcpy(new_mask, mask, mask_bytes);
  memcpy(new_vals, vals, val_bytes);
  /* Swap in: the previous pin (if any) is dropped. */
  free(run->pin_mask);
  free(run->pin_vals);
  run->pin_mask = new_mask;
  run->pin_vals = new_vals;
  /* Port C: push the new pin to the device Views. */
  if (xmf_dev_pin_sync(run))
    return -1;
  return 0;
}

/* User boundary floor loader (A5), auxiliary load port -- see
   xmf_core.h. data holds [fluid][slot][k*nytot+j] code-unit slots; the
   buffer is copied into the run and owned by it from here on (freed by
   xmf_finalize). Returns 0, or -1 on invalid arguments. */
int xmf_bc_set_user(XmfRun *run, const double *data) {
  size_t nt, bytes;
  double *buf;
  if (!run || !data)
    return -1;
  nt = (size_t)run->nytot * (size_t)run->nztot;
  bytes = (size_t)run->nfluid * XMF_SLOT_NVAR * nt * sizeof(double);
  buf = (double *)malloc(bytes);
  if (!buf)
    return -1;
  memcpy(buf, data, bytes);
  free(run->user_bc); /* replace any previously loaded floor */
  run->user_bc = buf;
  /* Port B: push the new floor to the device Views. The BC fill kernel
     reads from the device copy; the host pointer is never dereferenced
     by device kernels. */
  xmf_dev_bc_sync(run);
  return 0;
}

/* Analytic user boundary tape loader (A3 supplement), auxiliary load
   port -- see xmf_core.h. header holds four ints per tape: dest_var
   (the assigned field slot 0..4 = n,vx,vy,vz,P), dest_fluid (fluid
   index), nops (>= 1), nconst; ops/consts are concatenated exactly like
   xmf_ir_load. The set is copied onto the face (owned by the run from
   here on, freed by xmf_finalize) and replaces any previous set of that
   face. The face must already carry XMF_BC_USER. Returns ntape on
   success; -1 on invalid arguments / allocation failure (the previous
   set stays untouched then); ntape == 0 clears the face's set. */
int xmf_bc_load_tapes(XmfRun *run, int face, int ntape, const int *header,
                      const int *ops, const double *consts) {
  XmfTape *set;
  int t, p;
  size_t off_ops = 0, off_cst = 0;
  if (!run || face < 0 || face > 5 || ntape < 0)
    return -1;
  if (ntape == 0) { /* empty set: clear this face, other faces untouched */
    bc_free_tape_set(run->bc_tapes[face], run->nbc_tape[face]);
    run->bc_tapes[face] = NULL;
    run->nbc_tape[face] = 0;
    return 0;
  }
  if (!header || !ops)
    return -1;
  if (run->bc[face] != XMF_BC_USER)
    return -1; /* tapes only ever sit on user faces */
  set = (XmfTape *)calloc((size_t)ntape, sizeof(XmfTape));
  if (!set)
    return -1;
  for (t = 0; t < ntape; t++) {
    int var = header[4 * t + 0];
    int fl = header[4 * t + 1];
    int nops = header[4 * t + 2];
    int ncst = header[4 * t + 3];
    int t2;
    XmfTape *tp = &set[t];
    tp->dest_kind = var;
    tp->dest_fluid = fl;
    tp->nops = nops;
    tp->nconst = ncst;
    if (var < XMF_SLOT_RHO || var > XMF_SLOT_P)
      goto fail; /* assignment targets a stored fluid slot, never T/P_e */
    if (fl < 0 || fl >= run->nfluid)
      goto fail;
    for (t2 = 0; t2 < t; t2++)
      if (set[t2].dest_kind == var && set[t2].dest_fluid == fl)
        goto fail; /* one tape per slot per face */
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
        if (!bc_slot_loadable(run, face, arg))
          goto fail;
      }
      tp->ops[2 * p] = op;
      tp->ops[2 * p + 1] = arg;
    }
    off_ops += 2 * (size_t)nops;
    off_cst += (size_t)ncst;
  }
  /* New set complete: swap it in, drop the face's old one. */
  bc_free_tape_set(run->bc_tapes[face], run->nbc_tape[face]);
  run->bc_tapes[face] = set;
  run->nbc_tape[face] = ntape;
  /* Port B: push the updated tape set to the device Views. */
  xmf_dev_bc_sync(run);
  return ntape;

fail:
  bc_free_tape_set(set, ntape); /* NULL pointers from calloc are free-safe */
  return -1;
}
