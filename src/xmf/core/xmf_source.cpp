/* Semi-implicit source step (A5) -- own module, never merged back into
   xmf_core.c.

   xmf_step calls this after every hydro stage. With no tape loaded it is
   a no-op and the pure-hydro path stays bit-for-bit untouched. With
   tapes loaded it advances the in-place fields through ONE linearised
   backward-Euler source increment:

       U* = U^n + dt*L_f                     (already done by the sweeps)
       solve  U^{n+1} - U* = dt * R(U^{n+1})  with
              (I - dt*J) dU = dt*R(U*),   U^{n+1} = U* + dU,
       J = dR/dU at U*, one local dense N x N LU per cell, solved
       exactly (partial pivoting). No iteration -- the single linearised
       step IS the contract; its one-step residual is reported instead
       (run->source_resid, relative max over the step).

   N = xmf_ir_nstate(run) (B2): the five conservative unknowns per fluid
   plus -- exactly when the P_e block exists -- ONE trailing electron
   unknown. P_e is pressure-only: its rate is the gathered dP_e tape rate
   as-is (no lift), its S1/S3 diagonal entries are 1 (R_pe = Lc_pe,
   P_e IS the unknown), and the solved increment is floored like the
   fluid pressures before the write-back. With electrons off N stays
   5*nfluid and every loop below is bit-for-bit the pre-B2 one.

   R = T(w, L_c) lifts the tape rates L_c = (ndot, vdotx, vdoty, vdotz,
   Pdot) per fluid into conservative rates (the same ADDS convention as
   xmf_ir_apply; a bare dvx/dvy/dvz tape enters every fluid):
       rho_dot = m*ndot
       (rho v)_dot = m*ndot*v + rho*vdot
       E_dot = Pdot/(gamma-1) + 0.5*|v|^2*rho_dot + rho*v.vdot

   The Jacobian is the four-segment chain of the card,
       dR/dU = ( S1*S2 + S4 ) * S3
   S1 = dT/dL_c (block diagonal, per fluid), S2 = dL_c/dw from forward
   AD of the very loaded tapes (xmf_ir_eval with seeded slots), S3 =
   dw/dU (block diagonal), and S4 = dT/dw at FROZEN L_c (block diagonal;
   forbidden to omit -- the w-dependence of the lift itself). w uses the
   IR primitives (n = rho/mass_code, vx, vy, vz, P), so the S2 density
   column arrives as d/d(rho_stored) and is multiplied by mass_code to
   become d/dn. Beam flux F_b and grid coordinates r are never seeded:
   they stay frozen inside the Newton step (tape masks skip every column
   they would otherwise touch).

   Floors: the solved conservative state is floored at 1e-30 (rho and P)
   exactly like the hydro's cons5_to_prim before it is written back.

   C1a loop shape: the interior cells ride one Kokkos parallel loop -- the
   implicit source step is strictly cell-local (a tape reads only its own
   cell's slots and the writes land only in that cell), so every cell is
   an independent iteration with its own workspace row. The per-cell body
   (source_cell below) keeps the serial formulas verbatim; nothing is
   stored in XmfRun.

   C1b: the loop is a parallel_reduce, not a parallel_for -- the error
   flag and the report-only residual are now genuine Kokkos reductions
   (Sum<int> / Max<double>), never bare writes into shared heap scalars
   (a data race once the default execution space is OpenMP). The residual
   fold is an exact max; the error fold is an exact integer sum counting
   the reduction shares that saw a failing cell. Neither depends on
   iteration or reduction order, so the numbers stay bit-for-bit the C1a
   Serial ones. (The error flag deliberately does NOT use Max<int>: that
   reducer's final() overwrites the scalar with its neutral element
   INT_MIN when no cell failed, which reads as an error. Sum<int>'s
   neutral is 0, so the flag stays 0 on a clean sweep.)

   C2: the whole solve runs on the device Views -- the fields / P_e /
   mass table / tape set / beam flux through the SrcCtx bundle, the tape
   evaluator being the shared xmf::ir_eval of xmf_ir_eval.hpp (opcodes
   0..9 unchanged; no new chemistry entry). The per-cell workspace rows
   live in the default memory space (device memory on the CUDA tree,
   HostSpace on the host trees); the tape masks are built host-side from
   the host tapes and pushed once per step. The parallel_reduce shape,
   the two reducers and the cell-local formulas are the C1b ones
   verbatim; the memsets of the C1 workspace became explicit zero loops
   (memset is host-only). No mirror roundtrip happens inside the solve:
   the sweeps left the device fields at U* and the next stage pulls the
   mirror itself. */

#include "xmf_run.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "xmf_device.h"
#include "xmf_kokkos.h"

#define XMF_SRC_FLOOR 1e-30
#define XMF_SRC_PIVOT_FLOOR 1e-300

/* ---- C2 device bundle ---------------------------------------------------
   Everything the per-cell solve dereferences, as device pointers plus
   scalars. Built host-side from the run's DeviceState (the flattened
   tape set arrives through xmf_dev_tapes_sync at xmf_ir_load); copied
   into the kernel by value. */
struct SrcCtx {
  double *fields;        /* device mirror of run->fields */
  double *pe;            /* device P_e; NULL when electrons off */
  const double *mass_code;   /* [nfluid] device table */
  xmf::IrCtx ir;         /* device tape-evaluator context */
  const int *desc;       /* flattened tape descriptors, 6 ints per tape:
                            dest_kind, dest_fluid, nops, nconst,
                            ops offset (ints), consts offset (doubles) */
  const int *ops;        /* flattened (opcode, arg) pairs */
  const double *consts;  /* flattened constants */
  const unsigned *mask;  /* per-tape state-slot dependency masks */
  int ntape, nwords;
  int n;                 /* Newton unknown count (xmf_ir_nstate) */
  int nf5;               /* 5 * nfluid */
  int nfluid;
  int nxtot, nytot;
  size_t ncells;
  double gmm1, dt;
};

KOKKOS_INLINE_FUNCTION
static size_t scidx(const SrcCtx &s, int k, int j, int i) {
  return (((size_t)k * (size_t)s.nytot) + (size_t)j) * (size_t)s.nxtot +
         (size_t)i;
}

/* Device field-slot pointer of fluid f, slot var. */
KOKKOS_INLINE_FUNCTION
static double *src_slot(const SrcCtx &s, int f, int var) {
  return s.fields +
         (((size_t)f * XMF_SLOT_NVAR + (size_t)var) * s.ncells);
}

/* ---- Small dense linear algebra ---------------------------------------- */

KOKKOS_INLINE_FUNCTION
static void mat_mul(const double *A, const double *B, double *C, int n) {
  int i, j, k;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      double s = 0.0;
      for (k = 0; k < n; k++)
        s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
      C[(size_t)i * n + j] = s;
    }
}

/* In-place LU solve A*x = b with partial pivoting (b becomes x).
   Returns 0, or -1 when a pivot underflows the floor. */
KOKKOS_INLINE_FUNCTION
static int lu_solve(double *A, double *b, int n) {
  int k, i, j;
  for (k = 0; k < n; k++) {
    int p = k;
    double mx = fabs(A[(size_t)k * n + k]);
    double piv, t;
    for (i = k + 1; i < n; i++) {
      double v = fabs(A[(size_t)i * n + k]);
      if (v > mx) {
        mx = v;
        p = i;
      }
    }
    if (mx < XMF_SRC_PIVOT_FLOOR)
      return -1;
    if (p != k) {
      for (j = 0; j < n; j++) {
        t = A[(size_t)k * n + j];
        A[(size_t)k * n + j] = A[(size_t)p * n + j];
        A[(size_t)p * n + j] = t;
      }
      t = b[k];
      b[k] = b[p];
      b[p] = t;
    }
    piv = A[(size_t)k * n + k];
    for (i = k + 1; i < n; i++) {
      double f = A[(size_t)i * n + k] / piv;
      A[(size_t)i * n + k] = f;
      b[i] -= f * b[k];
      for (j = k + 1; j < n; j++)
        A[(size_t)i * n + j] -= f * A[(size_t)k * n + j];
    }
  }
  for (i = n - 1; i >= 0; i--) {
    double s = b[i];
    for (j = i + 1; j < n; j++)
      s -= A[(size_t)i * n + j] * b[j];
    b[i] = s / A[(size_t)i * n + i];
  }
  return 0;
}

/* ---- Tape state-slot dependency masks ------------------------------------
   Bit f*5+s of tape t's mask is set when any LOAD_SLOT of the tape reads
   the stored slot s of fluid f (directly, or through the derived T of the
   same fluid, which depends on stored rho and P). B2: when the P_e block
   exists the Newton unknown space gains one trailing bit (5*nfluid) for
   the electron pressure -- ELEC P_e reads set it directly, and the
   derived reads n_e / T_e depend on it too (T_e = P_e/(n_e kB)) besides
   the charged fluids' rho slots they derive from. Grid coordinates, F_b
   and constants set no bit: they are never independent variables of the
   Newton solve, so a tape touching only those (e.g. the body-force tape
   loading r) contributes nothing to any Jacobian column and is skipped
   in every seeded evaluation.

   C2: the mask BUILD stays host-side (it reads the host tape set), and
   the finished mask block is pushed to the device once per step. */

#define XMF_MASK_SET(m, bit) ((m)[(bit) >> 5] |= 1u << ((bit) & 31))
#define XMF_MASK_GET(m, bit) (((m)[(bit) >> 5] >> ((bit) & 31)) & 1u)

static void tape_mask(const XmfRun *run, const XmfTape *tp, unsigned *mask) {
  int p;
  int n = xmf_ir_nstate(run);
  size_t words = ((size_t)n + 31) / 32;
  memset(mask, 0, words * sizeof(unsigned));
  for (p = 0; p < tp->nops; p++) {
    int op = tp->ops[2 * p];
    int arg = tp->ops[2 * p + 1];
    int kind, a, b;
    if (op != XMF_OP_LOAD_SLOT)
      continue;
    kind = (arg >> 16) & 0xFF;
    a = (arg >> 8) & 0xFF;
    b = arg & 0xFF;
    if (kind == XMF_IR_ELEC && run->pe) {
      /* B2 Newton unknown N-1 = P_e. P_e reads set it directly; the
         quasi-neutral derived reads hang off it too: n_e = sum_ion q_s
         rho_s/mass_code (the charged fluids' rho slots, ions recognised
         by q_s != 0, names never consulted) and T_e = P_e*unit_P/(n_e kB)
         (P_e AND those rho slots). */
      int pe_bit = 5 * run->nfluid;
      int s;
      if (a == XMF_IR_PE || a == XMF_IR_TE)
        XMF_MASK_SET(mask, pe_bit);
      if (a == XMF_IR_NE || a == XMF_IR_TE)
        for (s = 0; s < run->nfluid; s++)
          if (run->charge[s] != 0.0)
            XMF_MASK_SET(mask, s * 5 + 0);
      continue;
    }
    if (kind != XMF_IR_FLUID)
      continue; /* grid / flux (and elec with the block absent): frozen
                   inside the Newton step */
    if (b == XMF_IR_T) {
      XMF_MASK_SET(mask, a * 5 + 0); /* T = P*unit_P/(n*kB): depends on */
      XMF_MASK_SET(mask, a * 5 + 4); /* stored rho and P of the fluid  */
    } else if (b <= XMF_IR_P) {
      XMF_MASK_SET(mask, a * 5 + b);
    }
  }
}

/* ---- Tape-rate gathering (same ADDS convention as xmf_ir_apply) ---------
   L_c has N = xmf_ir_nstate(run) entries: the five tape rates per fluid
   plus, B2, the trailing dP_e rate when the P_e block exists. C2 device
   twin: the tape set is the flattened device mirror (desc/ops/consts),
   evaluated by the shared xmf::ir_eval. */

KOKKOS_INLINE_FUNCTION
static int eval_lc(const SrcCtx &s, int k, int j, int i, double *Lc) {
  int t, f, q;
  for (q = 0; q < s.n; q++)
    Lc[q] = 0.0;
  for (t = 0; t < s.ntape; t++) {
    const int *desc = s.desc + 6 * t;
    int dest_kind = desc[0];
    int dest_fluid = desc[1];
    int nops = desc[2];
    int nconst = desc[3];
    const int *ops = s.ops + (size_t)desc[4];
    const double *consts = s.consts + (size_t)desc[5];
    int slot;
    double val;
    if (xmf::ir_eval(s.ir, ops, consts, nops, nconst, k, j, i, -1, &val,
                     NULL))
      return -1;
    if (dest_kind == XMF_IR_DPE) { /* electron rate: last entry */
      if (!s.pe)
        return -1; /* load-time contract, repeated defensively */
      Lc[s.n - 1] += val;
      continue;
    }
    slot = xmf::ir_write_slot(dest_kind);
    if (slot < 0)
      return -1;
    if (dest_fluid < 0) { /* bare dvx/dvy/dvz: every fluid */
      for (f = 0; f < s.nfluid; f++)
        Lc[f * 5 + slot] += val;
    } else {
      Lc[dest_fluid * 5 + slot] += val;
    }
  }
  return 0;
}

/* ---- Per-fluid segments of the chain -------------------------------------
   w = (n, vx, vy, vz, P) and lc = (ndot, vdotx, vdoty, vdotz, Pdot) of
   ONE fluid. S1 (dT/dLc), S4 (dT/dw at frozen Lc) and the lifted rate R5
   only need w and lc; S3 (dw/dU) only needs w. All 5x5 blocks are
   row-major; the caller embeds them block-diagonally into the N x N
   matrices. rho is the stored mass density m*n of the fluid.
   C2: the former memsets are explicit 25-element zero loops. */

KOKKOS_INLINE_FUNCTION
static void seg_s1_s4_r(double m, double rho, const double *w,
                        const double *lc, double gmm1, double *S1,
                        double *S4, double *R5) {
  double vx = w[1], vy = w[2], vz = w[3];
  double n = w[0];
  double ndot = lc[0], dx_ = lc[1], dy_ = lc[2], dz_ = lc[3], pdot = lc[4];
  double v2 = vx * vx + vy * vy + vz * vz;
  double vdv = vx * dx_ + vy * dy_ + vz * dz_;
  int q;
  for (q = 0; q < 25; q++) {
    S1[q] = 0.0;
    S4[q] = 0.0;
  }
  /* S1 rows: (rho_dot, (rho vx)_dot, (rho vy)_dot, (rho vz)_dot, E_dot);
     columns: (ndot, vdotx, vdoty, vdotz, Pdot). */
  S1[0] = m;
  S1[5] = m * vx;
  S1[6] = rho;
  S1[10] = m * vy;
  S1[12] = rho;
  S1[15] = m * vz;
  S1[18] = rho;
  S1[20] = 0.5 * m * v2;
  S1[21] = rho * vx;
  S1[22] = rho * vy;
  S1[23] = rho * vz;
  S1[24] = 1.0 / gmm1;
  /* S4 rows: same R rows; columns w = (n, vx, vy, vz, P); Lc frozen. */
  S4[5] = m * dx_;
  S4[6] = m * ndot;
  S4[10] = m * dy_;
  S4[12] = m * ndot;
  S4[15] = m * dz_;
  S4[18] = m * ndot;
  S4[20] = m * vdv;
  S4[21] = m * (ndot * vx + n * dx_);
  S4[22] = m * (ndot * vy + n * dy_);
  S4[23] = m * (ndot * vz + n * dz_);
  /* R = S1 * lc written out. */
  R5[0] = m * ndot;
  R5[1] = m * ndot * vx + rho * dx_;
  R5[2] = m * ndot * vy + rho * dy_;
  R5[3] = m * ndot * vz + rho * dz_;
  R5[4] = 0.5 * m * v2 * ndot + rho * vdv + pdot / gmm1;
}

KOKKOS_INLINE_FUNCTION
static void seg_s3(double m, double rho, const double *w, double gmm1,
                   double *S3) {
  double vx = w[1], vy = w[2], vz = w[3];
  double v2 = vx * vx + vy * vy + vz * vz;
  double rs = rho > XMF_SRC_FLOOR ? rho : XMF_SRC_FLOOR;
  int q;
  for (q = 0; q < 25; q++)
    S3[q] = 0.0;
  /* Rows w = (n, vx, vy, vz, P); columns U = (rho, rho vx, rho vy,
     rho vz, E). n = rho/m, v = momentum/rho, P = (gamma-1)(E - ek). */
  S3[0] = 1.0 / m;
  S3[5] = -vx / rs;
  S3[6] = 1.0 / rs;
  S3[10] = -vy / rs;
  S3[12] = 1.0 / rs;
  S3[15] = -vz / rs;
  S3[18] = 1.0 / rs;
  S3[20] = 0.5 * gmm1 * v2;
  S3[21] = -gmm1 * vx;
  S3[22] = -gmm1 * vy;
  S3[23] = -gmm1 * vz;
  S3[24] = gmm1;
}

/* ---- Per-cell linearised implicit source solve (C1a extraction) ---------
   The exact per-cell body of the former serial triple loop, lifted into a
   function so every interior cell becomes one independent iteration of a
   Kokkos parallel iteration. All arrays in the argument list are that
   cell's PRIVATE workspace row (the driver allocates one row of 6*n +
   6*n*n doubles per cell); the tape masks are shared read-only. The
   source step is strictly cell-local -- tapes read only this cell's slots
   and the writes land only in that cell -- so the iteration order cannot
   change the numerics. resid_p points at the iteration's private share of
   the step-wide max of the report-only one-step residual (the driver
   folds the shares with a Kokkos Max reducer). Returns 0, or -1 on
   tape-eval / LU failure; every failure return before section (f)
   precedes this cell's write-back, so a cell that fails its solve never
   leaves a partial state behind.

   C2: device twin over the SrcCtx bundle -- fields / P_e / mass table /
   tapes / masks are device pointers; the tape evaluator is the shared
   xmf::ir_eval. Every formula is the C1b one verbatim. */
KOKKOS_INLINE_FUNCTION
static int source_cell(const SrcCtx &s, int k, int j, int i, double *Wv,
                       double *Lc, double *R, double *Ustar, double *Unew,
                       double *b, double *J2, double *S1, double *S4,
                       double *S3, double *T1, double *A, double *resid_p) {
  int f, t, col;
  int n = s.n, nf5 = s.nf5;
  double gmm1 = s.gmm1, dt = s.dt;
  size_t c = scidx(s, k, j, i);
  /* (a) Read w and U* from the current (post-hydro) fields. */
  for (f = 0; f < s.nfluid; f++) {
    double rho = src_slot(s, f, XMF_SLOT_RHO)[c];
    double m = s.mass_code[f];
    Wv[f * 5 + 0] = rho / m;
    Wv[f * 5 + 1] = src_slot(s, f, XMF_SLOT_VX)[c];
    Wv[f * 5 + 2] = src_slot(s, f, XMF_SLOT_VY)[c];
    Wv[f * 5 + 3] = src_slot(s, f, XMF_SLOT_VZ)[c];
    Wv[f * 5 + 4] = src_slot(s, f, XMF_SLOT_P)[c];
    Ustar[f * 5 + 0] = rho;
    Ustar[f * 5 + 1] = rho * Wv[f * 5 + 1];
    Ustar[f * 5 + 2] = rho * Wv[f * 5 + 2];
    Ustar[f * 5 + 3] = rho * Wv[f * 5 + 3];
    Ustar[f * 5 + 4] =
        Wv[f * 5 + 4] / gmm1 +
        0.5 * rho * (Wv[f * 5 + 1] * Wv[f * 5 + 1] +
                     Wv[f * 5 + 2] * Wv[f * 5 + 2] +
                     Wv[f * 5 + 3] * Wv[f * 5 + 3]);
  }
  if (s.pe) {
    /* B2: P_e is pressure-only -- its "conservative" unknown IS the
       pressure itself (the SSP staging treats it the same way). */
    Wv[n - 1] = s.pe[c];
    Ustar[n - 1] = s.pe[c];
  }
  /* (b) Tape rates L_c at U* and the lift R = T(w, L_c); block-
         diagonal S1/S4/S3 assembled per fluid. */
  if (eval_lc(s, k, j, i, Lc))
    return -1;
  {
    size_t q;
    for (q = 0; q < (size_t)n * n; q++) {
      S1[q] = 0.0;
      S4[q] = 0.0;
      S3[q] = 0.0;
    }
  }
  for (f = 0; f < s.nfluid; f++) {
    double blk1[25], blk4[25], blk3[25], r5[5];
    int rr, cc;
    seg_s1_s4_r(s.mass_code[f], Ustar[f * 5 + 0], Wv + f * 5, Lc + f * 5,
                gmm1, blk1, blk4, r5);
    seg_s3(s.mass_code[f], Ustar[f * 5 + 0], Wv + f * 5, gmm1, blk3);
    for (rr = 0; rr < 5; rr++) {
      R[f * 5 + rr] = r5[rr];
      for (cc = 0; cc < 5; cc++) {
        S1[(size_t)(f * 5 + rr) * n + f * 5 + cc] = blk1[rr * 5 + cc];
        S4[(size_t)(f * 5 + rr) * n + f * 5 + cc] = blk4[rr * 5 + cc];
        S3[(size_t)(f * 5 + rr) * n + f * 5 + cc] = blk3[rr * 5 + cc];
      }
    }
  }
  if (s.pe) {
    /* B2 P_e row/column: the rate enters as-is (pressure-only, no
       lift -- S1[n-1][n-1] = 1 gives R_pe = Lc_pe) and the unknown
       IS the pressure (S3[n-1][n-1] = 1). S4 stays zero there: the
       lift has no w-dependence to freeze. */
    R[n - 1] = Lc[n - 1];
    S1[(size_t)(n - 1) * n + (n - 1)] = 1.0;
    S3[(size_t)(n - 1) * n + (n - 1)] = 1.0;
  }
  /* (c) S2 = dL_c/dw by forward AD of the loaded tapes. Columns
         seed stored slots; the density column arrives as
         d/d(rho_stored) and becomes d/dn after multiplying by the
         fluid mass. B2: the last column (when P_e exists) seeds the
         packed ELEC P_e code, and dP_e tapes write the last row.
         Tapes whose mask misses the column are skipped (r / F_b /
         constants are frozen inside the Newton step). */
  {
    size_t q;
    for (q = 0; q < (size_t)n * n; q++)
      J2[q] = 0.0;
  }
  for (col = 0; col < n; col++) {
    int seed = col < nf5
                   ? (((int)XMF_IR_FLUID << 16) | ((col / 5) << 8) |
                      (col % 5))
                   : (((int)XMF_IR_ELEC << 16) |
                      ((int)XMF_IR_PE << 8));
    for (t = 0; t < s.ntape; t++) {
      const int *desc = s.desc + 6 * t;
      int dest_kind = desc[0];
      int dest_fluid = desc[1];
      int nops = desc[2];
      int nconst = desc[3];
      const int *ops = s.ops + (size_t)desc[4];
      const double *consts = s.consts + (size_t)desc[5];
      double val, d;
      int ff0, ff1, ff;
      if (!XMF_MASK_GET(s.mask + (size_t)t * s.nwords, col))
        continue; /* tape independent of this state slot */
      if (xmf::ir_eval(s.ir, ops, consts, nops, nconst, k, j, i, seed, &val,
                       &d))
        return -1;
      if (dest_kind == XMF_IR_DPE) { /* P_e rate: last row */
        J2[(size_t)(n - 1) * n + col] += d;
        continue;
      }
      {
        int slot = xmf::ir_write_slot(dest_kind);
        if (slot < 0)
          return -1;
        if (dest_fluid < 0) {
          ff0 = 0;
          ff1 = s.nfluid;
        } else {
          ff0 = dest_fluid;
          ff1 = dest_fluid + 1;
        }
        for (ff = ff0; ff < ff1; ff++)
          J2[(size_t)(ff * 5 + slot) * n + col] += d;
      }
    }
  }
  for (f = 0; f < s.nfluid; f++) {
    /* d/d(rho_stored) -> d/dn chain factor at constant m */
    int rr;
    for (rr = 0; rr < n; rr++)
      J2[(size_t)rr * n + f * 5] *= s.mass_code[f];
  }
  /* (d) dR/dU = (S1*S2 + S4)*S3, then A = I - dt*dR/dU and the
         linearised backward-Euler right-hand side b = dt*R(U*). */
  mat_mul(S1, J2, T1, n);
  {
    size_t q;
    for (q = 0; q < (size_t)n * n; q++)
      T1[q] += S4[q]; /* M = S1*S2 + S4 */
  }
  mat_mul(T1, S3, A, n); /* dR/dU */
  {
    int rr, cc;
    for (rr = 0; rr < n; rr++)
      for (cc = 0; cc < n; cc++)
        A[(size_t)rr * n + cc] =
            (rr == cc ? 1.0 : 0.0) - dt * A[(size_t)rr * n + cc];
  }
  for (f = 0; f < n; f++)
    b[f] = dt * R[f];
  /* (e) One dense LU solve: dU, then U^{n+1} = U* + dU with the
         hydro floors applied to the result. */
  if (lu_solve(A, b, n))
    return -1;
  for (f = 0; f < s.nfluid; f++) {
    double U5[5], q5[5], rho, ek;
    int v;
    for (v = 0; v < 5; v++)
      U5[v] = Ustar[f * 5 + v] + b[f * 5 + v];
    if (U5[0] < XMF_SRC_FLOOR)
      U5[0] = XMF_SRC_FLOOR;
    rho = U5[0]; /* cons5_to_prim floors, kept identical */
    q5[0] = rho;
    q5[1] = U5[1] / rho;
    q5[2] = U5[2] / rho;
    q5[3] = U5[3] / rho;
    ek = 0.5 * (U5[1] * U5[1] + U5[2] * U5[2] + U5[3] * U5[3]) / rho;
    q5[4] = gmm1 * (U5[4] - ek);
    if (q5[4] < XMF_SRC_FLOOR)
      q5[4] = XMF_SRC_FLOOR;
    /* (f) Write the new state, then rebuild the conservative state
           consistent with what is stored (floors included) for the
           residual below. */
    src_slot(s, f, XMF_SLOT_RHO)[c] = q5[0];
    src_slot(s, f, XMF_SLOT_VX)[c] = q5[1];
    src_slot(s, f, XMF_SLOT_VY)[c] = q5[2];
    src_slot(s, f, XMF_SLOT_VZ)[c] = q5[3];
    src_slot(s, f, XMF_SLOT_P)[c] = q5[4];
    Unew[f * 5 + 0] = q5[0];
    Unew[f * 5 + 1] = q5[0] * q5[1];
    Unew[f * 5 + 2] = q5[0] * q5[2];
    Unew[f * 5 + 3] = q5[0] * q5[3];
    Unew[f * 5 + 4] = q5[4] / gmm1 +
                      0.5 * q5[0] * (q5[1] * q5[1] + q5[2] * q5[2] +
                                     q5[3] * q5[3]);
  }
  if (s.pe) {
    /* B2 P_e write-back: the solved increment lands DIRECTLY on the
       electron block (never on an ion P), floored like the fluid
       pressures. */
    double pe_new = Ustar[n - 1] + b[n - 1];
    if (pe_new < XMF_SRC_FLOOR)
      pe_new = XMF_SRC_FLOOR;
    s.pe[c] = pe_new;
    Unew[n - 1] = pe_new;
  }
  /* (g) One-step residual r = U* + dt*R(U^{n+1}) - U^{n+1}. The
         tapes are re-evaluated at the just-written state (the IR
         is strictly cell-local, so only this cell sees the update);
         F_b and r stay frozen. Report-only: the solve never
         iterates on it. */
  if (eval_lc(s, k, j, i, Lc))
    return -1;
  for (f = 0; f < s.nfluid; f++) {
    double blk1[25], blk4[25], r5[5];
    int v;
    double wn[5];
    wn[0] = Unew[f * 5 + 0] / s.mass_code[f];
    wn[1] = src_slot(s, f, XMF_SLOT_VX)[c];
    wn[2] = src_slot(s, f, XMF_SLOT_VY)[c];
    wn[3] = src_slot(s, f, XMF_SLOT_VZ)[c];
    wn[4] = src_slot(s, f, XMF_SLOT_P)[c];
    seg_s1_s4_r(s.mass_code[f], Unew[f * 5 + 0], wn, Lc + f * 5, gmm1,
                blk1, blk4, r5);
    for (v = 0; v < 5; v++) {
      double rv = Ustar[f * 5 + v] + dt * r5[v] - Unew[f * 5 + v];
      double scale =
          fabs(Ustar[f * 5 + v]) + dt * fabs(r5[v]) + XMF_SRC_FLOOR;
      double rel = fabs(rv) / scale;
      if (rel > *resid_p)
        *resid_p = rel;
    }
  }
  if (s.pe) {
    /* B2 P_e residual component: R_pe = Lc_pe re-evaluated at the
       just-written state (S1 pe diagonal = 1, no lift). */
    double rpe = Lc[n - 1];
    double rv = Ustar[n - 1] + dt * rpe - Unew[n - 1];
    double scale = fabs(Ustar[n - 1]) + dt * fabs(rpe) + XMF_SRC_FLOOR;
    double rel = fabs(rv) / scale;
    if (rel > *resid_p)
      *resid_p = rel;
  }
  return 0;
}

/* ---- The step -------------------------------------------------------------
   C1a: the interior-cell triple loop became ONE Kokkos parallel loop over
   the nx*ny*nz active cells -- the implicit source step is strictly
   cell-local, so every cell is an independent iteration. Each iteration
   owns one workspace row of the former per-loop buffers (Wv/Lc/R/Ustar/
   Unew/b + the six N x N matrices); the tape masks stay shared read-only.

   C1b: the loop is a parallel_reduce carrying two Kokkos reductions --
   Max<double> for the report-only residual and Sum<int> for the error
   flag (any failed cell). The C1a heap scalars the lambda wrote through
   captured pointers were a data race once more than one thread runs the
   loop, so they are gone. Both folds are exact (max / integer sum): the
   value cannot depend on the iteration or reduction order, hence the
   results stay bit-for-bit the C1a Serial ones. Note the error fold is
   Sum<int>, not Max<int> -- Max<int>'s final() overwrites the scalar
   with its neutral element INT_MIN when no cell failed, which the
   `if (err)` below would read as a failure; Sum<int>'s neutral is 0.
   A failed cell flags the error through its reduction share; the other
   cells finish their own independent work (nothing cross-cell exists to
   stop) and the step returns -1, which xmf_step turns into an aborted
   step (t untouched).

   C2: the workspace rows are a default-memory-space View (device memory
   on the CUDA tree); the host-built mask block is pushed once per step. */

int xmf_source_step(XmfRun *run, double dt) {
  int n, nf5, nwords, t;
  double gmm1;
  unsigned *mask = NULL;
  double resid = 0.0;
  int ret = 0;

  if (!run)
    return -1;
  run->source_resid = 0.0;
  if (run->ntape == 0)
    return 0; /* no sources loaded: pure-hydro path, bit for bit */

  /* B2: N = 5*nfluid plus one trailing P_e unknown when the electron
     block exists. The fluid 5*nfluid block of every matrix below is
     assembled exactly as before, so with electrons off -- and, cell by
     cell, for any tape set that neither reads P_e nor writes dP_e -- the
     solve stays the pre-B2 one. */
  n = xmf_ir_nstate(run);
  nf5 = 5 * run->nfluid;
  gmm1 = run->gamma - 1.0;
  nwords = (int)(((size_t)n + 31) / 32);

  mask = (unsigned *)calloc((size_t)run->ntape * nwords, sizeof(unsigned));
  if (!mask)
    return -1;
  for (t = 0; t < run->ntape; t++)
    tape_mask(run, &run->tapes[t], mask + (size_t)t * nwords);

  {
    const int ng = run->ng, nx = run->nx, ny = run->ny, nz = run->nz;
    size_t ncells_int = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t per_cell = (size_t)6 * n + (size_t)6 * n * n;
    size_t mask_tot = (size_t)run->ntape * (size_t)nwords;
    int err = 0; /* count of reduction shares that saw a failing cell
                    (Sum<int> reduction; 0 iff the sweep was clean) */
    try {
      /* Per-cell workspace rows in the default memory space (device on
         the CUDA tree). */
      Kokkos::View<double **, Kokkos::LayoutRight> ws_v("xmf_src_ws",
                                                         ncells_int,
                                                         per_cell);
      /* Push the host-built tape masks once for the whole step. */
      Kokkos::View<unsigned *> mask_v("xmf_src_mask", mask_tot);
      Kokkos::deep_copy(
          mask_v,
          Kokkos::View<const unsigned *, Kokkos::HostSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(mask,
                                                                mask_tot));
      /* Device bundle: fields / P_e / tables / flattened tapes from the
         run's DeviceState; the evaluator context mirrors xmf_ir.cpp's
         host builder. */
      SrcCtx s;
      xmf::DeviceState *dv = xmf::dev(run);
      s.fields = dv->fields.data();
      s.pe = run->pe ? dv->pe.data() : NULL;
      s.mass_code = dv->mass_code.data();
      s.ir = xmf::ir_ctx_device(run);
      s.desc = dv->tape_desc.data();
      s.ops = dv->tape_ops.data();
      s.consts = dv->tape_consts.data();
      s.mask = mask_v.data();
      s.ntape = dv->ntape;
      s.nwords = nwords;
      s.n = n;
      s.nf5 = nf5;
      s.nfluid = run->nfluid;
      s.nxtot = run->nxtot;
      s.nytot = run->nytot;
      s.ncells = run->ncells;
      s.gmm1 = gmm1;
      s.dt = dt;
      Kokkos::parallel_reduce(
          "xmf_source_step", ncells_int,
          KOKKOS_LAMBDA(const size_t idx, double &resid_upd, int &err_upd) {
            const int k = ng + (int)(idx / ((size_t)ny * (size_t)nx));
            const int j = ng + (int)((idx / (size_t)nx) % (size_t)ny);
            const int i = ng + (int)(idx % (size_t)nx);
            double *ws = &ws_v(idx, 0);
            double *Wv = ws;
            double *Lc = ws + n;
            double *R = ws + 2 * n;
            double *Ustar = ws + 3 * n;
            double *Unew = ws + 4 * n;
            double *b = ws + 5 * n;
            double *J2 = ws + 6 * n;
            double *S1 = ws + 6 * n + (size_t)n * n;
            double *S4 = ws + 6 * n + 2 * (size_t)n * n;
            double *S3 = ws + 6 * n + 3 * (size_t)n * n;
            double *T1 = ws + 6 * n + 4 * (size_t)n * n;
            double *A = ws + 6 * n + 5 * (size_t)n * n;
            if (source_cell(s, k, j, i, Wv, Lc, R, Ustar, Unew, b, J2, S1,
                            S4, S3, T1, A, &resid_upd))
              err_upd = 1; /* Sum<int>: share counts as failing */
          },
          Kokkos::Max<double>(resid), Kokkos::Sum<int>(err));
      if (err)
        ret = -1;
    } catch (...) {
      ret = -1;
    }
  }

  if (ret == 0)
    run->source_resid = resid;
  free(mask);
  return ret;
}

/* Auxiliary READ port for the last step's relative source residual (see
   xmf_core.h). 0 when no source step has run. */
double xmf_source_resid(const XmfRun *run) {
  return run ? run->source_resid : 0.0;
}
