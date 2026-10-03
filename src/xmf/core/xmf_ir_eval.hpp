/* Device-capable IR tape evaluator (C2) -- still the xmf_ir module.

   The dual-number stack machine of xmf_ir.c lifted to KOKKOS_INLINE
   functions so the implicit source step's per-cell solve can evaluate
   the loaded tapes on the device execution space. The opcode table 0..9
   (xmf_ir.h) is untouched and every arithmetic expression below is the
   verbatim xmf_ir.c one -- the host-side xmf_ir_eval now wraps this very
   code path, so host and device evaluations share one implementation
   instead of drifting apart.

   This header is C++ only (the pure-C translation units never see it)
   and must be included AFTER xmf_run.h: it uses XMF_SLOT_NVAR, XMF_KB
   and the XmfIr* enums. The evaluator is stateless except the IrCtx
   value bundle (pointers + scalars), which both execution spaces build
   from their own side of the mirror pair:

   - host callers  (xmf_ir.c apply / jac / eval_tape): mirrors of run;
   - device callers (xmf_source.cpp): the device Views of xmf_device.

   Grid centres x1/x2/x3, the beam flux F_b and constants are never
   seeded by the Newton solve -- the seed contract of xmf_ir.h carries
   over unchanged (seed_slot < 0 keeps every derivative 0, bit for bit
   the A4.6 value path). */

#ifndef XMF_IR_EVAL_HPP
#define XMF_IR_EVAL_HPP

#ifdef __cplusplus

#include <Kokkos_Macros.hpp>

#include <math.h>
#include <stddef.h>

#include "xmf_ir.h"

namespace xmf {

/* Value bundle the evaluator may dereference. All pointers are valid in
   the calling execution space (host mirrors or device Views). pe is NULL
   when electrons are off; flux/finf are NULL when no radiation state is
   loaded (FLUX slots then refuse to load exactly like the host code);
   rad_enabled selects the scanned flux vs the boundary value finf for
   FLUX reads. */
struct IrCtx {
  const double *fields;    /* [nfluid][5][ncells] */
  const double *pe;        /* [ncells], NULL when electrons off */
  const double *mass_code; /* [nfluid] */
  const double *charge;    /* [nfluid] */
  const double *x1;        /* cell centres of the three directions */
  const double *x2;
  const double *x3;
  const double *flux;      /* [nbeam][ncells] scanned flux, NULL = no rad */
  const double *finf;      /* [nbeam] boundary flux */
  int nfluid;
  int nbeam;
  int rad_enabled;
  int nxtot, nytot;
  size_t ncells;
  double unit_P;
};

/* Linear cell index of (k,j,i); i fastest (the xmf_run.h cidx layout). */
KOKKOS_INLINE_FUNCTION
size_t ir_cidx(const IrCtx &g, int k, int j, int i) {
  return (((size_t)k * (size_t)g.nytot) + (size_t)j) * (size_t)g.nxtot +
         (size_t)i;
}

/* A packed slot the evaluator can actually supply -- the xmf_ir.c
   slot_loadable contract on the IrCtx bundle. */
KOKKOS_INLINE_FUNCTION
int ir_slot_loadable(const IrCtx &g, int code) {
  int kind, a, b;
  if (code < 0)
    return 0;
  kind = (code >> 16) & 0xFF;
  a = (code >> 8) & 0xFF;
  b = code & 0xFF;
  switch (kind) {
  case XMF_IR_GRID:
    return a <= XMF_IR_X3 && b == 0;
  case XMF_IR_FLUID:
    return a < g.nfluid && b <= XMF_IR_T; /* T loadable (A5) */
  case XMF_IR_ELEC:
    /* With the P_e block allocated all three ELEC slots are loadable --
       P_e stored, n_e / T_e the quasi-neutral derived reads. Electrons
       off -> the whole kind refused. */
    return a <= XMF_IR_TE && g.pe != NULL;
  case XMF_IR_FLUX:
    /* Beam flux F_a: loadable exactly when a radiation state with at
       least a+1 beams has been loaded (flux non-NULL iff rad loaded). */
    return g.flux != NULL && a < g.nbeam;
  case XMF_IR_MAG:
    /* M1: the magnetic name space is enumerated but the evaluator has no
       B supply this slice -- refused, exactly the NB precedent (the
       default branch below would refuse it anyway; the explicit case is
       the contract marker). */
    return 0;
  default:
    return 0;
  }
}

/* Write kind (enum XmfIrWrite) -> fluid field slot it increments; -1
   otherwise. Device twin of the extern "C" xmf_ir_write_slot (the
   numbers coincide by the frozen enums, kept as an explicit switch). */
KOKKOS_INLINE_FUNCTION
int ir_write_slot(int kind) {
  switch (kind) {
  case XMF_IR_DN:
    return 0; /* XMF_SLOT_RHO */
  case XMF_IR_DVX:
    return 1; /* XMF_SLOT_VX */
  case XMF_IR_DVY:
    return 2; /* XMF_SLOT_VY */
  case XMF_IR_DVZ:
    return 3; /* XMF_SLOT_VZ */
  case XMF_IR_DP:
    return 4; /* XMF_SLOT_P */
  default:
    return -1;
  }
}

/* Read one packed slot at cell (k,j,i) and its derivative with respect
   to the single seeded stored slot -- the xmf_ir.c slot_value_seed
   contract, verbatim arithmetic. *out_d is the directional derivative;
   it is 0 whenever seed_slot < 0 (the apply path). */
KOKKOS_INLINE_FUNCTION
int ir_slot_value_seed(const IrCtx &g, int code, size_t c, int k, int j,
                       int i, int seed_slot, double *out_v, double *out_d) {
  int kind = (code >> 16) & 0xFF;
  int a = (code >> 8) & 0xFF;
  int b = code & 0xFF;
  int skind = -1, sa = -1, sb = -1;
  if (seed_slot >= 0) {
    skind = (seed_slot >> 16) & 0xFF;
    sa = (seed_slot >> 8) & 0xFF;
    sb = seed_slot & 0xFF;
  }
  switch (kind) {
  case XMF_IR_GRID: {
    double v;
    if (b != 0)
      return -1;
    if (a == XMF_IR_X1)
      v = g.x1[i];
    else if (a == XMF_IR_X2)
      v = g.x2[j];
    else if (a == XMF_IR_X3)
      v = g.x3[k];
    else
      return -1;
    *out_v = v;
    *out_d = (seed_slot == code) ? 1.0 : 0.0;
    return 0;
  }
  case XMF_IR_FLUID: {
    if (a >= g.nfluid)
      return -1;
    const double *fa = g.fields + ((size_t)a * XMF_SLOT_NVAR) * g.ncells;
    if (b == XMF_IR_T) {
      /* T = P_code*unit_P/(n*kB); n = rho/mass_code floored against 0. */
      double rho = fa[c];
      double pc = fa[(size_t)XMF_IR_P * g.ncells + c];
      double m = g.mass_code[a];
      double n = rho / m;
      double n_safe = n > 1e-30 ? n : 1e-30;
      *out_v = pc * g.unit_P / (n_safe * XMF_KB);
      if (skind == XMF_IR_FLUID && sa == a && sb == 0)
        *out_d = -pc * g.unit_P / (n_safe * n_safe * XMF_KB) / m;
      else if (skind == XMF_IR_FLUID && sa == a && sb == XMF_IR_P)
        *out_d = g.unit_P / (n_safe * XMF_KB);
      else
        *out_d = 0.0;
      return 0;
    }
    if (b == 0) { /* number density n = rho / mass_code */
      double m = g.mass_code[a];
      *out_v = fa[c] / m;
      *out_d = (skind == XMF_IR_FLUID && sa == a && sb == 0) ? 1.0 / m : 0.0;
      return 0;
    }
    if (b <= XMF_IR_P) {
      *out_v = fa[(size_t)b * g.ncells + c];
      *out_d = (seed_slot == code) ? 1.0 : 0.0;
      return 0;
    }
    return -1;
  }
  case XMF_IR_ELEC: {
    if (!g.pe)
      return -1;
    if (a == XMF_IR_PE) {
      *out_v = g.pe[c];
      *out_d = (seed_slot == code) ? 1.0 : 0.0;
      return 0;
    }
    if (a == XMF_IR_NE || a == XMF_IR_TE) {
      /* B1 quasi-neutral derived reads: n_e = sum_ion q_s n_s with
         n_s = rho_s/mass_code_s (ions recognised by q_s != 0, never by
         name) and T_e = P_e*unit_P/(n_e kB), n_e floored at 1e-30 exactly
         like the fluid T closure. Seeded derivatives run along the stored
         fluid rho slots (and the P_e slot itself). */
      double ne = 0.0;
      int s;
      for (s = 0; s < g.nfluid; s++)
        ne += g.charge[s] *
              g.fields[((size_t)s * XMF_SLOT_NVAR) * g.ncells + c] /
              g.mass_code[s];
      if (a == XMF_IR_NE) {
        *out_v = ne;
        *out_d = 0.0;
        if (seed_slot == code)
          *out_d = 1.0;
        else if (skind == XMF_IR_FLUID && sa >= 0 && sa < g.nfluid &&
                 sb == 0 && g.charge[sa] != 0.0)
          *out_d = g.charge[sa] / g.mass_code[sa];
        return 0;
      }
      { /* XMF_IR_TE */
        double ne_safe = ne > 1e-30 ? ne : 1e-30;
        *out_v = g.pe[c] * g.unit_P / (ne_safe * XMF_KB);
        *out_d = 0.0;
        if (seed_slot == code)
          *out_d = 1.0;
        else if (skind == XMF_IR_ELEC && sa == XMF_IR_PE && sb == 0)
          *out_d = g.unit_P / (ne_safe * XMF_KB);
        else if (skind == XMF_IR_FLUID && sa >= 0 && sa < g.nfluid &&
                 sb == 0 && g.charge[sa] != 0.0)
          *out_d = -g.pe[c] * g.unit_P / (ne_safe * ne_safe * XMF_KB) *
                   (g.charge[sa] / g.mass_code[sa]);
        return 0;
      }
    }
    return -1;
  }
  case XMF_IR_FLUX:
    if (!g.flux || a >= g.nbeam)
      return -1;
    *out_v = g.rad_enabled ? g.flux[(size_t)a * g.ncells + c] : g.finf[a];
    *out_d = (seed_slot == code) ? 1.0 : 0.0;
    return 0;
  default:
    return -1;
  }
}

/* Maximum stack depth of one tape evaluation (the xmf_ir.c lock). */
#define XMF_IR_EVAL_STACK 64

/* One tape at cell (k,j,i) on dual numbers {v, vdot}. ops carries nops
   (opcode, argument) pairs, consts nconst doubles -- either straight
   from an XmfTape (host path) or from the flattened device tape arrays.
   The frozen opcodes 0..9 and the dual arithmetic are the xmf_ir.c
   xmf_ir_eval ones, verbatim. Returns 0, or -1 on a malformed tape /
   unreadable slot / unloadable seed / a POW base <= 0 whenever a
   derivative is requested. Writes *out_v always and *out_vdot when it
   is not NULL. */
KOKKOS_INLINE_FUNCTION
int ir_eval(const IrCtx &g, const int *ops, const double *consts, int nops,
            int nconst, int k, int j, int i, int seed_slot, double *out_v,
            double *out_vdot) {
  struct Dual {
    double v;
    double d;
  };
  Dual st[XMF_IR_EVAL_STACK];
  int sp = 0, p;
  size_t c = ir_cidx(g, k, j, i);
  if (seed_slot >= 0 && !ir_slot_loadable(g, seed_slot))
    return -1;
  for (p = 0; p < nops; p++) {
    int op = ops[2 * p];
    int arg = ops[2 * p + 1];
    switch (op) {
    case XMF_OP_LOAD_CONST:
      if (sp >= XMF_IR_EVAL_STACK || arg < 0 || arg >= nconst)
        return -1;
      st[sp].v = consts[arg];
      st[sp].d = 0.0;
      sp++;
      break;
    case XMF_OP_LOAD_SLOT: {
      double v, dv;
      if (sp >= XMF_IR_EVAL_STACK)
        return -1;
      if (ir_slot_value_seed(g, arg, c, k, j, i, seed_slot, &v, &dv))
        return -1;
      st[sp].v = v;
      st[sp].d = dv;
      sp++;
      break;
    }
    case XMF_OP_NEG:
      if (sp < 1)
        return -1;
      st[sp - 1].v = -st[sp - 1].v;
      st[sp - 1].d = -st[sp - 1].d;
      break;
    case XMF_OP_EXP:
      if (sp < 1)
        return -1;
      st[sp - 1].v = exp(st[sp - 1].v);
      st[sp - 1].d = st[sp - 1].v * st[sp - 1].d;
      break;
    case XMF_OP_LOG: {
      double a;
      if (sp < 1)
        return -1;
      a = st[sp - 1].v;
      st[sp - 1].v = log(a);
      st[sp - 1].d = st[sp - 1].d / a;
      break;
    }
    default: {
      Dual b, a;
      double v, d;
      if (sp < 2)
        return -1;
      b = st[--sp];
      a = st[--sp];
      if (op == XMF_OP_ADD) {
        v = a.v + b.v;
        d = a.d + b.d;
      } else if (op == XMF_OP_SUB) {
        v = a.v - b.v;
        d = a.d - b.d;
      } else if (op == XMF_OP_MUL) {
        v = a.v * b.v;
        d = a.d * b.v + a.v * b.d;
      } else if (op == XMF_OP_DIV) {
        v = a.v / b.v;
        d = (a.d * b.v - a.v * b.d) / (b.v * b.v);
      } else if (op == XMF_OP_POW) {
        if (a.v <= 0.0 && (seed_slot >= 0 || out_vdot))
          return -1; /* derivative needs log a and b/a: require a > 0.
                        The seedless value-only path (xmf_ir_apply) keeps
                        pow(a,b) itself, exactly the A4.6 value. */
        v = pow(a.v, b.v);
        d = v * (b.d * log(a.v) + b.v * a.d / a.v);
      } else
        return -1;
      st[sp].v = v;
      st[sp].d = d;
      sp++;
    }
    }
  }
  if (sp != 1)
    return -1;
  *out_v = st[0].v;
  if (out_vdot)
    *out_vdot = st[0].d;
  return 0;
}

}  // namespace xmf

#endif /* __cplusplus */

#endif /* XMF_IR_EVAL_HPP */
