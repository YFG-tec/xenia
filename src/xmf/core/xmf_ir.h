/* Source-term opcode / slot-kind contract (A4.6).

   Python (xmf.source_ir) compiles names into these integers. The C
   kernel must not parse identifier strings. This header is the numeric
   lock: the enums below are frozen values mirrored by xmf/source_ir.py,
   followed by the internal tape container and the seams used by the
   stack machine in xmf_ir.c.

   xmf_step stays pure hydro in this slice: tapes, once loaded through
   the xmf_ir_load helper (declared in xmf_core.h), are evaluated only
   by an explicit xmf_ir_apply call, never automatically. */

#ifndef XMF_IR_H
#define XMF_IR_H

enum XmfIrOp {
  XMF_OP_LOAD_CONST = 0,
  XMF_OP_LOAD_SLOT = 1,
  XMF_OP_ADD = 2,
  XMF_OP_SUB = 3,
  XMF_OP_MUL = 4,
  XMF_OP_DIV = 5,
  XMF_OP_POW = 6,
  XMF_OP_NEG = 7,
  XMF_OP_EXP = 8,
  XMF_OP_LOG = 9
};

enum XmfIrKind {
  XMF_IR_GRID = 0,
  XMF_IR_FLUID = 1,
  XMF_IR_ELEC = 2,
  XMF_IR_FLUX = 3,
  /* A3 supplement: neighbour reads of boundary assignment tapes. Appended
     kind only -- the opcode table above stays frozen. Source tapes never
     carry it (xmf_ir_load refuses it), so it never reaches xmf_ir_apply,
     xmf_ir_jac or the implicit source solve; only xmf_bc.c evaluates it. */
  XMF_IR_NB = 4,
  /* M block 1 (M1): magnetic-field reads (packed slot a = 0/1/2 =
     B1/B2/B3 component). Appended kind only -- the opcode table above
     stays frozen and the fluid five slots never carry it. This slice
     allocates the NAME SPACE only: the tape evaluator has no magnetic
     state it can supply yet (the staggered B blocks are not tape slots),
     so xmf_ir_load / ir_slot_loadable refuse the kind exactly like the
     NB precedent -- appended, enumerated, never silently loadable. */
  XMF_IR_MAG = 5
};

/* Packed slot: (kind << 16) | (a << 8) | b
   GRID:  a = 0/1/2 = x1/x2/x3  (r is x1)
   FLUID: a = fluid index, b = 0..5 = n,vx,vy,vz,P,T
   ELEC:  a = 0/1/2 = n_e,P_e,T_e
   FLUX:  a = beam index
   NB:    a = fluid index, b = (layer << 4) | var with layer 1..2 and
          var = 0..5 = n,vx,vy,vz,P,T (same numbers as FLUID). Layers
          count ACTIVE cells inward from the face being filled:
          @1 = the face's outermost active cell, @2 = the next one in.
          GRID reads in a boundary tape resolve at the ghost cell being
          filled; NB reads resolve at that layer of the same transverse
          row.
   MAG:   a = 0/1/2 = B1/B2/B3 component (M1 name space). Appended kind;
          the C kernel refuses to load it in this slice (no magnetic
          tape-slot supply exists yet), so it never reaches evaluation. */

enum XmfIrGrid { XMF_IR_X1 = 0, XMF_IR_X2 = 1, XMF_IR_X3 = 2 };
enum XmfIrFluidVar {
  XMF_IR_N = 0,
  XMF_IR_VX = 1,
  XMF_IR_VY = 2,
  XMF_IR_VZ = 3,
  XMF_IR_P = 4,
  XMF_IR_T = 5
};
enum XmfIrElec { XMF_IR_NE = 0, XMF_IR_PE = 1, XMF_IR_TE = 2 };
enum XmfIrWrite {
  XMF_IR_DN = 0,
  XMF_IR_DVX = 1,
  XMF_IR_DVY = 2,
  XMF_IR_DVZ = 3,
  XMF_IR_DP = 4,
  /* B2 appended write name: the increment enters the optional electron
     pressure block P_e DIRECTLY (never a fluid slot, never "write into
     ion P then project"). dest_fluid is -1 and the run must have the P_e
     block (electrons on) -- xmf_ir_load refuses it otherwise. Appended
     name only: the opcode table above stays frozen. */
  XMF_IR_DPE = 5,
  /* M block 1 (M1) appended write names: dBx/dBy/dBz increments for the
     magnetic block (kind = the block's own direction, never a fluid).
     The numbers are assigned now so the IR stays append-only, but this
     slice wires no magnetic increment target -- xmf_ir_load refuses these
     kinds exactly like the rk4 precedent (enumerated, never silently
     runnable). dest_fluid is always -1. */
  XMF_IR_DBX = 6,
  XMF_IR_DBY = 7,
  XMF_IR_DBZ = 8
};

/* One compiled tape after xmf_ir_load has copied it into the run.
   Internal container only -- the Python side never sees this struct, it
   ships flat int/double arrays. ops holds nops (opcode, argument) pairs:
   ops[2*t] is the opcode (enum XmfIrOp) and ops[2*t+1] the argument --
   a const index for LOAD_CONST, a packed slot for LOAD_SLOT, ignored
   otherwise. */
typedef struct {
  int dest_kind;  /* enum XmfIrWrite: which increment the value enters */
  int dest_fluid; /* fluid index, or -1 for bare dvx/dvy/dvz = every fluid
                     and for dP_e (the electron pressure block is not a
                     fluid, B2) */
  int nops;
  int nconst;
  int *ops;       /* 2*nops ints */
  double *consts; /* nconst doubles */
} XmfTape;

/* ---- Internal seams (NOT public API) ------------------------------------
   No XMF_API here and nothing from this block ever enters xmf_core.h;
   these are only the seams between xmf_ir.c and the rest of the kernel.
   C linkage: C++ translation units (C1a loop modules) call into the C
   stack machine. */

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate one tape at cell (k,j,i) on dual numbers {v, vdot} (A4.7).
   The frozen opcodes 0..9 are unchanged; the arithmetic is the dual
   extension of A4.6:
     LOAD_CONST  v = c                 vdot = 0
     LOAD_SLOT   v = slot value        vdot = 1 if the slot is the seeded
                                        independent variable, else 0
     ADD / SUB   v = a +/- b           vdot = adot +/- bdot
     MUL         v = a b               vdot = adot b + a bdot
     DIV         v = a / b             vdot = (adot b - a bdot) / b^2
     POW         v = a^b               vdot = a^b (bdot log a + b adot/a)
     NEG         v = -a                vdot = -adot
     EXP         v = e^a               vdot = e^a adot
     LOG         v = log a             vdot = adot / a
   seed_slot is -1 (no independent variable: every vdot stays 0; this is
   the xmf_ir_apply path, whose .v values remain bit-for-bit the A4.6
   values) or a packed slot code (the same encoding LOAD_SLOT carries),
   which must be loadable. Writes *out_v always and *out_vdot when
   out_vdot is not NULL. Returns 0, or -1 on a malformed tape /
   unreadable slot (cannot happen for tapes accepted by xmf_ir_load) /
   unloadable seed / a POW base <= 0 whenever a derivative is
   requested. */
int xmf_ir_eval(const struct XmfRun *run, const XmfTape *tape, int k, int j,
                int i, int seed_slot, double *out_v, double *out_vdot);

/* Free the run's loaded tape set (NULL-safe); afterwards
   run->tapes == NULL and run->ntape == 0. */
void xmf_ir_clear(struct XmfRun *run);

/* Map a write kind (enum XmfIrWrite) to the fluid field slot it
   increments (DN -> 0, DVX -> 1, DVI -> 2, DVZ -> 3, DP -> 4); -1
   otherwise. DPE stays -1 here on purpose: P_e is not a fluid slot, so
   xmf_ir_apply / xmf_source.c branch on dest_kind == XMF_IR_DPE itself
   and write the electron block directly. Shared with xmf_source.c, which
   gathers L_c with the same ADDS convention as xmf_ir_apply. */
int xmf_ir_write_slot(int kind);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
