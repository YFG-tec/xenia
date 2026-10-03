#ifndef XMF_CORE_H
#define XMF_CORE_H

#include "xmf_config.h"

#ifdef _WIN32
#define XMF_API __declspec(dllexport)
#else
#define XMF_API
#endif

typedef struct XmfRun XmfRun;

/* The kernel mixes C translation units (xmf_core.c and friends) with C++
   ones (C1a: the Kokkos loop modules + the lifecycle base); every exported
   symbol keeps C linkage, so Python's ctypes loader finds the unmangled
   names on all toolchains. */
#ifdef __cplusplus
extern "C" {
#endif

/* Setup. Besides the flat config blob, the three per-direction ACTIVE face
   lattices are mandatory: xf1/xf2/xf3 point to n[0]+1 / n[1]+1 / n[2]+1
   absolute interface positions of the active domain (ghosts not included;
   the variable-length arrays never enter the flat XmfConfig). The kernel
   validates strict monotonicity (xf[i+1] > xf[i], else NULL), pads ghosts
   with the adjacent active width, derives centres/widths and the A/dV
   metrics (cartesian included: A=1, dV=dx). For spherical geometry any x1
   interface (ghosts included) at r <= 0 fails setup. The Python front end
   builds these arrays (uniform or stretched alike); the kernel knows no
   stretch recipe, only faces. */
XMF_API XmfRun *xmf_setup(const XmfConfig *cfg, const double *xf1,
                          const double *xf2, const double *xf3);
XMF_API int xmf_step(XmfRun *run);
XMF_API double xmf_time(const XmfRun *run);
XMF_API double xmf_dt(const XmfRun *run);
XMF_API int xmf_ncell(const XmfRun *run);
XMF_API void xmf_fields(const XmfRun *run, double *x, double *rho, double *v,
                        double *p);
XMF_API void xmf_finalize(XmfRun *run);

/* Read-only mesh queries (helper style; they copy data out and mutate
   nothing). d is the direction: 0=x1, 1=x2, 2=x3. See xmf_core.c for the
   exact semantics (lengths, units, return values). */
XMF_API int xmf_mesh_ncell(const XmfRun *run, int d);  /* n_d + 2*ng */
XMF_API int xmf_mesh_nface(const XmfRun *run, int d);  /* ncell + 1 */
XMF_API int xmf_mesh_copy_x(const XmfRun *run, int d, double *out);
XMF_API int xmf_mesh_copy_xf(const XmfRun *run, int d, double *out);
XMF_API int xmf_mesh_copy_dx(const XmfRun *run, int d, double *out);
XMF_API int xmf_mesh_copy_A(const XmfRun *run, int d, double *out);
XMF_API int xmf_mesh_copy_dV(const XmfRun *run, int d, double *out);

/* Multi-fluid state queries (A2, helper style: they copy data in or out
   and mutate nothing else). Fields are laid out [fluid][slot][cell] with
   the fixed slot order (n, vx, vy, vz, P) == offsets 0..4; fluids are
   addressed only by integer index f in [0, xmf_state_nfluid). Every
   fluid/slot block is xmf_state_ncells(run) doubles long:
   nxtot*nytot*nztot including ghosts -- not the interior x1 count
   returned by xmf_ncell. pack copies a caller buffer INTO the kernel,
   unpack copies OUT into a caller buffer; both return the number of
   doubles moved, or -1 (moving nothing) on invalid f / slot / buffer.
   The optional electron pressure P_e (electron_mode == pressure-only) is
   one separate block of the same length -- never a fluid slot, never
   counted in nfluid; copy_pe writes it out, write_pe copies into it, and
   both return -1 without touching anything when electrons are off. */
XMF_API int xmf_state_ncells(const XmfRun *run); /* ghosts included */
XMF_API int xmf_state_nfluid(const XmfRun *run);
XMF_API int xmf_state_pack(XmfRun *run, int f, int var, const double *in);
XMF_API int xmf_state_unpack(const XmfRun *run, int f, int var, double *out);
XMF_API int xmf_state_copy_pe(const XmfRun *run, double *out);
XMF_API int xmf_state_write_pe(XmfRun *run, const double *in);

/* Physics constants + per-fluid masses (A5). Auxiliary LOAD PORT at the
   same helper level as xmf_ir_load -- NOT a fifth public verb and NOT a
   field of the flat XmfConfig. mass_amu carries nfluid atomic masses
   (amu); the kernel converts each to the code mass density per number
   density mass_code = mass_amu*XMF_AMU/unit_density and stores unit_P =
   unit_density*unit_velocity^2 for the T = P*unit_P/(n*kB) closure.
   Until this port is called, every mass_code is 1.0 and every unit is
   1.0, so LOAD n returns slot 0 unchanged and its seeded derivative is
   exactly 1 -- the pre-A5 numerics, bit for bit. Returns 0, or -1 on
   invalid arguments (NULL mass_amu, non-positive mass/unit, no run). */
XMF_API int xmf_phys_load(XmfRun *run, const double *mass_amu,
                          double unit_density, double unit_velocity,
                          double unit_length);

/* Per-fluid charges (B1). Auxiliary LOAD PORT at the same helper level as
   xmf_phys_load -- NOT a fifth public verb and NOT a field of the flat
   XmfConfig (the variable-length charge table never enters the blob).
   charge_e carries nfluid charges in elementary-charge multiples q_s;
   the kernel stores them as-is and its ion role is q_s != 0 -- species
   names are never consulted and the kind/charge consistency is enforced
   by the Python config layer. The charges feed the quasi-neutral
   electron closure (n_e = sum_ion q_s n_s, share w_s = q_s n_s / n_e,
   P_eff = P_s + w_s P_e in the sweeps) and the derived IR reads n_e/T_e.
   Until this port is called every charge is 0.0, the closure gate stays
   off and the hydro path is bit-for-bit the pre-B1 one. Call it after
   xmf_phys_load (n_e divides by mass_code) and before the first step.
   Returns 0, or -1 on invalid arguments (NULL charge_e, a non-finite
   entry, no run). */
XMF_API int xmf_charge_load(XmfRun *run, const double *charge_e);

/* Per-face boundary helpers (A3). Face indices are fixed: 0=x1beg,
   1=x1end, 2=x2beg, 3=x2end, 4=x3beg, 5=x3end; the type integers are
   enum XmfBcType from xmf_config.h (reflect=0, copy=1, outflow=2,
   user=3, none=4). xmf_bc_type returns the type stored for one face,
   or -1 on invalid arguments. xmf_fill_bc is a helper on the same
   level as xmf_state_pack -- it immediately re-pads every face's ghost
   cells according to the run's current six face types (the exact fill
   that setup and every step perform); it is not a fifth public verb.
   Returns 0, or -1 on invalid arguments. */
XMF_API int xmf_bc_type(const XmfRun *run, int face);
XMF_API int xmf_fill_bc(XmfRun *run);

/* Source-tape helpers (A4.6), helper level like xmf_state_pack -- NOT a
   fifth public verb: the public surface stays the four-step API
   (setup / step / fields / finalize). Tapes are compiled by Python
   (xmf.source_ir) into integer opcodes and packed slots; the numeric
   contract lives in xmf_ir.h. The kernel never parses a formula string,
   never recognises a component name and carries no specialized chemistry
   or body-force formula -- only the loaded integer tapes.

   xmf_ir_load replaces any previously loaded tape set. header carries
   four ints per tape -- dest_kind (enum XmfIrWrite), dest_fluid (fluid
   index, or -1 for bare dvx/dvy/dvz = every fluid and for dP_e), nops
   (>= 1), nconst -- then ops holds nops (opcode, argument) pairs per
   tape and consts holds nconst doubles per tape, both concatenated in
   tape order (consts may be NULL when the whole set has zero
   constants). Slots the kernel cannot supply in this slice are refused
   at load: electron slots when electrons are off (no P_e block; with
   the block present P_e is stored and n_e/T_e are the B1 quasi-neutral
   derived reads), and FLUX when no beam state is loaded. B2: the write
   name dP_e is refused whenever the P_e block is absent and never takes
   a fluid index. Returns the tape count, or -1 on invalid arguments /
   allocation failure (the previous set stays untouched then);
   ntape == 0 clears the set.

   Locked convention: one xmf_ir_apply call evaluates every tape on
   every interior cell (ghosts are not evaluated) and ADDS the tape's
   instantaneous value to the write target (dvx -> vx += expr,
   dn -> n += expr, dP -> P += expr, and B2's dP_e -> P_e += expr
   DIRECTLY on the electron block -- never through a fluid slot). It
   does NOT multiply by dt and
   does NOT run hydro. Time semantics (rate vs. increment, how dt
   enters) are A5's business and are not invented here. xmf_step never
   calls apply and never forms J; with zero tapes loaded apply is a
   no-op and xmf_step stays bit-for-bit the pure-hydro step. Returns 0,
   or -1 on invalid arguments. xmf_ir_ntape returns the loaded tape
   count.

   Forward-AD helpers (A4.7), same helper level as xmf_ir_apply. The
   stack machine now evaluates every tape on dual numbers {v, vdot};
   the opcode set 0..9 is unchanged. The independent variables u of the
   implicit Jacobian are exactly the stored per-fluid slots
   (n, vx, vy, vz, P) per fluid, indexed f*5 + slot, so
   N = xmf_ir_nstate(run) = 5 * nfluid. B2: when the P_e block exists
   N = 5 * nfluid + 1 and the electron pressure is the LAST unknown --
   it never enters the fluid five slots; with electrons off N stays
   5 * nfluid. Grid coordinates r/x1/x2/x3, the beam flux F_b and
   constants{} are never seeded: loading them always carries vdot == 0
   and xmf_ir_jac opens no column for them.

   xmf_ir_jac writes the dense row-major N*N Jacobian J = dL_c/du of
   one interior cell (k, j, i are full grid indices, ghosts included):
   row = write-target component in the same f*5 + slot order as the
   state (B2: dP_e tapes own the last row), column = seeded state slot
   (B2: the last column seeds the packed ELEC P_e code when the block
   exists). Each column seeds one slot with udot = 1 (all others 0) and
   runs the very tapes already loaded; a bare dvx/dvy/dvz tape ADDS its
   derivative to every fluid's vx/vy/vz row, matching xmf_ir_apply's
   ADDS convention. An empty tape set writes all zeros. xmf_ir_jac
   never evaluates ghosts and never mutates the run. Returns 0, or -1
   on invalid arguments (non-interior cell, NULL buffer).

   xmf_ir_eval_tape evaluates one loaded tape (index t) at one interior
   cell with a single seeded slot direction -- a self-check seam, not
   an implicit column: seed_slot is -1 (no seed) or any loadable packed
   slot code (the same encoding LOAD_SLOT carries, e.g. the r grid
   slot), whose derivative is seeded with 1. Writes the tape value to
   *out_v and its directional derivative to *out_vdot; for the gravity
   tape dvx = -G*M/r^2, seeding r gives d(dvx)/dr = 2*G*M/r^3. Seeding
   r here never makes r an independent variable of xmf_ir_jac. Returns
   0, or -1 on invalid arguments (bad tape index, non-interior cell,
   unloadable seed, NULL buffers). */
XMF_API int xmf_ir_load(XmfRun *run, int ntape, const int *header,
                        const int *ops, const double *consts);
XMF_API int xmf_ir_apply(XmfRun *run);
XMF_API int xmf_ir_ntape(const XmfRun *run);
XMF_API int xmf_ir_jac(const XmfRun *run, int k, int j, int i, double *J);
XMF_API int xmf_ir_nstate(const XmfRun *run);
XMF_API int xmf_ir_eval_tape(const XmfRun *run, int t, int k, int j, int i,
                             int seed_slot, double *out_v, double *out_vdot);

/* Radiation beam state (A5), auxiliary LOAD PORT like xmf_ir_load -- the
   53-beam table never enters the flat XmfConfig. mode/dilution/sense are
   the RadiationScan enums kept by the card: this slice implements
   mode == 1 (grid_line), dilution == 0 (none), sense == 0 (inward); any
   other value is refused (returns -1, no state stored). absorber is the
   fluid INDEX whose number density sets the optical depth -- the kernel
   never recognises a component name. sigma/hnu/finf carry nbeam values
   each (cm^2, erg, erg/cm^2/s). Allocates the transmitted-flux state
   F_b[nbeam][ncells] and the absorbed increment dF_b; a later call
   replaces the previous state. When enabled == 0 the scan never runs and
   FLUX slots read the boundary value finf (vacuum). Returns nbeam, or -1
   on invalid arguments / unsupported mode / dilution / sense.

   xmf_rad_nbeam returns the loaded beam count (0 when no state).
   xmf_rad_flux_copy copies beam b's transmitted flux over the whole cell
   block (ghosts included, xmf_state_ncells doubles) into out; it returns
   the number of doubles written, or -1 on invalid arguments. This is the
   auxiliary read port for F_b -- not a fifth public verb. */
XMF_API int xmf_rad_load(XmfRun *run, int enabled, int mode, int dilution,
                         int sense, int absorber, int nbeam,
                         const double *sigma, const double *hnu,
                         const double *finf);
XMF_API int xmf_rad_nbeam(const XmfRun *run);
XMF_API int xmf_rad_flux_copy(const XmfRun *run, int b, double *out);

/* Multi-absorber extension of the beam state (B3), APPENDED auxiliary
   load port at the same helper level as xmf_rad_load -- NOT a fifth
   public verb and never part of the flat XmfConfig. Requires a beam
   table already loaded by xmf_rad_load (its nbeam sets the row length);
   absorbers carries nabs fluid INDICES (the kernel never recognises a
   component name) and sigma the nabs*nbeam photo cross sections in
   row-major order (row s = absorber s, column b = beam b). While this
   table
   is loaded the scan's per-beam optical depth becomes
       tau_b = (sum_s sigma[s*nbeam+b] * n_{absorbers[s]}) * ds
   instead of the single-absorber product; with no table loaded (the
   default) the single-absorber walk stays bit for bit the A5 one. A
   later successful call replaces the previous multi table. nabs < 2 is
   refused -- a single absorber already rides the flat port. Returns
   nabs, or -1 on invalid arguments (no beam state, NULL tables, bad
   absorber index, nabs < 2) with the previous state untouched. */
XMF_API int xmf_rad_load_absorbers(XmfRun *run, int nabs,
                                   const int *absorbers,
                                   const double *sigma);

/* Frozen-cell pin mask (Port C), auxiliary LOAD PORT at the same helper
   level as xmf_bc_set_user -- NOT a fifth public verb. mask carries
   ncells ints (0/1, ghosts included); vals carries
   nfluid*XMF_SLOT_NVAR*ncells doubles (same flat layout as
   xmf_state_pack: rho, vx, vy, vz, P per fluid). Every step and every
   ghost fill automatically overwrites masked cells with their pinned
   values. P_e is never touched; B is never touched (the CT path is not
   pinned by this port). NULL mask or all-zero mask clears the pin. A
   later successful call replaces the previous pin. Returns 0, or -1 on
   invalid arguments / allocation failure (the previous pin stays
   loaded). */
XMF_API int xmf_bc_set_pin(XmfRun *run, const int *mask, const double *vals);

/* User boundary floor (A5), auxiliary LOAD PORT. data carries
   nfluid*XMF_SLOT_NVAR*(nztot*nytot) doubles: the pinned base state in
   code-unit slots [fluid][slot][k*nytot+j]. Every fill of an x1 user
   face writes it back to that side's ghost layers AND the innermost
   active cell; no function pointers. With no buffer loaded a user face
   keeps the pre-A5 copy padding. Returns 0, or -1 on invalid arguments. */
XMF_API int xmf_bc_set_user(XmfRun *run, const double *data);

/* Magnetic-field state (M block 1: M2 staggered storage + M3 B boundary
   conditions), auxiliary LOAD PORT at the same helper level as
   xmf_charge_load / xmf_bc_set_user -- NOT a fifth public verb and never
   part of the flat XmfConfig.

   xmf_mag_load opens the magnetic gate: it allocates the three
   face-centred B blocks (bf) and the three edge-centred E blocks (ef)
   with the staggered layouts documented in xmf_run.h, copies b1/b2/b3
   (full blocks, ghosts included) in as the initial B and leaves E
   calloc'd zero. reflect_mode is enum XmfMagReflect (wall=0 /
   symmetric=1, the M3 reflect tier). A later call replaces the previous
   magnetic state. Returns 0, or -1 on invalid arguments (NULL run or
   block pointers, unknown reflect mode) or device-mirror failure; the
   gate stays as it was then. Until this port succeeds every magnetic
   helper refuses (-1) and the fluid path is bit for bit the pre-M one.

   xmf_mag_pack / xmf_mag_unpack move one whole B block (ghosts
   included) into / out of the kernel, component comp 0/1/2; both return
   the number of doubles moved, or -1 (moving nothing) on invalid
   arguments or a closed gate. xmf_mag_unpack_e reads one edge-centred E
   block out (the M2 read seam -- block 1 keeps E all zero; no E write
   port this slice); same return convention.

   xmf_mag_set_user loads the tangential-B floor for x1 USER faces:
   [tcomp][k*nytot+j] with tcomp 0=B2, 1=B3, code units. Every ghost
   fill of an x1 user face writes it back to the ghost layers of B2/B3
   and reconstructs B1's ghost faces from the discrete divB = 0
   constraint; without a floor the tangential ghosts fall back to copy
   @1. The buffer is cell-row sized (one value per transverse cell row
   of the boundary plane), so a tangential component's TOP face row --
   the face row its own direction has one extra of -- takes the copy @1
   fallback as well. The buffer is copied and owned by the run (freed
   by xmf_finalize). Requires an open gate. Returns 0, or -1 on invalid
   arguments / closed gate. */
XMF_API int xmf_mag_load(XmfRun *run, int reflect_mode, const double *b1,
                         const double *b2, const double *b3);
XMF_API int xmf_mag_pack(XmfRun *run, int comp, const double *in);
XMF_API int xmf_mag_unpack(const XmfRun *run, int comp, double *out);
XMF_API int xmf_mag_unpack_e(const XmfRun *run, int comp, double *out);
XMF_API int xmf_mag_set_user(XmfRun *run, const double *data);

/* Analytic user boundary tapes (A3 supplement), auxiliary LOAD PORT at
   the same helper level as xmf_bc_set_user -- NOT a fifth public verb.
   Tapes are compiled by Python (xmf.bc_ir) into the frozen opcode table
   of xmf_ir.h; the kernel never parses a formula string. ASSIGNMENT
   semantics, unlike the source tapes: each tape writes one fluid slot
   of every ghost cell of the face (the whole transverse plane, all
   ghost layers), evaluated at the ghost's own cell centre; slots that
   carry no tape fall back to copy @1. The optional P_e block is left
   untouched.

   header carries four ints per tape -- dest_var (the field slot 0..4 =
   n,vx,vy,vz,P the expression assigns), dest_fluid (fluid index),
   nops (>= 1), nconst -- then ops holds nops (opcode, argument) pairs
   per tape and consts holds nconst doubles per tape, both concatenated
   in tape order (consts may be NULL when the whole set has zero
   constants). Reads are restricted to the ghost's own grid centre
   (r/x1/x2/x3), baked-in constants and the face's two innermost active
   layers (KIND_NB slots: @1/@2, face-relative); fluid self-reads,
   electron slots and beam fluxes are refused at load, as are layers
   deeper than the face has active cells. Boundary tapes never enter
   run->tapes, xmf_ir_apply, xmf_ir_jac or the implicit source solve.

   The face must already carry XMF_BC_USER. A face's tape set fills at
   every ghost fill (setup IC fill, every step start, xmf_fill_bc).
   Returns ntape, or -1 on invalid arguments / allocation failure (the
   face's previous set stays untouched then); ntape == 0 clears the
   face's set. */
XMF_API int xmf_bc_load_tapes(XmfRun *run, int face, int ntape,
                              const int *header, const int *ops,
                              const double *consts);

/* Largest relative one-step residual of the linearised implicit source
   solve over the last completed step: max |U* + dt*R(U^{n+1}) - U^{n+1}|
   normalised component-wise. Report-only -- the solve is a single
   linearised backward-Euler step and never iterates on it. 0 when no
   source step ran (no tapes loaded). Auxiliary read port, helper level. */
XMF_API double xmf_source_resid(const XmfRun *run);

/* M5: maximum normalised discrete divB over the interior cells. C-IDF-4
   area-weighted form: for each cell c, (div B)_c = sum_d (A_d[f+1] B_d
   [f+1] - A_d[f] B_d[f]) / dV_c, normalised by the largest single
   absolute term in the sum (dimensionless; 0 for a constructively
   divergence-free initial condition). The threshold 1e-6 is enforced in
   the test suite, not here -- this is pure report, never aborts. Returns
   0 when the magnetic gate is closed. Auxiliary read port, helper level
   like xmf_source_resid. */
XMF_API double xmf_divb_max(const XmfRun *run);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
