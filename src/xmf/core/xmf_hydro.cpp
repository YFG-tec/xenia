/* The HD Godunov operator of the ideal-Euler kernel: face states +
   Riemann solvers + direction sweeps + CFL + the whole-step L_f and the
   conservative-state helpers the SSP RK staging in xmf_step calls (A4.5
   split out of xmf_core.c; the public surface stays xmf_core.h).

   C1a: this module is C++ and its sweep loops ride the Kokkos interface.
   Each direction sweep is a pair of MDRange kernels covering every
   interior face / cell of that direction (2D ~nx*ny, 3D ~nx*ny*nz
   simultaneously parallel): a read-only face kernel writes numerical
   fluxes into a device face-flux View, then a cell kernel applies its
   own two buffered fluxes with the unchanged dt*A/dV coefficients and
   the unchanged per-cell order (left face first, right face second).
   CFL stays a serial per-cell max; the SSP staging helpers stay
   whole-block loops.

   C2: the execution space of every loop above is the DEFAULT execution
   space of the build -- the CUDA device in the WSL CUDA tree, the
   OpenMP host in the Windows tree. The face/cell kernels, the B1/B2
   passes, the heat deposit and the three conservative-state helpers are
   KOKKOS_INLINE functions over the HydroP value bundle (device pointers
   + scalars), and every arithmetic expression is the verbatim C1a one.
   The sweep buffers are face-flux Views in the default MEMORY SPACE
   (device memory in the CUDA tree, HostSpace in the host trees), scoped
   to each sweep; no line workspaces remain. The stage buffers U0/Us and
   the heat scratch are device allocations handed through as opaque void*
   by the pure-C xmf_step (xmf_dev_dvec / xmf_dev_dfree). CFL
   (compute_dt) dispatches on run->dev (Port C): with a device attached
   the reduction runs as a parallel_reduce (Max) over the device Views
   (one formula, two paths -- the per-cell expression is verbatim the
   host one, B face-averaged to cell center before the fast-magnetosonic
   formula); without a device the serial HOST loop runs over the host
   mirror. The ghost refill keeps the locked sync frequency: one
   xmf_fill_bc per ACTIVE direction (RK3 in 1D is three refills per
   step) -- never "once per step".

   Hydro (A4): with empty sources the step is the hyperbolic operator L_f,
   a direction-split Godunov scheme. Face left/right states are rebuilt
   before every Riemann solve, dispatched by the XmfConfig.recon integer
   (enum XmfRecon: pcm=0, plm=1, weno3=2): pcm keeps the adjacent cell
   centres (first order, bit for bit the original path); plm is van
   Leer's limited linear reconstruction and weno3 the Jiang-Shu
   three-point finite-volume reconstruction, both applied component-wise
   to the five primitive variables (never characteristic space) on the
   uniform cartesian mesh. plm/weno3 stencils reach i-1..i+2, so
   xmf_setup refuses them when ng < 2; an out-of-range recon value fails
   setup -- nothing is ever silently run as PCM. One call of L_f sweeps
   every active direction in
   turn; the SSP time integrator (XmfConfig.rk) wraps that WHOLE step
   operator, never a single direction: rk1 is one forward Euler
   U^{n+1} = U^n + dt*L(U^n); rk2 is Heun; rk3 is Shu-Osher. All SSP
   convex combinations are taken on the conservative state U and only then
   converted back to primitives. rk4 is enumerated but not implemented;
   xmf_setup refuses it (no silent fallback to rk1). dt is computed once
   per full step from U^n and shared by every stage; t advances once, at
   the end of a successful step. Every fluid is swept separately along
   every active direction; a sweep writes only that fluid's own five slots.
   Active directions are those with n_d > 1: x1 always (setup enforces
   nx >= 8), x2 when ny > 1, x3 when nz > 1. Ghosts are refilled (fill_bc)
   before each direction's sweep. Three Riemann solvers are dispatched by
   the XmfConfig.riemann integer (enum XmfRiemann: hll=0, hllc=1,
   tvdlf=2, hlld=3); all carry the full interface state -- the
   direction's normal velocity plus both tangential velocities -- and
   tangential momentum rides the mass flux only. This is the HD branch:
   gate-closed runs and gate-open neutrals stay on it. M4 added the ion
   MHD branch below (8-component states fed P_eff and B_s = sqrt(w_s) B,
   hll / hlld / tvdlf, Miyoshi-Kusano HLLD): with the magnetic gate
   open every charge > 0 fluid rides it, and the zero-field limit of the
   MHD HLL stays bit-for-bit the HD HLL (the locked zero-B regression).

   B1 electron closure (pressure-only, share-by-charge): electrons are NOT
   a fifth momentum fluid. When the closure gate is on (the optional P_e
   block exists AND xmf_charge_load has delivered at least one non-zero
   q_s), every ion's sweep in all three directions sees
   P_eff = P_s + w_s P_e with the charge share w_s = q_s n_s / n_e and the
   quasi-neutral density n_e = sum_ion q_s n_s: one pass sweeps
   (rho, v, P_eff) and keeps rho and the momenta, a second pass sweeps
   (rho*P_s/P_eff, v, P_s) and keeps only the pressure -- the ion internal
   energy never rides P_e. P_e itself is advanced afterwards by one
   pseudo-fluid sweep built from n_e and the charge-weighted velocity
   u_e = sum_ion q_s n_s u_s / n_e, keeping only its pressure increment.
   The CFL ion sound speed reads P_eff. Cells with n_e <= 0 fall back to
   w_s = 0, u_e = 0 and P_eff = P_s. Neutrals always carry w_s = 0. With
   the gate off (electrons off, or no charges loaded) none of this runs
   and the sweep below is bit-for-bit the pre-B1 path -- P_eff = P and the
   P_e block untouched.

   B2 shock-heating share (enum XmfShockHeat, stored in the flat blob's
   former reserved0 slot -- byte position frozen, name moved): the Godunov
   sweep's NON-ADIABATIC pressure increment -- the entropy-producing part
   H = max(P_B - P_s^n (rho'_1/rho'_0)^gamma, 0) of pass B, measured with
   the pass's own scaled density rho' -- is split between P_s and P_e per
   tier: ions-only (f_s = 0) keeps every increment in P_s and is bit-for-
   bit the B1 two-pass path; by-charge (f_s = w_s) hands the charge share
   w_s of each ion's increment to P_e; electron (f_s = 1) hands the whole
   increment to P_e. Tóth's warning is the contract: T_e/T_i is NOT unique
   at a shock, so none of the three tiers is physical truth -- they are
   switchable model options. The removed share lands on P_e AFTER the same
   direction's pseudo-fluid sweep (so the pseudo sweep stays tier-blind and
   the reversible PdV part of P_e is untouched by the split), only at cells
   that carry an ion above the density floor. With the closure gate off or
   the tier ions-only the scratch buffer stays NULL and every loop below is
   the B1 one verbatim.

   CFL (A1 stretch supplement): dt = cfl / max_cell sum_d (s_{d,cell} /
   dx_{d,cell}), the sum running over active directions only (n_d > 1) and
   the max over every interior cell; s_{d,cell} = max_fluid (|v_d| + c_s)
   at that cell. B3 hard gate: with the electron closure on and two or
   more charged fluids loaded, the P_e pseudo-sweep's own |u_{e,d}| + c_e
   (assembled exactly as in helec_state_at) joins that per-cell max.
   A degenerate direction (n_d == 1) contributes nothing;
   on a uniform grid this reduces bit for bit to the classic
   cfl / sum_d (s_d / dx_d) and 1D exactly to cfl*dx/s. dt_max (when > 0)
   caps dt from above. Cell widths come from the per-cell mesh dx arrays --
   never from a scalar spacing. The flux update of every sweep is the
   area-weighted finite-volume form
       U_i <- U_i - (dt/dV_i) (A_{i+1/2} F_{i+1/2} - A_{i-1/2} F_{i-1/2})
   in every direction and geometry (cartesian A = 1, dV = dx, so a uniform
   cartesian grid stays bit-for-bit the old dt/dx update). */
#include "xmf_run.h"

#include <math.h>

#include "xmf_device.h"
#include "xmf_kokkos.h"

/* ---- HD Godunov operator L_f (A4) --------------------------------------
   Five-component ideal Euler per fluid, swept per direction. Primitive
   state q = (rho, vn, vt1, vt2, P): the swept direction's normal velocity
   plus both tangential velocities; tangential momentum is carried by the
   mass flux only (no tangential pressure forces). Conservative state
   U = (rho, rho*vn, rho*vt1, rho*vt2, E), E = P/(gamma-1) +
   0.5*rho*(vn^2+vt1^2+vt2^2). With the B1 electron closure gate OFF the
   Riemann pressure is P alone (P_eff = P) and the step never touches the
   optional P_e block (A2); with the gate on, ions are swept with
   P_eff = P_s + w_s P_e and P_e is advanced by its own pseudo-fluid
   pressure-increment sweep (see the B1 block below).

   Face left/right states are built just before each Riemann solve,
   dispatched by run->recon (enum XmfRecon): pcm keeps the two adjacent
   cell centres (first order, unchanged bit for bit); plm interpolates
   each component to the face with the van Leer limited slope; weno3 is
   the Jiang-Shu three-point finite-volume reconstruction, eps = 1e-6.
   Every one of the five primitive components is reconstructed on its own
   -- no characteristic decomposition -- and only for uniform cartesian
   spacings (slopes are centre-to-centre differences). The optional P_e
   block never enters reconstruction. Time integration is the SSP RK
   family (rk1/rk2 Heun/rk3 Shu-Osher) wrapped around the whole-step
   operator L_f below -- one stage always sweeps every active direction
   before the next stage starts; see xmf_step. */

/* ---- C2 device bundle ---------------------------------------------------
   The line kernels run on the default execution space, so they read the
   device Views through this POD value bundle (device pointers + scalars)
   instead of the host XmfRun. Built host-side by hydro_p() from the run's
   DeviceState mirror pair; copied into every kernel by value. All pointer
   members point into the DEFAULT memory space. */
struct HydroP {
  double *fields;      /* device mirror of run->fields */
  double *pe;          /* device mirror of run->pe; NULL when electrons off */
  double *bf[3];       /* M4: device face-centred B; bf[0] NULL = the
                          magnetic gate is closed (no MHD sweep then) */
  const double *ne_stage;  /* M4: stage-start n_e snapshot the ion sweeps
                              freeze their charge shares w_s against; NULL
                              while the magnetic gate is closed */
  const double *charge;    /* [nfluid] device table */
  const double *mass_code; /* [nfluid] device table */
  const double *A;         /* swept direction's face areas, length ltot+1 */
  const double *dV;        /* swept direction's volume factors, length ltot */
  double *heat;            /* B2 scratch (device); NULL = ions-only tier */
  int nfluid, ng;
  int nline;  /* active cells along the swept direction */
  int ltot;   /* cells along the swept direction, ghosts included */
  int recon, riemann, shock_heat;
  int nxtot, nytot, nztot; /* for the device cidx / staggered mag index */
  size_t ncells;
  size_t stride;         /* cell-index stride of the swept direction */
  double gamma, gmm1, dt;
};

/* Linear cell index on the device bundle (the xmf_run.h cidx layout). */
KOKKOS_INLINE_FUNCTION
size_t hcidx(const HydroP &h, int k, int j, int i) {
  return (((size_t)k * (size_t)h.nytot) + (size_t)j) * (size_t)h.nxtot +
         (size_t)i;
}

/* Pointer to fluid f's slot var on the device field block. */
KOKKOS_INLINE_FUNCTION
double *hslot(const HydroP &h, int f, int var) {
  return h.fields +
         (((size_t)f * XMF_SLOT_NVAR + (size_t)var) * h.ncells);
}

/* The direction sweep's slot permutation (the frozen XMF_SWEEP_MAP
   numbers, as a device-callable lookup):
       x1: (vn,vt1,vt2) = (vx,vy,vz)  ->  {0,1,2,3,4}
       x2: (vn,vt1,vt2) = (vy,vx,vz)  ->  {0,2,1,3,4}
       x3: (vn,vt1,vt2) = (vz,vx,vy)  ->  {0,3,1,2,4}  */
KOKKOS_INLINE_FUNCTION
int sweep_map_v(int dir, int v) {
  if (v == 0 || v == 4)
    return v;
  if (dir == 0)
    return v;
  if (dir == 1)
    return (v == 1) ? 2 : (v == 2) ? 1 : 3;
  return (v == 1) ? 3 : (v == 2) ? 1 : 2;
}

/* Host-side bundle builder: pointers from the run's DeviceState, scalars
   from the run itself. The caller fills nline/ltot/stride for its
   direction. heat is the opaque device scratch (or NULL). */
static HydroP hydro_p(const XmfRun *run, int dir, double dt, void *heat) {
  xmf::DeviceState *dv = xmf::dev(run);
  HydroP h;
  int d;
  h.fields = dv->fields.data();
  h.pe = run->pe ? dv->pe.data() : NULL;
  /* M4: the three face-centred B Views; a default-constructed (gate
     closed) View carries data() == NULL, which is the device-side gate
     test the sweep dispatch relies on. */
  for (d = 0; d < 3; d++)
    h.bf[d] = dv->bf[d].data();
  h.ne_stage = dv->ne_stage.data(); /* NULL while the gate is closed */
  h.charge = dv->charge.data();
  h.mass_code = dv->mass_code.data();
  h.A = dv->A[dir].data();
  h.dV = dv->dV[dir].data();
  h.heat = (double *)heat;
  h.nfluid = run->nfluid;
  h.ng = run->ng;
  h.nline = 0;
  h.ltot = 0;
  h.recon = run->recon;
  h.riemann = run->riemann;
  h.shock_heat = run->shock_heat;
  h.nxtot = run->nxtot;
  h.nytot = run->nytot;
  h.nztot = run->nztot;
  h.ncells = run->ncells;
  h.stride = 0;
  h.gamma = run->gamma;
  h.gmm1 = run->gamma - 1.0;
  h.dt = dt;
  return h;
}

KOKKOS_INLINE_FUNCTION
static void prim5_to_cons(const double q[5], double gmm1, double U[5]) {
  double rho = q[0], vn = q[1], vt1 = q[2], vt2 = q[3], p = q[4];
  U[0] = rho;
  U[1] = rho * vn;
  U[2] = rho * vt1;
  U[3] = rho * vt2;
  U[4] = p / gmm1 + 0.5 * rho * (vn * vn + vt1 * vt1 + vt2 * vt2);
}

KOKKOS_INLINE_FUNCTION
static void cons5_to_prim(const double U[5], double gmm1, double q[5]) {
  double r = U[0] > 1e-30 ? U[0] : 1e-30;
  double ek;
  q[0] = r;
  q[1] = U[1] / r;
  q[2] = U[2] / r;
  q[3] = U[3] / r;
  ek = 0.5 * (U[1] * U[1] + U[2] * U[2] + U[3] * U[3]) / r;
  q[4] = gmm1 * (U[4] - ek);
  if (q[4] < 1e-30)
    q[4] = 1e-30;
}

/* Physical flux of one primitive state along the swept direction. */
KOKKOS_INLINE_FUNCTION
static void flux5(const double q[5], double gmm1, double F[5]) {
  double rho = q[0], vn = q[1], p = q[4];
  double E = p / gmm1 + 0.5 * rho * (vn * vn + q[2] * q[2] + q[3] * q[3]);
  F[0] = rho * vn;
  F[1] = rho * vn * vn + p;
  F[2] = rho * vn * q[2];
  F[3] = rho * vn * q[3];
  F[4] = (E + p) * vn;
}

/* Outermost signal speeds of the Riemann fan. Shared Davis estimate for
   HLL and HLLC: SL = min(vnL - cL, vnR - cR), SR = max(vnL + cL, vnR + cR).
   Rho and pressure are floored so an idle (all-zero) extra fluid produces
   defined numbers instead of NaN. */
KOKKOS_INLINE_FUNCTION
static void wavespeeds(const double ql[5], const double qr[5], double gamma,
                       double *SL, double *SR) {
  double rl = ql[0] > 1e-30 ? ql[0] : 1e-30;
  double rr = qr[0] > 1e-30 ? qr[0] : 1e-30;
  double pl = ql[4] > 1e-30 ? ql[4] : 1e-30;
  double pr = qr[4] > 1e-30 ? qr[4] : 1e-30;
  double cl = sqrt(gamma * pl / rl);
  double cr = sqrt(gamma * pr / rr);
  double vl = ql[1], vr = qr[1];
  *SL = vl - cl;
  if (vr - cr < *SL)
    *SL = vr - cr;
  *SR = vr + cr;
  if (vl + cl > *SR)
    *SR = vl + cl;
}

/* ---- M4 ion MHD Riemann (magnetic gate open) ----------------------------
   The 8-component ideal-MHD operator the gate-open ion sweeps ride.
   State q8 = (rho, vn, vt1, vt2, P, Bn, Bt1, Bt2): the swept direction's
   normal velocity and magnetic component first, both tangentials after.
   Conservative state U8 = (rho, rho vn, rho vt1, rho vt2, E, Bn, Bt1,
   Bt2) with E = P/(gamma-1) + rho v^2/2 + |B|^2/2 -- the ion sweep feeds
   the per-ion pressure P_eff = P_s + w_s P_e and the per-ion field
   B_s = sqrt(w_s) B, so each ion's energy carries exactly ITS magnetic
   share |B_s|^2/2 = w_s |B|^2/2 (one share per ion -- never N copies of
   |B|^2/2 summed). Three solvers share the Davis outer-wave estimate
   with the fast-magnetosonic speed; hlld is the Miyoshi-Kusano (2005)
   four-state five-wave solver (transcribed against the read-only
   PLUTO/SSH reference, never a paste). The module's induction-equation
   flux rows are NOT used: only the five fluid rows of the numerical flux
   are returned and scattered into the fluid five slots; B advances by
   the one-shot finite-volume induction in xmf_mag.c, never here. */

/* Staggered face index of bf[comp] on the device bundle (the xmf_run.h
   layout): i0/i1/i2 carry the full coordinate triple; the comp entry is
   the FACE index along the component's own direction, the other two are
   CELL indices. */
KOKKOS_INLINE_FUNCTION
static size_t hbidx(const HydroP &h, int comp, int i0, int i1, int i2) {
  int c[3];
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  int n1 = (d1 == 0) ? h.nxtot : (d1 == 1) ? h.nytot : h.nztot;
  int nd = (comp == 0) ? h.nxtot : (comp == 1) ? h.nytot : h.nztot;
  c[0] = i0;
  c[1] = i1;
  c[2] = i2;
  return (((size_t)c[d2] * (size_t)n1) + (size_t)c[d1]) * (size_t)(nd + 1) +
         (size_t)c[comp];
}

/* Cell-centred B: the average of the two face values along each
   direction (the card's B_d^cell = (B_d^face + B_d^{face+1})/2). The
   face values are then reconstructed to the interface with the very
   same pcm/plm/weno3 the fluid slots ride -- the raw face-centred
   normal B is never used directly as the Riemann B_n (that is M5/CT). */
KOKKOS_INLINE_FUNCTION
static void hcell_b_center(const HydroP &h, int k, int j, int i,
                           double bc[3]) {
  bc[0] = 0.5 * (h.bf[0][hbidx(h, 0, i, j, k)] +
                 h.bf[0][hbidx(h, 0, i + 1, j, k)]);
  bc[1] = 0.5 * (h.bf[1][hbidx(h, 1, i, j, k)] +
                 h.bf[1][hbidx(h, 1, i, j + 1, k)]);
  bc[2] = 0.5 * (h.bf[2][hbidx(h, 2, i, j, k)] +
                 h.bf[2][hbidx(h, 2, i, j, k + 1)]);
}

KOKKOS_INLINE_FUNCTION
static void mhd_prim_to_cons(const double q[8], double gmm1, double U[8]) {
  double rho = q[0], vn = q[1], vt1 = q[2], vt2 = q[3], p = q[4];
  U[0] = rho;
  U[1] = rho * vn;
  U[2] = rho * vt1;
  U[3] = rho * vt2;
  U[4] = p / gmm1 + 0.5 * rho * (vn * vn + vt1 * vt1 + vt2 * vt2) +
         0.5 * (q[5] * q[5] + q[6] * q[6] + q[7] * q[7]);
  U[5] = q[5];
  U[6] = q[6];
  U[7] = q[7];
}

/* Physical flux of one MHD primitive state along the swept direction. */
KOKKOS_INLINE_FUNCTION
static void mhd_flux_phys(const double q[8], double gmm1, double F[8]) {
  double rho = q[0], vn = q[1], vt1 = q[2], vt2 = q[3], p = q[4];
  double bn = q[5], bt1 = q[6], bt2 = q[7];
  double b2 = bn * bn + bt1 * bt1 + bt2 * bt2;
  double pt = p + 0.5 * b2;
  double E = p / gmm1 + 0.5 * rho * (vn * vn + vt1 * vt1 + vt2 * vt2) +
             0.5 * b2;
  double vdb = vn * bn + vt1 * bt1 + vt2 * bt2;
  F[0] = rho * vn;
  F[1] = rho * vn * vn + pt - bn * bn;
  F[2] = rho * vn * vt1 - bn * bt1;
  F[3] = rho * vn * vt2 - bn * bt2;
  F[4] = (E + pt) * vn - bn * vdb;
  F[5] = 0.0; /* induction rows: computed for completeness, never
                 scattered -- the one-shot induction owns B */
  F[6] = bt1 * vn - bn * vt1;
  F[7] = bt2 * vn - bn * vt2;
}

/* Fast-magnetosonic speed squared of the Davis estimate:
   cf^2 = 1/2 (a^2 + b^2 + sqrt((a^2+b^2)^2 - 4 a^2 bn^2)) with
   a^2 = gamma P/rho, b^2 = |B|^2/rho, bn^2 = B_n^2/rho. b2 and bn2
   arrive RAW (energy-density units, |B|^2) and are normalised by rho
   here -- every caller feeds the same convention. The zero-field
   branch returns a^2 as-is, so a B = 0 state stays bit-for-bit the HD
   wavespeeds' sqrt(gamma*P/rho) (the locked zero-B regression). */
KOKKOS_INLINE_FUNCTION
static double mhd_cf2(double gamma, double rho, double p, double b2,
                      double bn2) {
  double a2 = gamma * p / rho;
  double s, disc;
  if (b2 <= 0.0)
    return a2;
  b2 /= rho;
  bn2 /= rho;
  s = a2 + b2;
  disc = s * s - 4.0 * a2 * bn2;
  if (disc < 0.0)
    disc = 0.0;
  return 0.5 * (s + sqrt(disc));
}

/* Outermost signal speeds of the MHD Riemann fan (Davis estimate with
   the fast-magnetosonic speed). Rho/pressure floors exactly as the HD
   wavespeeds above. */
KOKKOS_INLINE_FUNCTION
static void mhd_wavespeeds(const double ql[8], const double qr[8],
                           double gamma, double *SL, double *SR) {
  double rl = ql[0] > 1e-30 ? ql[0] : 1e-30;
  double rr = qr[0] > 1e-30 ? qr[0] : 1e-30;
  double pl = ql[4] > 1e-30 ? ql[4] : 1e-30;
  double pr = qr[4] > 1e-30 ? qr[4] : 1e-30;
  double bl2 = ql[5] * ql[5] + ql[6] * ql[6] + ql[7] * ql[7];
  double br2 = qr[5] * qr[5] + qr[6] * qr[6] + qr[7] * qr[7];
  double cl = sqrt(mhd_cf2(gamma, rl, pl, bl2, ql[5] * ql[5]));
  double cr = sqrt(mhd_cf2(gamma, rr, pr, br2, qr[5] * qr[5]));
  double vl = ql[1], vr = qr[1];
  *SL = vl - cl;
  if (vr - cr < *SL)
    *SL = vr - cr;
  *SR = vr + cr;
  if (vl + cl > *SR)
    *SR = vl + cl;
}

/* MHD numerical flux through one face. solver is one of hll / hlld /
   tvdlf (the three gate-open solvers; hllc is refused by the gate and
   never arrives). Only the five fluid rows are written to F. */
KOKKOS_INLINE_FUNCTION
static void mhd_riemann_flux(int solver, double gamma, const double ql[8],
                             const double qr[8], double F[5]) {
  double gmm1 = gamma - 1.0;
  double UL[8], UR[8], FL[8], FR[8];
  double SL, SR;
  int c;
  mhd_prim_to_cons(ql, gmm1, UL);
  mhd_prim_to_cons(qr, gmm1, UR);
  mhd_flux_phys(ql, gmm1, FL);
  mhd_flux_phys(qr, gmm1, FR);

  if (solver == XMF_RIEMANN_TVDLF) {
    /* Local Lax-Friedrichs / Rusanov: the local speed bound is the
       larger of the two one-sided |vn| + cf estimates. */
    double rl = ql[0] > 1e-30 ? ql[0] : 1e-30;
    double rr = qr[0] > 1e-30 ? qr[0] : 1e-30;
    double pl = ql[4] > 1e-30 ? ql[4] : 1e-30;
    double pr = qr[4] > 1e-30 ? qr[4] : 1e-30;
    double bl2 = ql[5] * ql[5] + ql[6] * ql[6] + ql[7] * ql[7];
    double br2 = qr[5] * qr[5] + qr[6] * qr[6] + qr[7] * qr[7];
    double cl = sqrt(mhd_cf2(gamma, rl, pl, bl2, ql[5] * ql[5]));
    double cr = sqrt(mhd_cf2(gamma, rr, pr, br2, qr[5] * qr[5]));
    double s = fabs(ql[1]) + cl;
    double t = fabs(qr[1]) + cr;
    if (t > s)
      s = t;
    for (c = 0; c < 5; c++)
      F[c] = 0.5 * (FL[c] + FR[c] - s * (UR[c] - UL[c]));
    return;
  }

  mhd_wavespeeds(ql, qr, gamma, &SL, &SR);
  if (SL >= 0.0) {
    for (c = 0; c < 5; c++)
      F[c] = FL[c];
    return;
  }
  if (SR <= 0.0) {
    for (c = 0; c < 5; c++)
      F[c] = FR[c];
    return;
  }

  if (solver == XMF_RIEMANN_HLLD) {
    /* Miyoshi-Kusano (2005) four-state five-wave HLLD. Star states:
       the single-star pair (L-star / R-star) between SL, SM and the
       Alfven speeds S1L, S1R; the double-star pair across the Alfven
       wings; the contact wave SM and the total pressure pts come from
       the jump relations. Degenerate fans (an Alfven speed collapsing
       against SL or SR) revert to the HLL middle state with
       S1L = S1R = SM -- the reference implementation's revert_to_hllc,
       never a different solver. */
    double den = SR - SL;
    double Bx = (SR * qr[5] - SL * ql[5]) / den; /* star-region normal B */
    double ptL =
        ql[4] + 0.5 * (ql[5] * ql[5] + ql[6] * ql[6] + ql[7] * ql[7]);
    double ptR =
        qr[4] + 0.5 * (qr[5] * qr[5] + qr[6] * qr[6] + qr[7] * qr[7]);
    double duL = SL - ql[1];
    double duR = SR - qr[1];
    double scrh = duR * UR[0] - duL * UL[0];
    double SM = (duR * UR[1] - duL * UL[1] - ptR + ptL) / scrh;
    double pts = (duR * UR[0] * ptL - duL * UL[0] * ptR +
                  UL[0] * UR[0] * duR * duL * (qr[1] - ql[1])) /
                 scrh;
    double usL0 = UL[0] * duL / (SL - SM);
    double usR0 = UR[0] * duR / (SR - SM);
    double sqrL = sqrt(usL0 > 0.0 ? usL0 : 0.0);
    double sqrR = sqrt(usR0 > 0.0 ? usR0 : 0.0);
    double S1L = SM - fabs(Bx) / sqrL;
    double S1R = SM + fabs(Bx) / sqrR;
    double usL[8], usR[8], ussl[5], ussr[5];
    double vsL = 0.0, wsL = 0.0, vsR = 0.0, wsR = 0.0;
    int revert;
    usL[5] = Bx;
    usR[5] = Bx;
    revert = ((S1L - SL) < 1.0e-4 * (SM - SL)) ||
             ((S1R - SR) > -1.0e-4 * (SR - SM));
    if (revert) {
      /* Degenerate fan (an Alfven wave collapsed against SL or SR): the
         star / double-star construction degenerates, so return the HLL
         middle flux -- the flux of the very HLL middle state the revert
         falls back to (the reference implementation's revert_to_hllc
         intent: never resolve the Alfven wings there). SL < 0 < SR is
         guaranteed by the two fast paths above, so the two-wave average
         is the whole answer; no star state is assembled at all. */
      for (c = 0; c < 5; c++)
        F[c] = (SR * FL[c] - SL * FR[c] + SL * SR * (UR[c] - UL[c])) / den;
      return;
    }
    {
      double scrhL = (UL[0] * duL * duL - Bx * Bx) /
                     (UL[0] * duL * (SL - SM) - Bx * Bx);
      double scrhR = (UR[0] * duR * duR - Bx * Bx) /
                     (UR[0] * duR * (SR - SM) - Bx * Bx);
      double kL = Bx / (UL[0] * duL);
      double kR = Bx / (UR[0] * duR);
      double eL, eR;
      usL[6] = UL[6] * scrhL;
      usL[7] = UL[7] * scrhL;
      usR[6] = UR[6] * scrhR;
      usR[7] = UR[7] * scrhR;
      vsL = ql[2] - kL * (usL[6] - UL[6]);
      wsL = ql[3] - kL * (usL[7] - UL[7]);
      vsR = qr[2] - kR * (usR[6] - UR[6]);
      wsR = qr[3] - kR * (usR[7] - UR[7]);
      usL[0] = usL0;
      usL[1] = usL0 * SM;
      usL[2] = usL0 * vsL;
      usL[3] = usL0 * wsL;
      usR[0] = usR0;
      usR[1] = usR0 * SM;
      usR[2] = usR0 * vsR;
      usR[3] = usR0 * wsR;
      eL = (ql[1] * Bx + ql[2] * UL[6] + ql[3] * UL[7]) -
           (SM * Bx + vsL * usL[6] + wsL * usL[7]);
      eR = (qr[1] * Bx + qr[2] * UR[6] + qr[3] * UR[7]) -
           (SM * Bx + vsR * usR[6] + wsR * usR[7]);
      usL[4] = (duL * UL[4] - ptL * ql[1] + pts * SM + Bx * eL) / (SL - SM);
      usR[4] = (duR * UR[4] - ptR * qr[1] + pts * SM + Bx * eR) / (SR - SM);
    }
    if (S1L >= 0.0) {
      /* Left star region: F = FL + SL (U*L - UL). */
      for (c = 0; c < 5; c++)
        F[c] = FL[c] + SL * (usL[c] - UL[c]);
    } else if (S1R <= 0.0) {
      /* Right star region: F = FR + SR (U*R - UR). */
      for (c = 0; c < 5; c++)
        F[c] = FR[c] + SR * (usR[c] - UR[c]);
    } else {
      /* Between the Alfven waves: the double-star states (the contact
         and the normal-field continuity fix the shared tangential B). */
      double sBx = Bx > 0.0 ? 1.0 : -1.0;
      double denom = sqrL + sqrR;
      double vss =
          (sqrL * vsL + sqrR * vsR + (usR[6] - usL[6]) * sBx) / denom;
      double wss =
          (sqrL * wsL + sqrR * wsR + (usR[7] - usL[7]) * sBx) / denom;
      double Bs1 = (sqrL * usR[6] + sqrR * usL[6] +
                    sqrL * sqrR * (vsR - vsL) * sBx) /
                   denom;
      double Bs2 = (sqrL * usR[7] + sqrR * usL[7] +
                    sqrL * sqrR * (wsR - wsL) * sBx) /
                   denom;
      double eL, eR;
      ussl[0] = usL[0];
      ussl[1] = ussl[0] * SM;
      ussl[2] = ussl[0] * vss;
      ussl[3] = ussl[0] * wss;
      ussr[0] = usR[0];
      ussr[1] = ussr[0] * SM;
      ussr[2] = ussr[0] * vss;
      ussr[3] = ussr[0] * wss;
      eL = (SM * Bx + vsL * usL[6] + wsL * usL[7]) -
           (SM * Bx + vss * Bs1 + wss * Bs2);
      eR = (SM * Bx + vsR * usR[6] + wsR * usR[7]) -
           (SM * Bx + vss * Bs1 + wss * Bs2);
      ussl[4] = usL[4] - sqrL * eL * sBx;
      ussr[4] = usR[4] + sqrR * eR * sBx;
      if (SM >= 0.0) {
        for (c = 0; c < 5; c++)
          F[c] = FL[c] + S1L * (ussl[c] - usL[c]) + SL * (usL[c] - UL[c]);
      } else {
        for (c = 0; c < 5; c++)
          F[c] = FR[c] + S1R * (ussr[c] - usR[c]) + SR * (usR[c] - UR[c]);
      }
    }
    return;
  }

  /* Default: MHD HLL. Two-wave average applied component-wise to the
     full eight-component vector; the five fluid rows are returned. */
  {
    double den = SR - SL;
    for (c = 0; c < 5; c++)
      F[c] = (SR * FL[c] - SL * FR[c] + SL * SR * (UR[c] - UL[c])) / den;
  }
}

/* Numerical flux through one face. solver is the XmfConfig.riemann
   integer (enum XmfRiemann: hll=0, hllc=1, tvdlf=2, hlld=3). C2: the
   former memcpy(F, FL/FR, ...) fast paths are explicit five-element
   loops -- memcpy is not device-callable and an element copy is the
   same bits. */
KOKKOS_INLINE_FUNCTION
static void riemann_flux(int solver, double gamma, const double ql[5],
                         const double qr[5], double F[5]) {
  double gmm1 = gamma - 1.0;
  double UL[5], UR[5], FL[5], FR[5];
  double SL, SR;
  int c;
  if (solver == XMF_RIEMANN_HLLD) {
    /* M4: hlld is the MHD solver. Gate-closed runs and gate-open
       neutrals reach it ONLY through this zero-field limit: the very
       same 8-component Miyoshi-Kusano solver the gate-open ions ride,
       fed B = 0, five fluid rows returned. No silent fallback to
       another solver -- and with B = 0 the limit is the textbook HD
       HLLD structure (the degeneracy guard inside keeps every state
       finite). */
    double ql8[8], qr8[8];
    int v;
    for (v = 0; v < 5; v++) {
      ql8[v] = ql[v];
      qr8[v] = qr[v];
    }
    ql8[5] = 0.0;
    ql8[6] = 0.0;
    ql8[7] = 0.0;
    qr8[5] = 0.0;
    qr8[6] = 0.0;
    qr8[7] = 0.0;
    mhd_riemann_flux(XMF_RIEMANN_HLLD, gamma, ql8, qr8, F);
    return;
  }
  prim5_to_cons(ql, gmm1, UL);
  prim5_to_cons(qr, gmm1, UR);
  flux5(ql, gmm1, FL);
  flux5(qr, gmm1, FR);

  if (solver == XMF_RIEMANN_TVDLF) {
    /* Local Lax-Friedrichs / Rusanov: the local speed bound is the larger
       of the two one-sided |vn| + c estimates. */
    double rl = ql[0] > 1e-30 ? ql[0] : 1e-30;
    double rr = qr[0] > 1e-30 ? qr[0] : 1e-30;
    double pl = ql[4] > 1e-30 ? ql[4] : 1e-30;
    double pr = qr[4] > 1e-30 ? qr[4] : 1e-30;
    double cl = sqrt(gamma * pl / rl);
    double cr = sqrt(gamma * pr / rr);
    double s = fabs(ql[1]) + cl;
    double t = fabs(qr[1]) + cr;
    if (t > s)
      s = t;
    for (c = 0; c < 5; c++)
      F[c] = 0.5 * (FL[c] + FR[c] - s * (UR[c] - UL[c]));
    return;
  }

  wavespeeds(ql, qr, gamma, &SL, &SR);
  if (SL >= 0.0) {
    for (c = 0; c < 5; c++)
      F[c] = FL[c];
    return;
  }
  if (SR <= 0.0) {
    for (c = 0; c < 5; c++)
      F[c] = FR[c];
    return;
  }

  if (solver == XMF_RIEMANN_HLLC) {
    /* Contact-preserving middle wave. SM from the normal-momentum
       balance; the denominator is rhoL*(SL-vnL) - rhoR*(SR-vnR) < 0
       strictly, so it never vanishes. Star states follow Toro's compact
       form; tangential velocities are unchanged across the middle wave.
       rho and P are floored exactly as in wavespeeds() so that an idle
       (all-zero) extra fluid yields a defined finite flux, never NaN --
       without the floors an all-zero fluid gives SM = 0/0 here. */
    double rl = UL[0] > 1e-30 ? UL[0] : 1e-30;
    double rr = UR[0] > 1e-30 ? UR[0] : 1e-30;
    double vl = ql[1], vr = qr[1];
    double pl = ql[4] > 1e-30 ? ql[4] : 1e-30;
    double pr = qr[4] > 1e-30 ? qr[4] : 1e-30;
    double SM = (pr - pl + rl * vl * (SL - vl) - rr * vr * (SR - vr)) /
                (rl * (SL - vl) - rr * (SR - vr));
    double Us[5];
    /* Denominators are sign-definite here: qk = S_K - v_K satisfies
       qk_L = SL - vl <= -c_l < 0 and qk_R = SR - vr >= c_r > 0 (pressure
       floors keep the sound speeds positive); in the SM >= 0 branch
       SL - SM < 0, in the SM < 0 branch SR - SM > 0. No zero guards
       needed -- and a sign-flipping floor here would be wrong. */
    if (SM >= 0.0) { /* left star region: F = FL + SL*(U*L - UL) */
      double qk = SL - vl;
      double rstar = rl * qk / (SL - SM);
      Us[0] = rstar;
      Us[1] = rstar * SM;
      Us[2] = rstar * ql[2];
      Us[3] = rstar * ql[3];
      Us[4] = rstar * (UL[4] / rl + (SM - vl) * (SM + pl / (rl * qk)));
      for (c = 0; c < 5; c++)
        F[c] = FL[c] + SL * (Us[c] - UL[c]);
    } else { /* right star region: F = FR + SR*(U*R - UR) */
      double qk = SR - vr;
      double rstar = rr * qk / (SR - SM);
      Us[0] = rstar;
      Us[1] = rstar * SM;
      Us[2] = rstar * qr[2];
      Us[3] = rstar * qr[3];
      Us[4] = rstar * (UR[4] / rr + (SM - vr) * (SM + pr / (rr * qk)));
      for (c = 0; c < 5; c++)
        F[c] = FR[c] + SR * (Us[c] - UR[c]);
    }
    return;
  }

  /* Default: HLL. Two-wave average applied component-wise to the full
     five-component vector; tangentials ride along with the mass flux. */
  {
    double den = SR - SL;
    for (c = 0; c < 5; c++)
      F[c] = (SR * FL[c] - SL * FR[c] + SL * SR * (UR[c] - UL[c])) / den;
  }
}

/* ---- Face left/right states (A4 supplement: LINEAR + WENO3) -----------
   Built per face, per primitive component (never characteristic space),
   uniform cartesian spacings only. The face sits between line cells m and
   m+1; the stencils reach q[m-1]..q[m+2], which is why plm/weno3 require
   ng >= 2 (checked at setup). pcm never enters these helpers -- it keeps
   the raw adjacent centres, bit for bit the original first-order path.

   LINEAR (plm) is van Leer's limited slope:
       Delta_+ q_i = q_{i+1} - q_i,   Delta_- q_i = q_i - q_{i-1},
       Delta q_i = 2 Delta_+ Delta_- / (Delta_+ + Delta_-) when
       Delta_+ Delta_- > 0, else 0;
       q_L = q_m + (1/2) Delta q_m,  q_R = q_{m+1} - (1/2) Delta q_{m+1}.

   WENO3 is the Jiang-Shu three-point finite-volume reconstruction with
   eps = 1e-6. Left state (bias-left stencil, centre i = m):
       beta_0 = (q_i - q_{i-1})^2,        beta_1 = (q_{i+1} - q_i)^2,
       alpha_0 = (1/3)/(eps + beta_0)^2,  alpha_1 = (2/3)/(eps + beta_1)^2,
       omega_k = alpha_k / (alpha_0 + alpha_1),
       q_L = omega_0 (-1/2 q_{i-1} + 3/2 q_i)
           + omega_1 ( 1/2 q_i     + 1/2 q_{i+1}).
   Right state is the mirror image, centre i = m+1, stencil pointing
   toward i+2:
       beta_0 = (q_{i+1} - q_{i+2})^2,    beta_1 = (q_i - q_{i+1})^2,
       alpha_0 = (1/3)/(eps + beta_0)^2,  alpha_1 = (2/3)/(eps + beta_1)^2,
       q_R = omega_0 (-1/2 q_{i+2} + 3/2 q_{i+1})
           + omega_1 ( 1/2 q_{i+1} + 1/2 q_i). */

#define XMF_WENO3_EPS 1.0e-6

/* van Leer limited slope at one cell (uniform mesh). */
KOKKOS_INLINE_FUNCTION
static double plm_slope(double qm1, double q0, double qp1) {
  double dp = qp1 - q0; /* Delta_+ */
  double dm = q0 - qm1; /* Delta_- */
  if (dp * dm > 0.0)
    return 2.0 * dp * dm / (dp + dm);
  return 0.0;
}

/* Face states at the face between centres q0 (cell m) and qp1 (cell m+1);
   qm1 is cell m-1 and qp2 is cell m+2. recon is run->recon; only plm and
   weno3 arrive here (pcm keeps the raw centres in the face kernel). Writes the
   five-component left/right states of the face. */
KOKKOS_INLINE_FUNCTION
static void face_states(int recon, const double *qm1, const double *q0,
                        const double *qp1, const double *qp2, double ql[5],
                        double qr[5]) {
  int v;
  if (recon == XMF_RECON_PLM) {
    for (v = 0; v < 5; v++) {
      double sl = plm_slope(qm1[v], q0[v], qp1[v]); /* slope of cell m */
      double sr = plm_slope(q0[v], qp1[v], qp2[v]); /* slope of cell m+1 */
      ql[v] = q0[v] + 0.5 * sl;
      qr[v] = qp1[v] - 0.5 * sr;
    }
    return;
  }
  /* XMF_RECON_WENO3, Jiang-Shu three-point finite volume. */
  for (v = 0; v < 5; v++) {
    double b0, b1, a0, a1, sum, w0, w1;
    /* Left state: centre i = m, stencil {m-1, m, m+1}. */
    b0 = (q0[v] - qm1[v]) * (q0[v] - qm1[v]);
    b1 = (qp1[v] - q0[v]) * (qp1[v] - q0[v]);
    a0 = (1.0 / 3.0) / ((XMF_WENO3_EPS + b0) * (XMF_WENO3_EPS + b0));
    a1 = (2.0 / 3.0) / ((XMF_WENO3_EPS + b1) * (XMF_WENO3_EPS + b1));
    sum = a0 + a1;
    w0 = a0 / sum;
    w1 = a1 / sum;
    ql[v] = w0 * (-0.5 * qm1[v] + 1.5 * q0[v]) +
            w1 * (0.5 * q0[v] + 0.5 * qp1[v]);
    /* Right state: centre i = m+1, stencil pointing toward i+2,
       i.e. {m, m+1, m+2}. */
    b0 = (qp1[v] - qp2[v]) * (qp1[v] - qp2[v]);
    b1 = (q0[v] - qp1[v]) * (q0[v] - qp1[v]);
    a0 = (1.0 / 3.0) / ((XMF_WENO3_EPS + b0) * (XMF_WENO3_EPS + b0));
    a1 = (2.0 / 3.0) / ((XMF_WENO3_EPS + b1) * (XMF_WENO3_EPS + b1));
    sum = a0 + a1;
    w0 = a0 / sum;
    w1 = a1 / sum;
    qr[v] = w0 * (-0.5 * qp2[v] + 1.5 * qp1[v]) +
            w1 * (0.5 * qp1[v] + 0.5 * q0[v]);
  }
}

/* M4: face left/right states for the 8-component MHD sweep -- the very
   face_states formulas above, component-wise over all eight entries
   (rho, vn, vt1, vt2, P_eff, Bn, Bt1, Bt2). Cell-centred B components
   are reconstructed exactly like the fluid slots (the card's "same
   recon"); only plm and weno3 arrive here (pcm keeps the adjacent
   centres in the MHD face kernel). */
KOKKOS_INLINE_FUNCTION
static void face_states8(int recon, const double *qm1, const double *q0,
                         const double *qp1, const double *qp2, double ql[8],
                         double qr[8]) {
  int v;
  if (recon == XMF_RECON_PLM) {
    for (v = 0; v < 8; v++) {
      double sl = plm_slope(qm1[v], q0[v], qp1[v]); /* slope of cell m */
      double sr = plm_slope(q0[v], qp1[v], qp2[v]); /* slope of cell m+1 */
      ql[v] = q0[v] + 0.5 * sl;
      qr[v] = qp1[v] - 0.5 * sr;
    }
    return;
  }
  /* XMF_RECON_WENO3, Jiang-Shu three-point finite volume. */
  for (v = 0; v < 8; v++) {
    double b0, b1, a0, a1, sum, w0, w1;
    /* Left state: centre i = m, stencil {m-1, m, m+1}. */
    b0 = (q0[v] - qm1[v]) * (q0[v] - qm1[v]);
    b1 = (qp1[v] - q0[v]) * (qp1[v] - q0[v]);
    a0 = (1.0 / 3.0) / ((XMF_WENO3_EPS + b0) * (XMF_WENO3_EPS + b0));
    a1 = (2.0 / 3.0) / ((XMF_WENO3_EPS + b1) * (XMF_WENO3_EPS + b1));
    sum = a0 + a1;
    w0 = a0 / sum;
    w1 = a1 / sum;
    ql[v] = w0 * (-0.5 * qm1[v] + 1.5 * q0[v]) +
            w1 * (0.5 * q0[v] + 0.5 * qp1[v]);
    /* Right state: centre i = m+1, stencil {m, m+1, m+2}. */
    b0 = (qp1[v] - qp2[v]) * (qp1[v] - qp2[v]);
    b1 = (q0[v] - qp1[v]) * (q0[v] - qp1[v]);
    a0 = (1.0 / 3.0) / ((XMF_WENO3_EPS + b0) * (XMF_WENO3_EPS + b0));
    a1 = (2.0 / 3.0) / ((XMF_WENO3_EPS + b1) * (XMF_WENO3_EPS + b1));
    sum = a0 + a1;
    w0 = a0 / sum;
    w1 = a1 / sum;
    qr[v] = w0 * (-0.5 * qp2[v] + 1.5 * qp1[v]) +
            w1 * (0.5 * qp1[v] + 0.5 * q0[v]);
  }
}

/* ---- B1 quasi-neutral electron closure gate -------------------------------
   Electrons are NOT a fifth momentum fluid. When the closure gate is on
   (the optional P_e block exists AND at least one loaded charge q_s is
   non-zero), the ion sweeps see P_eff = P_s + w_s P_e in all three
   directions, P_e is re-integrated with the assembled n_e / u_e keeping
   only its pressure increment, and the CFL ion sound speed reads P_eff.
   Everything below hangs off elec_closure_active(); with the gate off none
   of it runs and the hydro path stays bit-for-bit the pre-B1 one. The
   kernel never recognises a species name: the ion role is q_s != 0.
   elec_closure_active itself stays a host function -- it gates the
   dispatch host-side and reads the charge mirror (immutable after load). */

int elec_closure_active(const XmfRun *run) {
  int f;
  if (!run || !run->pe)
    return 0;
  for (f = 0; f < run->nfluid; f++)
    if (run->charge[f] != 0.0)
      return 1;
  return 0;
}

/* The density floor XMF_RHO_FLOOR lives in xmf_run.h (M4: the induction
   module shares the vacuum gate). */

/* ---- Device-side cell helpers (C2) -------------------------------------
   The B1/B2 device twins of the host cell_ne / cell_w / cell_ionized
   helpers below (the host ones keep compute_dt's serial loop verbatim):
   same tests, same arithmetic, reading the HydroP device bundle. */

KOKKOS_INLINE_FUNCTION
static int hcell_ionized(const HydroP &h, size_t c) {
  int f;
  for (f = 0; f < h.nfluid; f++)
    if (h.charge[f] > 0.0 && hslot(h, f, XMF_SLOT_RHO)[c] > XMF_RHO_FLOOR)
      return 1;
  return 0;
}

KOKKOS_INLINE_FUNCTION
static double hcell_ne(const HydroP &h, size_t c) {
  int f;
  double ne = 0.0;
  for (f = 0; f < h.nfluid; f++)
    ne += h.charge[f] * hslot(h, f, XMF_SLOT_RHO)[c] / h.mass_code[f];
  return ne;
}

KOKKOS_INLINE_FUNCTION
static double hcell_w(const HydroP &h, int f, size_t c, double ne) {
  if (h.charge[f] <= 0.0 || ne <= 0.0 || !hcell_ionized(h, c))
    return 0.0;
  return h.charge[f] * hslot(h, f, XMF_SLOT_RHO)[c] / h.mass_code[f] / ne;
}

/* ---- Direction-split sweep (C2sup: MDRange face + cell kernels) --------
   Each sweep of fluid f along direction dir is a pair of MDRange kernels
   covering every interior face / cell of that direction (2D ~nx*ny, 3D
   ~nx*ny*nz): a read-only face kernel writes the numerical fluxes into a
   device face-flux View [line][face][5] (or [line][face][5] for the five
   fluid rows of the MHD solver), then a cell kernel applies its own two
   buffered fluxes with the unchanged dt*A/dV coefficients and the
   unchanged per-cell order (left face first, right face second), so the
   update stays bit for bit the line-serial one in every dimension. The
   B1 two-pass sweep keeps the old read order via a pristine-state
   snapshot View plus two face-flux Views (pass A fluxes built before the
   pass-A scatter, pass B fluxes after it). The line workspaces
   (XmfLineWs) are gone; every sweep buffer lives in the default memory
   space and dies with the sweep scope. The direction-split order of
   lf_all_dirs and the fill_bc roundtrip frequency are untouched; the 1D
   degenerate run walks the same kernels (t0 = t1 = 1).

   The local primitive order is always q = (rho, vn, vt1, vt2, P) with vn
   the SWEPT direction's normal velocity, so the Riemann solvers stay
   direction-agnostic. The gather/scatter maps below permute the stored
   slots (n, vx, vy, vz, P) accordingly: x1 is the identity, x2 swaps
   vx/vy, x3 rotates vz to the front. Scattering writes slot map[v] from
   q[v], the exact inverse permutation.

   M4: with the magnetic gate open, ion fluids ride the 8-component MHD
   sweep (sweep_dir_mhd) instead of the plain HD or B1 two-pass one. The
   face kernel builds per-cell 8-component states (P_eff = P_s + w_s P_e,
   B_s = sqrt(w_s) B from the device face-centred B mirror) and dispatches
   the MHD Riemann solver (hll / hlld / tvdlf); the cell kernel applies
   the five fluid flux rows, recovers primitives with the per-ion magnetic
   share removed, and scatters. Neutrals keep the plain HD sweep (w_s = 0
   => B_s = 0). The B1 two-pass closure yields to the MHD sweep for ions
   when the gate is open; P_e still advances by its own pseudo-HD sweep. */

/* Face-flux buffer of one direction sweep: [transverse line][face][5],
   device memory (default memory space), allocated per sweep. */
typedef Kokkos::View<double ***, Kokkos::LayoutRight> XmfFaceWs;

/* The face kernel's transverse base index (the old line decomposition,
   line = a*t0 + b, kept verbatim: a outer transverse, b inner). */
KOKKOS_INLINE_FUNCTION
static size_t hline_base(const HydroP &h, int dir, int ng, int a, int b) {
  if (dir == 0)
    return hcidx(h, ng + a, ng + b, 0);
  if (dir == 1)
    return hcidx(h, ng + a, 0, ng + b);
  return hcidx(h, 0, ng + a, ng + b);
}

/* Gather the five primitives (rho, vn, vt1, vt2, P) of fluid f at the
   line cell base + ml*stride straight from the device field block. */
KOKKOS_INLINE_FUNCTION
static void hprim_at(const HydroP &h, int f, int dir, size_t base, int ml,
                     double *q) {
  const double *slot0 = h.fields + ((size_t)f * XMF_SLOT_NVAR) * h.ncells;
  const size_t c = base + (size_t)ml * h.stride;
  int v;
  for (v = 0; v < XMF_SLOT_NVAR; v++)
    q[v] = slot0[(size_t)sweep_map_v(dir, v) * h.ncells + c];
}

/* M4: build the 8-component MHD state of fluid f at one cell (ml along
   the line from base). q8 = (rho, vn, vt1, vt2, P_eff, Bn, Bt1, Bt2)
   with P_eff = P_s + w_s P_e (electrons off -> P_s) and B_s = sqrt(w_s)
   B_cell_centre. The charge share w_s is clamped to [0,1] and read from
   the stage-start n_e snapshot when available (ne_stage), or from the
   live fields otherwise. The cell-centred B is the average of the two
   face values along each direction (hcell_b_center). */
KOKKOS_INLINE_FUNCTION
static void hmhd_state_at(const HydroP &h, int f, int dir, size_t base,
                          int ml, double *q8) {
  const double *slot0 = h.fields + ((size_t)f * XMF_SLOT_NVAR) * h.ncells;
  const size_t c = base + (size_t)ml * h.stride;
  int v;
  for (v = 0; v < XMF_SLOT_NVAR; v++)
    q8[v] = slot0[(size_t)sweep_map_v(dir, v) * h.ncells + c];
  double ne = h.ne_stage ? h.ne_stage[c] : hcell_ne(h, c);
  double w = hcell_w(h, f, c, ne);
  if (w < 0.0)
    w = 0.0;
  if (w > 1.0)
    w = 1.0;
  if (h.pe)
    q8[4] += w * h.pe[c]; /* P_eff = P_s + w_s P_e */
  int I = (int)(c % (size_t)h.nxtot);
  int J = (int)((c / (size_t)h.nxtot) % (size_t)h.nytot);
  int K = (int)(c / ((size_t)h.nxtot * (size_t)h.nytot));
  double bc[3];
  hcell_b_center(h, K, J, I, bc);
  int cn = dir;
  int ct1 = sweep_map_v(dir, 2) - 1;
  int ct2 = sweep_map_v(dir, 3) - 1;
  double sw = sqrt(w);
  q8[5] = sw * bc[cn];
  q8[6] = sw * bc[ct1];
  q8[7] = sw * bc[ct2];
}

/* B1 pass states of one cell. Pass A carries (rho, v, P_s + w_s P_e)
   built from the fields as they stand; pass B carries
   (rho*P_s/P_eff, v, P_s) and is rebuilt from the PRISTINE snapshot plus
   the fields' CURRENT charge share. Cells with n_e <= 0 carry w_s = 0,
   so P_eff falls back to P_s there and the scale to 1. */
KOKKOS_INLINE_FUNCTION
static void hpe_state_a_at(const HydroP &h, int f, int dir, size_t base,
                           int ml, double *qa) {
  double q[5];
  hprim_at(h, f, dir, base, ml, q);
  const size_t c = base + (size_t)ml * h.stride;
  const double ne = hcell_ne(h, c);
  const double w = hcell_w(h, f, c, ne);
  qa[0] = q[0];
  qa[1] = q[1];
  qa[2] = q[2];
  qa[3] = q[3];
  qa[4] = q[4] + w * h.pe[c];
}

KOKKOS_INLINE_FUNCTION
static void hpe_state_b(const HydroP &h, int f, size_t c, const double *q,
                        double *qb) {
  const double ne = hcell_ne(h, c);
  const double w = hcell_w(h, f, c, ne);
  const double peff = q[4] + w * h.pe[c];
  const double scale = peff > 1e-30 ? q[4] / peff : 1.0;
  qb[0] = q[0] * scale;
  qb[1] = q[1];
  qb[2] = q[2];
  qb[3] = q[3];
  qb[4] = q[4];
}

/* The electron pseudo-fluid state of one cell, assembled from the CURRENT
   ion fields (quasi-neutrality) exactly like the old line gather:
   n_e = sum_ion q_s n_s, u_e = sum_ion q_s n_s u_s / n_e, and the pseudo
   mass density is the charge-share-weighted ion mass density
   rho_e = sum_ion (n_s / n_e) rho_s. Vacuum cells (no ion above the
   density floor) carry a floor state in the sweep and their stored P_e
   is never updated. */
KOKKOS_INLINE_FUNCTION
static void helec_state_at(const HydroP &h, int dir, size_t c, double *qe) {
  double ne = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  double u[3];
  double rhoe = 0.0;
  int f;
  for (f = 0; f < h.nfluid; f++) {
    double ns;
    if (h.charge[f] == 0.0)
      continue;
    ns = h.charge[f] * hslot(h, f, XMF_SLOT_RHO)[c] / h.mass_code[f];
    ne += ns;
    mx += ns * hslot(h, f, XMF_SLOT_VX)[c];
    my += ns * hslot(h, f, XMF_SLOT_VY)[c];
    mz += ns * hslot(h, f, XMF_SLOT_VZ)[c];
  }
  if (hcell_ionized(h, c)) {
    u[0] = mx / ne;
    u[1] = my / ne;
    u[2] = mz / ne;
    for (f = 0; f < h.nfluid; f++) {
      double ns;
      if (h.charge[f] == 0.0)
        continue;
      ns = h.charge[f] * hslot(h, f, XMF_SLOT_RHO)[c] / h.mass_code[f];
      rhoe += (ns / ne) * hslot(h, f, XMF_SLOT_RHO)[c];
    }
    qe[0] = rhoe;
    qe[1] = u[sweep_map_v(dir, 1) - 1];
    qe[2] = u[sweep_map_v(dir, 2) - 1];
    qe[3] = u[sweep_map_v(dir, 3) - 1];
    qe[4] = h.pe[c];
  } else {
    qe[0] = 1e-30;
    qe[1] = 0.0;
    qe[2] = 0.0;
    qe[3] = 0.0;
    qe[4] = 1e-30;
  }
}

/* Sweep every interior cell of fluid f along direction dir
   (0=x1, 1=x2, 2=x3). C2sup: one MDRange face kernel (all interior
   faces of the direction, read-only) plus one MDRange cell kernel (all
   interior cells; each iteration writes only its own cell). Transverse
   ghost cells are not updated (fill_bc owns them). Returns 0, or -1
   when the face-flux allocation fails. */
static int sweep_dir(XmfRun *run, int f, int dir, double dt) {
  int ng = run->ng;
  int n = (dir == 0) ? run->nx : (dir == 1) ? run->ny : run->nz;
  int ltot = (dir == 0) ? run->nxtot : (dir == 1) ? run->nytot : run->nztot;
  size_t stride = (dir == 0) ? 1
                  : (dir == 1) ? (size_t)run->nxtot
                               : (size_t)run->nxtot * (size_t)run->nytot;
  /* Transverse extents in the original outer/inner loop order
     (dir 0: (k,j); dir 1: (k,i); dir 2: (j,i)). */
  int t0 = (dir == 0) ? run->ny : run->nx;
  int t1 = (dir == 0) ? run->nz : (dir == 1) ? run->nz : run->ny;
  int nlines = t1 * t0;
  int nface = n + 1; /* interior faces per line: ng-1 .. ng+n-1 */
  HydroP h = hydro_p(run, dir, dt, NULL);
  h.nline = n;
  h.ltot = ltot;
  h.stride = stride;
  try {
    XmfFaceWs flux("xmf_sweep_flux", (size_t)nlines, (size_t)nface, 5);
    Kokkos::parallel_for(
        "xmf_hydro_sweep_faces",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, nface}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng - 1 + s; /* face between cells m and m+1 */
          const size_t base = hline_base(h, dir, ng, a, b);
          const int line = a * t0 + b;
          double q0[5], q1[5], F[5];
          hprim_at(h, f, dir, base, m, q0);
          hprim_at(h, f, dir, base, m + 1, q1);
          if (h.recon == XMF_RECON_PCM) {
            riemann_flux(h.riemann, h.gamma, q0, q1, F);
          } else {
            double qm1[5], qp2[5], ql[5], qr[5];
            hprim_at(h, f, dir, base, m - 1, qm1);
            hprim_at(h, f, dir, base, m + 2, qp2);
            face_states(h.recon, qm1, q0, q1, qp2, ql, qr);
            riemann_flux(h.riemann, h.gamma, ql, qr, F);
          }
          for (int v = 0; v < 5; v++)
            flux(line, s, v) = F[v];
        });
    Kokkos::parallel_for(
        "xmf_hydro_sweep_cells",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, n}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const size_t c = base + (size_t)m * h.stride;
          const int line = a * t0 + b;
          double *slot0 =
              h.fields + ((size_t)f * XMF_SLOT_NVAR) * h.ncells;
          double q[5], U[5];
          hprim_at(h, f, dir, base, m, q);
          prim5_to_cons(q, h.gmm1, U);
          const double crl = h.dt * h.A[m] / h.dV[m];     /* left face m-1 */
          const double clr = h.dt * h.A[m + 1] / h.dV[m]; /* right face m */
          for (int v = 0; v < 5; v++)
            U[v] += crl * flux(line, s, v);
          for (int v = 0; v < 5; v++)
            U[v] -= clr * flux(line, s + 1, v);
          cons5_to_prim(U, h.gmm1, q);
          for (int v = 0; v < XMF_SLOT_NVAR; v++)
            slot0[(size_t)sweep_map_v(dir, v) * h.ncells + c] = q[v];
        });
  } catch (...) {
    return -1;
  }
  return 0;
}

/* ---- M4 ion MHD sweep (magnetic gate open) ------------------------------
   MDRange face + cell kernels for the 8-component ideal-MHD operator.
   The face kernel builds per-cell 8-component states (P_eff + B_s) and
   dispatches the MHD Riemann solver; only the five fluid flux rows are
   stored. The cell kernel converts to conservative (with the per-ion
   magnetic energy share |B_s|^2/2), applies the two face fluxes (left
   first, right second -- the unchanged per-cell order), recovers
   primitives with the magnetic share removed and P_s = P_eff - w_s P_e,
   and scatters. Neutrals never dispatch here (w_s = 0 => B_s = 0, the
   plain HD path). Gate-closed runs never dispatch here. */
static int sweep_dir_mhd(XmfRun *run, int f, int dir, double dt) {
  int ng = run->ng;
  int n = (dir == 0) ? run->nx : (dir == 1) ? run->ny : run->nz;
  int ltot = (dir == 0) ? run->nxtot : (dir == 1) ? run->nytot : run->nztot;
  size_t stride = (dir == 0) ? 1
                  : (dir == 1) ? (size_t)run->nxtot
                               : (size_t)run->nxtot * (size_t)run->nytot;
  int t0 = (dir == 0) ? run->ny : run->nx;
  int t1 = (dir == 0) ? run->nz : (dir == 1) ? run->nz : run->ny;
  int nlines = t1 * t0;
  int nface = n + 1;
  HydroP h = hydro_p(run, dir, dt, NULL);
  h.nline = n;
  h.ltot = ltot;
  h.stride = stride;
  try {
    XmfFaceWs flux("xmf_sweepm_flux", (size_t)nlines, (size_t)nface, 5);
    Kokkos::parallel_for(
        "xmf_hydro_sweep_mhd_faces",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, nface}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng - 1 + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const int line = a * t0 + b;
          double q0[8], q1[8], F[5];
          hmhd_state_at(h, f, dir, base, m, q0);
          hmhd_state_at(h, f, dir, base, m + 1, q1);
          if (h.recon == XMF_RECON_PCM) {
            mhd_riemann_flux(h.riemann, h.gamma, q0, q1, F);
          } else {
            double qm1[8], qp2[8], ql[8], qr[8];
            hmhd_state_at(h, f, dir, base, m - 1, qm1);
            hmhd_state_at(h, f, dir, base, m + 2, qp2);
            face_states8(h.recon, qm1, q0, q1, qp2, ql, qr);
            mhd_riemann_flux(h.riemann, h.gamma, ql, qr, F);
          }
          for (int v = 0; v < 5; v++)
            flux(line, s, v) = F[v];
        });
    Kokkos::parallel_for(
        "xmf_hydro_sweep_mhd_cells",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, n}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const size_t c = base + (size_t)m * h.stride;
          const int line = a * t0 + b;
          double *slot0 =
              h.fields + ((size_t)f * XMF_SLOT_NVAR) * h.ncells;
          double q8[8], U[5];
          hmhd_state_at(h, f, dir, base, m, q8);
          /* prim -> cons (5 fluid rows with the per-ion magnetic share
             |B_s|^2/2 in the energy). */
          {
            double U8[8];
            mhd_prim_to_cons(q8, h.gmm1, U8);
            for (int v = 0; v < 5; v++)
              U[v] = U8[v];
          }
          const double crl = h.dt * h.A[m] / h.dV[m];     /* left face m-1 */
          const double clr = h.dt * h.A[m + 1] / h.dV[m]; /* right face m */
          for (int v = 0; v < 5; v++)
            U[v] += crl * flux(line, s, v);
          for (int v = 0; v < 5; v++)
            U[v] -= clr * flux(line, s + 1, v);
          /* cons -> prim with the magnetic share removed: subtract the
             same |B_s|^2/2 = w_s |B|^2/2 the sweep added, then recover
             P_s from P_eff (frozen w_s, P_e of the stage). */
          {
            double ne = h.ne_stage ? h.ne_stage[c] : hcell_ne(h, c);
            double w = hcell_w(h, f, c, ne);
            if (w < 0.0)
              w = 0.0;
            if (w > 1.0)
              w = 1.0;
            int I = (int)(c % (size_t)h.nxtot);
            int J = (int)((c / (size_t)h.nxtot) % (size_t)h.nytot);
            int K = (int)(c / ((size_t)h.nxtot * (size_t)h.nytot));
            double bc[3];
            hcell_b_center(h, K, J, I, bc);
            double r = U[0] > 1e-30 ? U[0] : 1e-30;
            double ek = 0.5 * (U[1] * U[1] + U[2] * U[2] + U[3] * U[3]) / r;
            double b2s = w * (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
            double peff = h.gmm1 * (U[4] - ek - 0.5 * b2s);
            double ps = h.pe ? peff - w * h.pe[c] : peff;
            if (ps < 1e-30)
              ps = 1e-30;
            double q[5];
            q[0] = r;
            q[1] = U[1] / r;
            q[2] = U[2] / r;
            q[3] = U[3] / r;
            q[4] = ps;
            for (int v = 0; v < XMF_SLOT_NVAR; v++)
              slot0[(size_t)sweep_map_v(dir, v) * h.ncells + c] = q[v];
          }
        });
  } catch (...) {
    return -1;
  }
  return 0;
}

/* ---- B1 device closure passes (C2sup: MDRange) --------------------------
   The two-pass ion sweep and the P_e pseudo-fluid sweep of the B1 block,
   device-side. The line-parallel form is replaced by the same face/cell
   MDRange split as sweep_dir, doubled for the two passes and ordered so
   every read sees the very field state the old line code saw:

     gather  -- snapshot the pristine five primitives of every line cell
                (device View qsv, scoped to the sweep);
     faces A -- pass-A face fluxes from the pristine fields
                (P_eff = P_s + w_s P_e states), read-only;
     cells A -- apply the buffered pass-A fluxes, keep rho and the three
                momenta, scatter them back;
     faces B -- pass-B face fluxes from the PRISTINE snapshot states with
                the POST-pass-A charge share, read-only;
     cells B -- apply the buffered pass-B fluxes, keep only the pressure;
                with a heat scratch (B2) the keep block measures pass B's
                own increment H and splits it.

   Every interior cell is one parallel iteration of each kernel and
   writes only its own cell (heat[c] too), so the deposits stay disjoint
   exactly like the old parallel line space. */
static int sweep_dir_pe(XmfRun *run, int f, int dir, double dt, void *heat) {
  int ng = run->ng;
  int n = (dir == 0) ? run->nx : (dir == 1) ? run->ny : run->nz;
  int ltot = (dir == 0) ? run->nxtot : (dir == 1) ? run->nytot : run->nztot;
  size_t stride = (dir == 0) ? 1
                  : (dir == 1) ? (size_t)run->nxtot
                               : (size_t)run->nxtot * (size_t)run->nytot;
  int t0 = (dir == 0) ? run->ny : run->nx;
  int t1 = (dir == 0) ? run->nz : (dir == 1) ? run->nz : run->ny;
  int nlines = t1 * t0;
  int nface = n + 1;
  HydroP h = hydro_p(run, dir, dt, heat);
  h.nline = n;
  h.ltot = ltot;
  h.stride = stride;
  try {
    XmfFaceWs qsv("xmf_sweepp_qsv", (size_t)nlines, (size_t)ltot, 5);
    XmfFaceWs fa("xmf_sweepp_fa", (size_t)nlines, (size_t)nface, 5);
    XmfFaceWs fb("xmf_sweepp_fb", (size_t)nlines, (size_t)nface, 5);
    /* Gather: snapshot the pristine five primitives of every line cell. */
    Kokkos::parallel_for(
        "xmf_hydro_sweep_pe_gather",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, ltot}),
        KOKKOS_LAMBDA(const int a, const int b, const int m) {
          const size_t base = hline_base(h, dir, ng, a, b);
          const int line = a * t0 + b;
          double q[5];
          hprim_at(h, f, dir, base, m, q);
          for (int v = 0; v < 5; v++)
            qsv(line, m, v) = q[v];
        });
    /* Pass A face fluxes: P_eff states from the pristine fields. */
    Kokkos::parallel_for(
        "xmf_hydro_sweep_pe_faces_a",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, nface}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng - 1 + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const int line = a * t0 + b;
          double a0[5], a1[5], F[5];
          hpe_state_a_at(h, f, dir, base, m, a0);
          hpe_state_a_at(h, f, dir, base, m + 1, a1);
          if (h.recon == XMF_RECON_PCM) {
            riemann_flux(h.riemann, h.gamma, a0, a1, F);
          } else {
            double am1[5], ap2[5], ql[5], qr[5];
            hpe_state_a_at(h, f, dir, base, m - 1, am1);
            hpe_state_a_at(h, f, dir, base, m + 2, ap2);
            face_states(h.recon, am1, a0, a1, ap2, ql, qr);
            riemann_flux(h.riemann, h.gamma, ql, qr, F);
          }
          for (int v = 0; v < 5; v++)
            fa(line, s, v) = F[v];
        });
    /* Pass A cell update: keep rho + momenta, scatter back. */
    Kokkos::parallel_for(
        "xmf_hydro_sweep_pe_cells_a",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, n}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const size_t c = base + (size_t)m * h.stride;
          const int line = a * t0 + b;
          double *slot0 =
              h.fields + ((size_t)f * XMF_SLOT_NVAR) * h.ncells;
          double q[5], U[5];
          hprim_at(h, f, dir, base, m, q);
          const double ne = hcell_ne(h, c);
          const double w = hcell_w(h, f, c, ne);
          q[4] = q[4] + w * h.pe[c];
          prim5_to_cons(q, h.gmm1, U);
          const double crl = h.dt * h.A[m] / h.dV[m];     /* left face m-1 */
          const double clr = h.dt * h.A[m + 1] / h.dV[m]; /* right face m */
          for (int v = 0; v < 5; v++)
            U[v] += crl * fa(line, s, v);
          for (int v = 0; v < 5; v++)
            U[v] -= clr * fa(line, s + 1, v);
          cons5_to_prim(U, h.gmm1, q);
          for (int v = 0; v < 4; v++)
            slot0[(size_t)sweep_map_v(dir, v) * h.ncells + c] = q[v];
        });
    /* Pass B face fluxes: pristine snapshot states with post-A charge
       share (the old pass-B setup re-read hcell_ne/w after pass A's
       scatter -- the fields now carry the pass-A rho and momenta). */
    Kokkos::parallel_for(
        "xmf_hydro_sweep_pe_faces_b",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, nface}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng - 1 + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const int line = a * t0 + b;
          double b0[5], b1[5], q[5], F[5];
          size_t c;
          int v;
          c = base + (size_t)m * h.stride;
          for (v = 0; v < 5; v++)
            q[v] = qsv(line, m, v);
          hpe_state_b(h, f, c, q, b0);
          c = base + (size_t)(m + 1) * h.stride;
          for (v = 0; v < 5; v++)
            q[v] = qsv(line, m + 1, v);
          hpe_state_b(h, f, c, q, b1);
          if (h.recon == XMF_RECON_PCM) {
            riemann_flux(h.riemann, h.gamma, b0, b1, F);
          } else {
            double bm1[5], bp2[5], ql[5], qr[5];
            c = base + (size_t)(m - 1) * h.stride;
            for (v = 0; v < 5; v++)
              q[v] = qsv(line, m - 1, v);
            hpe_state_b(h, f, c, q, bm1);
            c = base + (size_t)(m + 2) * h.stride;
            for (v = 0; v < 5; v++)
              q[v] = qsv(line, m + 2, v);
            hpe_state_b(h, f, c, q, bp2);
            face_states(h.recon, bm1, b0, b1, bp2, ql, qr);
            riemann_flux(h.riemann, h.gamma, ql, qr, F);
          }
          for (v = 0; v < 5; v++)
            fb(line, s, v) = F[v];
        });
    /* Pass B cell update: keep only the pressure. With a heat scratch
       (B2), measure pass B's own increment H and split it. */
    Kokkos::parallel_for(
        "xmf_hydro_sweep_pe_cells_b",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, n}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const size_t c = base + (size_t)m * h.stride;
          const int line = a * t0 + b;
          double *slot0 =
              h.fields + ((size_t)f * XMF_SLOT_NVAR) * h.ncells;
          double q[5], qe[5], U[5];
          for (int v = 0; v < 5; v++)
            q[v] = qsv(line, m, v); /* pristine primitives (pre pass A) */
          hpe_state_b(h, f, c, q, qe);
          prim5_to_cons(qe, h.gmm1, U);
          const double crl = h.dt * h.A[m] / h.dV[m];     /* left face m-1 */
          const double clr = h.dt * h.A[m + 1] / h.dV[m]; /* right face m */
          for (int v = 0; v < 5; v++)
            U[v] += crl * fb(line, s, v);
          for (int v = 0; v < 5; v++)
            U[v] -= clr * fb(line, s + 1, v);
          cons5_to_prim(U, h.gmm1, qe);
          const double pB = qe[4];
          double ps_new = pB;
          if (h.heat && hcell_ionized(h, c)) {
            const double ne = hcell_ne(h, c);
            const double w = hcell_w(h, f, c, ne);
            const double peff = q[4] + w * h.pe[c];
            const double scale = peff > 1e-30 ? q[4] / peff : 1.0;
            const double rho0 = q[0] * scale;
            const double rho1 = qe[0];
            const double share =
                (h.shock_heat == XMF_SHOCK_HEAT_BY_CHARGE) ? w : 1.0;
            double H = pB - q[4] * pow(rho1 / rho0, h.gamma);
            if (!(H > 0.0))
              H = 0.0;
            ps_new = pB - share * H;
            if (ps_new < 1e-30)
              ps_new = 1e-30;
            h.heat[c] += pB - ps_new;
          }
          slot0[(size_t)sweep_map_v(dir, 4) * h.ncells + c] = ps_new;
        });
  } catch (...) {
    return -1;
  }
  return 0;
}

/* ---- B1 P_e pseudo-fluid sweep (C2sup: MDRange) -------------------------
   The electron pressure rides one pseudo-fluid sweep per direction,
   assembled from the ion fields. One MDRange face kernel (all interior
   faces, read-only, buffering the five-component fluxes) plus one
   MDRange cell kernel (each iteration assembles its own cell's state,
   applies its own two face fluxes and commits P_e under the unchanged
   write-back gate). */
static int sweep_elec_dir(XmfRun *run, int dir, double dt) {
  int ng = run->ng;
  int n = (dir == 0) ? run->nx : (dir == 1) ? run->ny : run->nz;
  int ltot = (dir == 0) ? run->nxtot : (dir == 1) ? run->nytot : run->nztot;
  size_t stride = (dir == 0) ? 1
                  : (dir == 1) ? (size_t)run->nxtot
                               : (size_t)run->nxtot * (size_t)run->nytot;
  int t0 = (dir == 0) ? run->ny : run->nx;
  int t1 = (dir == 0) ? run->nz : (dir == 1) ? run->nz : run->ny;
  int nlines = t1 * t0;
  int nface = n + 1;
  HydroP h = hydro_p(run, dir, dt, NULL);
  h.nline = n;
  h.ltot = ltot;
  h.stride = stride;
  try {
    XmfFaceWs fe("xmf_sweepe_f", (size_t)nlines, (size_t)nface, 5);
    Kokkos::parallel_for(
        "xmf_hydro_sweep_elec_faces",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, nface}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng - 1 + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const int line = a * t0 + b;
          double q0[5], q1[5], F[5];
          helec_state_at(h, dir, base + (size_t)m * h.stride, q0);
          helec_state_at(h, dir, base + (size_t)(m + 1) * h.stride, q1);
          if (h.recon == XMF_RECON_PCM) {
            riemann_flux(h.riemann, h.gamma, q0, q1, F);
          } else {
            double qm1[5], qp2[5], ql[5], qr[5];
            helec_state_at(h, dir, base + (size_t)(m - 1) * h.stride, qm1);
            helec_state_at(h, dir, base + (size_t)(m + 2) * h.stride, qp2);
            face_states(h.recon, qm1, q0, q1, qp2, ql, qr);
            riemann_flux(h.riemann, h.gamma, ql, qr, F);
          }
          for (int v = 0; v < 5; v++)
            fe(line, s, v) = F[v];
        });
    Kokkos::parallel_for(
        "xmf_hydro_sweep_elec_cells",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {t1, t0, n}),
        KOKKOS_LAMBDA(const int a, const int b, const int s) {
          const int m = ng + s;
          const size_t base = hline_base(h, dir, ng, a, b);
          const size_t c = base + (size_t)m * h.stride;
          const int line = a * t0 + b;
          double qe[5], U[5];
          helec_state_at(h, dir, c, qe);
          prim5_to_cons(qe, h.gmm1, U);
          const double crl = h.dt * h.A[m] / h.dV[m];     /* left face m-1 */
          const double clr = h.dt * h.A[m + 1] / h.dV[m]; /* right face m */
          for (int v = 0; v < 5; v++)
            U[v] += crl * fe(line, s, v);
          for (int v = 0; v < 5; v++)
            U[v] -= clr * fe(line, s + 1, v);
          cons5_to_prim(U, h.gmm1, qe);
          if (hcell_ionized(h, c) && qe[4] > XMF_RHO_FLOOR)
            h.pe[c] = qe[4];
        });
  } catch (...) {
    return -1;
  }
  return 0;
}


/* ---- Three-direction CFL (per-cell metrics, A1 stretch supplement) ------
   dt = cfl / max_cell sum_d (s_{d,cell} / dx_{d,cell}). The sum runs over
   ACTIVE directions only (n_d > 1): a degenerate direction (n_d == 1) is
   left out, which makes 1D reduce exactly to cfl*dx/s. The max runs over
   every interior cell, and s_{d,cell} is that cell's own |v_d| + c_s --
   per-cell widths from mesh[d].dx, never a scalar spacing. An idle
   (all-zero) extra fluid contributes nothing (guarded by the rho/P
   floors); a cell with no live fluid keeps the 1e-12 speed floor.

   Uniform-grid bit-for-bit note: when every dx_{d,cell} equals one shared
   constant, dividing by it commutes with the max (the argmax cell is the
   same and the very same float division gets computed), so the formula
   collapses exactly to the old cfl / sum_d (s_d / dx_d) with run-wide
   s_d -- the empty-table pcm Sod pins stay untouched. dt_max (when > 0)
   caps dt from above.

   B3 hard gate (the old yardstick merged its ELCI sweep's dt into
   dt_next, mmd_model_adv.c:225-229): when the electron closure is on AND
   at least two charged fluids are loaded, the P_e pseudo-fluid sweep's
   own signal speed |u_{e,d}| + c_e joins the same per-cell max. The
   electron state is assembled exactly as helec_state_at does --
   n_e = sum q_s n_s, u_e = sum q_s n_s u_s / n_e,
   rho_e = sum (n_s/n_e) rho_s, c_e = sqrt(gamma P_e / rho_e) -- so the
   CFL bounds the very state that sweep advances. With a single charged
   fluid the assembly reduces to rho_e = rho_ion and
   c_e <= sqrt(gamma (P_s + P_e)/rho_s) -- the ion's own P_eff sound
   speed -- so the ion term already covers the pseudo sweep and the gate
   stays out: the single-ion dt stays bit for bit the B1 one (u_e =
   (ns*v)/ns could otherwise round a hair above v and flip the last ulp
   of the max). Vacuum cells (no ion above the density floor) carry the
   floor state in the pseudo sweep and contribute no signal speed here,
   exactly like an idle fluid in the loop below.

   C2: this whole block stays a serial HOST loop over the freshly pulled
   host mirror (xmf_step pulls before compute_dt) -- the reduction order
   and every comparison keep the host max bit for bit. */

/* True when cell c carries at least one ion species whose density is
   strictly above the hydro floor -- genuine plasma, not a floored vacuum
   cell. The ion role is q_s > 0 (species names are never consulted). */
static int cell_ionized(const XmfRun *run, size_t c) {
  int f;
  for (f = 0; f < run->nfluid; f++)
    if (run->charge[f] > 0.0 && fld(run, f, XMF_SLOT_RHO)[c] > XMF_RHO_FLOOR)
      return 1;
  return 0;
}

/* Quasi-neutral electron number density of one cell:
   n_e = sum_ion q_s n_s = sum_f q_s rho_f / mass_code[f]
   (neutrals carry q_s = 0 and drop out). */
static double cell_ne(const XmfRun *run, size_t c) {
  int f;
  double ne = 0.0;
  for (f = 0; f < run->nfluid; f++)
    ne += run->charge[f] * fld(run, f, XMF_SLOT_RHO)[c] / run->mass_code[f];
  return ne;
}

/* Charge share w_s of fluid f at one cell: q_s n_s / n_e for ions. Cells
   with n_e <= 0 -- realised as cells with no ion above the density floor
   (cell_ionized false), which is what a swept-to-vacuum cell becomes after
   the cons->prim floor -- fall back to w_s = 0 (hydro then sees
   P_eff = P_s), as do non-positive charges (neutrals by the q_s != 0 ion
   rule). */
static double cell_w(const XmfRun *run, int f, size_t c, double ne) {
  if (run->charge[f] <= 0.0 || ne <= 0.0 || !cell_ionized(run, c))
    return 0.0;
  return run->charge[f] * fld(run, f, XMF_SLOT_RHO)[c] / run->mass_code[f] /
         ne;
}

/* M4: host-side staggered index of bf[comp] (the xmf_run.h layout; the
   comp entry of c is the FACE index along the component's direction,
   the other two are CELL indices). */
static size_t host_bidx(const XmfRun *run, int comp, const int c[3]) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  int n1 = (d1 == 0) ? run->nxtot : (d1 == 1) ? run->nytot : run->nztot;
  int nd = (comp == 0) ? run->nxtot : (comp == 1) ? run->nytot : run->nztot;
  return (((size_t)c[d2] * (size_t)n1) + (size_t)c[d1]) * (size_t)(nd + 1) +
         (size_t)c[comp];
}

/* M4: cell-centred B (average of the two face values along each
   direction) -- the same definition the ion MHD sweeps feed. */
static void cell_b_center(const XmfRun *run, int k, int j, int i,
                          double bc[3]) {
  int c0[3] = {i, j, k};
  int cx[3] = {i + 1, j, k};
  int cy[3] = {i, j + 1, k};
  int cz[3] = {i, j, k + 1};
  bc[0] = 0.5 * (run->bf[0][host_bidx(run, 0, c0)] +
                 run->bf[0][host_bidx(run, 0, cx)]);
  bc[1] = 0.5 * (run->bf[1][host_bidx(run, 1, c0)] +
                 run->bf[1][host_bidx(run, 1, cy)]);
  bc[2] = 0.5 * (run->bf[2][host_bidx(run, 2, c0)] +
                 run->bf[2][host_bidx(run, 2, cz)]);
}

static double cell_signal_sum(const XmfRun *run, int i, int j, int k) {
  int nd[3];
  int d, f;
  double inv = 0.0;
  size_t c = cidx(run, k, j, i);
  /* B1: the ion sound speed reads P_eff = P_s + w_s P_e when the closure
     gate is on; with run->pe == NULL the whole branch is skipped and the
     formula below is bit-for-bit the pre-B1 one. */
  int ele = elec_closure_active(run);
  /* M4: with the magnetic gate open the ions' signal speed is the
     fast-magnetosonic one; the gate-closed path below stays bit-for-bit
     the pre-M4 one (mag == 0 skips every new branch). */
  int mag = run->bf[0] != NULL;
  double ne = (ele || mag) ? cell_ne(run, c) : 0.0;
  double bc[3];
  bc[0] = 0.0;
  bc[1] = 0.0;
  bc[2] = 0.0;
  if (mag)
    cell_b_center(run, k, j, i, bc);
  /* B3 gate: charged-fluid count and the assembled electron state. */
  int nchg = 0;
  double ue0 = 0.0, ue1 = 0.0, ue2 = 0.0, rhoe = 0.0, pe_use = 0.0;
  if (ele) {
    for (f = 0; f < run->nfluid; f++)
      if (run->charge[f] > 0.0)
        nchg++;
    if (nchg >= 2) {
      /* Exactly helec_state_at's sums (the skip-charge==0 form; the
         cell_ne value above is bit for bit this ne -- neutral terms add
         exactly 0.0). */
      double mx = 0.0, my = 0.0, mz = 0.0;
      ne = 0.0;
      for (f = 0; f < run->nfluid; f++) {
        double ns;
        if (run->charge[f] == 0.0)
          continue;
        ns = run->charge[f] * fld(run, f, XMF_SLOT_RHO)[c] /
             run->mass_code[f];
        ne += ns;
        mx += ns * fld(run, f, XMF_SLOT_VX)[c];
        my += ns * fld(run, f, XMF_SLOT_VY)[c];
        mz += ns * fld(run, f, XMF_SLOT_VZ)[c];
      }
      if (cell_ionized(run, c)) {
        ue0 = mx / ne;
        ue1 = my / ne;
        ue2 = mz / ne;
        for (f = 0; f < run->nfluid; f++) {
          double ns;
          if (run->charge[f] == 0.0)
            continue;
          ns = run->charge[f] * fld(run, f, XMF_SLOT_RHO)[c] /
               run->mass_code[f];
          rhoe += (ns / ne) * fld(run, f, XMF_SLOT_RHO)[c];
        }
        pe_use = run->pe[c];
      }
      /* else: vacuum cell -- rhoe/pe_use stay 0 and the merge below
         skips the electron term, the idle-fluid floor behaviour. */
    }
  }
  nd[0] = run->nx;
  nd[1] = run->ny;
  nd[2] = run->nz;
  for (d = 0; d < 3; d++) {
    int var, idx;
    double s = 0.0;
    if (nd[d] <= 1)
      continue; /* n_d == 1 directions never enter the sum */
    var = (d == 0) ? XMF_SLOT_VX : (d == 1) ? XMF_SLOT_VY : XMF_SLOT_VZ;
    idx = (d == 0) ? i : (d == 1) ? j : k;
    for (f = 0; f < run->nfluid; f++) {
      const double *fn = fld(run, f, XMF_SLOT_RHO);
      const double *fv = fld(run, f, var);
      const double *fp = fld(run, f, XMF_SLOT_P);
      double puse = fp[c];
      double cs;
      if (ele) {
        double w = cell_w(run, f, c, ne);
        puse = fp[c] + w * run->pe[c];
      }
      if (fn[c] <= 1e-30 || puse <= 1e-30)
        continue; /* zero-density/pressure fluid: no signal speed */
      if (mag && run->charge[f] > 0.0) {
        /* M4: ion fast-magnetosonic speed, the card's formula with the
           per-ion quantities a_s^2 = gamma P_eff,s / rho_s,
           b_s^2 = |B_s|^2/rho_s = w_s |B|^2/rho_s and
           b_{s,n}^2 = (B_{s,n})^2/rho_s = w_s B_n^2/rho_s
           (B_s = sqrt(w_s) B, w_s clamped to [0,1] exactly as in the
           sweep). The zero-field branch returns a_s^2 as-is, so a B = 0
           ion stays bit-for-bit the HD |u| + c_s. A dilute heavy ion
           may pinch dt through a tiny a_s -- that is physics, not a
           bug; dt_max still caps. */
        double w = cell_w(run, f, c, ne);
        double b2, bn2, a2, cf2v, sig, bn;
        if (w < 0.0)
          w = 0.0;
        if (w > 1.0)
          w = 1.0;
        /* b_s^2 = w_s |B|^2 / rho_s (energy-density |B|^2 normalised by
           the ion's OWN mass density -- fn[c] > 1e-30 is guarded by the
           continue above); same convention as mhd_cf2. */
        b2 = w * (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]) / fn[c];
        bn = (d == 0) ? bc[0] : (d == 1) ? bc[1] : bc[2];
        bn2 = w * bn * bn / fn[c];
        a2 = run->gamma * puse / fn[c];
        if (b2 <= 0.0)
          cf2v = a2;
        else {
          double ssum = a2 + b2;
          double disc = ssum * ssum - 4.0 * a2 * bn2;
          if (disc < 0.0)
            disc = 0.0;
          cf2v = 0.5 * (ssum + sqrt(disc));
        }
        sig = fabs(fv[c]) + sqrt(cf2v);
        if (sig > s)
          s = sig;
      } else {
        /* Neutrals (and the whole gate-closed path): unchanged. */
        cs = sqrt(run->gamma * puse / fn[c]);
        if (fabs(fv[c]) + cs > s)
          s = fabs(fv[c]) + cs;
      }
    }
    if (rhoe > 1e-30 && pe_use > 1e-30) {
      /* B3 hard gate: the pseudo sweep's own |u_{e,d}| + c_e. nchg >= 2
         is folded into rhoe/pe_use being non-zero. */
      double ue_d = (d == 0) ? ue0 : (d == 1) ? ue1 : ue2;
      double se = fabs(ue_d) + sqrt(run->gamma * pe_use / rhoe);
      if (se > s)
        s = se;
    }
    s = s > 1e-12 ? s : 1e-12;
    inv += s / run->mesh[d].dx[idx];
  }
  return inv;
}

/* ---- Port C: device CFL bundle & kernel -----------------------------------
   Value-type POD bundle (copied into kernels by value, same convention as
   HydroP / BcP). Carries the device pointers and scalars needed for the
   per-cell signal sum; the host elec_closure_active gate and the
   magnetic gate are captured as ints so the device lambda never calls
   them. MHD extension over the hd CflP: carries the three face-centred
   B device pointers (bf) so the per-cell signal sum face-averages B to
   cell center before the fast-magnetosonic formula, matching the host
   cell_b_center exactly. */
struct CflP {
  double *fields;
  double *pe;              /* NULL when electrons off */
  double *bf[3];           /* face-centred B; bf[0] NULL = gate closed */
  const double *charge;    /* [nfluid] */
  const double *mass_code; /* [nfluid] */
  const double *dx[3];     /* per-direction cell widths */
  int nfluid, ng;
  int nx, ny, nz;
  int nxtot, nytot, nztot;
  size_t ncells;
  double gamma;
  int ele;                 /* elec_closure_active captured as int */
  int mag;                 /* run->bf[0] != NULL captured as int */
};

KOKKOS_INLINE_FUNCTION
static size_t cfl_cidx(const CflP &cp, int k, int j, int i) {
  return (((size_t)k * (size_t)cp.nytot) + (size_t)j) * (size_t)cp.nxtot +
         (size_t)i;
}

KOKKOS_INLINE_FUNCTION
static double *cfl_slot(const CflP &cp, int f, int var) {
  return cp.fields +
         (((size_t)f * XMF_SLOT_NVAR + (size_t)var) * cp.ncells);
}

KOKKOS_INLINE_FUNCTION
static int cfl_cell_ionized(const CflP &cp, size_t c) {
  int f;
  for (f = 0; f < cp.nfluid; f++)
    if (cp.charge[f] > 0.0 &&
        cfl_slot(cp, f, XMF_SLOT_RHO)[c] > XMF_RHO_FLOOR)
      return 1;
  return 0;
}

KOKKOS_INLINE_FUNCTION
static double cfl_cell_ne(const CflP &cp, size_t c) {
  int f;
  double ne = 0.0;
  for (f = 0; f < cp.nfluid; f++)
    ne += cp.charge[f] * cfl_slot(cp, f, XMF_SLOT_RHO)[c] / cp.mass_code[f];
  return ne;
}

KOKKOS_INLINE_FUNCTION
static double cfl_cell_w(const CflP &cp, int f, size_t c, double ne) {
  if (cp.charge[f] <= 0.0 || ne <= 0.0 || !cfl_cell_ionized(cp, c))
    return 0.0;
  return cp.charge[f] * cfl_slot(cp, f, XMF_SLOT_RHO)[c] / cp.mass_code[f] /
         ne;
}

/* Staggered face index of bf[comp] on the CflP bundle (same layout as
   hbidx on HydroP). */
KOKKOS_INLINE_FUNCTION
static size_t cfl_bidx(const CflP &cp, int comp, int i0, int i1, int i2) {
  int c[3];
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  int n1 = (d1 == 0) ? cp.nxtot : (d1 == 1) ? cp.nytot : cp.nztot;
  int nd = (comp == 0) ? cp.nxtot : (comp == 1) ? cp.nytot : cp.nztot;
  c[0] = i0;
  c[1] = i1;
  c[2] = i2;
  return (((size_t)c[d2] * (size_t)n1) + (size_t)c[d1]) * (size_t)(nd + 1) +
         (size_t)c[comp];
}

/* Cell-centred B: average of the two face values along each direction
   (same formula as the host cell_b_center and the sweep's
   hcell_b_center). The face-centred B mirror is already on the device
   (Port A/B); this reads it directly. */
KOKKOS_INLINE_FUNCTION
static void cfl_cell_b_center(const CflP &cp, int k, int j, int i,
                              double bc[3]) {
  bc[0] = 0.5 * (cp.bf[0][cfl_bidx(cp, 0, i, j, k)] +
                 cp.bf[0][cfl_bidx(cp, 0, i + 1, j, k)]);
  bc[1] = 0.5 * (cp.bf[1][cfl_bidx(cp, 1, i, j, k)] +
                 cp.bf[1][cfl_bidx(cp, 1, i, j + 1, k)]);
  bc[2] = 0.5 * (cp.bf[2][cfl_bidx(cp, 2, i, j, k)] +
                 cp.bf[2][cfl_bidx(cp, 2, i, j, k + 1)]);
}

/* Per-cell signal sum on device: same formula, same floors, same branches
   as the host cell_signal_sum. MHD extension: when the magnetic gate is
   open, B is face-averaged to cell center (cfl_cell_b_center) before
   the fast-magnetosonic formula -- exactly matching the host path's
   cell_b_center -> cell_signal_sum chain. The three direction widths
   are read from the per-direction dx Views (cp.dx[d][idx]), matching
   the host's run->mesh[d].dx[idx]. Returns inv = sum_d s_d/dx_d for
   the cell. */
KOKKOS_INLINE_FUNCTION
static double cfl_cell_signal_sum(const CflP &cp, int i, int j, int k) {
  int nd[3];
  int d, f;
  double inv = 0.0;
  size_t c = cfl_cidx(cp, k, j, i);
  int ele = cp.ele;
  int mag = cp.mag;
  double ne = (ele || mag) ? cfl_cell_ne(cp, c) : 0.0;
  double bc[3];
  bc[0] = 0.0;
  bc[1] = 0.0;
  bc[2] = 0.0;
  if (mag)
    cfl_cell_b_center(cp, k, j, i, bc);
  /* B3 gate: charged-fluid count and the assembled electron state. */
  int nchg = 0;
  double ue0 = 0.0, ue1 = 0.0, ue2 = 0.0, rhoe = 0.0, pe_use = 0.0;
  if (ele) {
    for (f = 0; f < cp.nfluid; f++)
      if (cp.charge[f] > 0.0)
        nchg++;
    if (nchg >= 2) {
      double mx = 0.0, my = 0.0, mz = 0.0;
      ne = 0.0;
      for (f = 0; f < cp.nfluid; f++) {
        double ns;
        if (cp.charge[f] == 0.0)
          continue;
        ns = cp.charge[f] * cfl_slot(cp, f, XMF_SLOT_RHO)[c] /
             cp.mass_code[f];
        ne += ns;
        mx += ns * cfl_slot(cp, f, XMF_SLOT_VX)[c];
        my += ns * cfl_slot(cp, f, XMF_SLOT_VY)[c];
        mz += ns * cfl_slot(cp, f, XMF_SLOT_VZ)[c];
      }
      if (cfl_cell_ionized(cp, c)) {
        ue0 = mx / ne;
        ue1 = my / ne;
        ue2 = mz / ne;
        for (f = 0; f < cp.nfluid; f++) {
          double ns;
          if (cp.charge[f] == 0.0)
            continue;
          ns = cp.charge[f] * cfl_slot(cp, f, XMF_SLOT_RHO)[c] /
               cp.mass_code[f];
          rhoe += (ns / ne) * cfl_slot(cp, f, XMF_SLOT_RHO)[c];
        }
        pe_use = cp.pe[c];
      }
    }
  }
  nd[0] = cp.nx;
  nd[1] = cp.ny;
  nd[2] = cp.nz;
  for (d = 0; d < 3; d++) {
    int var, idx;
    double s = 0.0;
    if (nd[d] <= 1)
      continue; /* n_d == 1 directions never enter the sum */
    var = (d == 0) ? XMF_SLOT_VX : (d == 1) ? XMF_SLOT_VY : XMF_SLOT_VZ;
    idx = (d == 0) ? i : (d == 1) ? j : k;
    for (f = 0; f < cp.nfluid; f++) {
      const double *fn = cfl_slot(cp, f, XMF_SLOT_RHO);
      const double *fv = cfl_slot(cp, f, var);
      const double *fp = cfl_slot(cp, f, XMF_SLOT_P);
      double puse = fp[c];
      double cs;
      if (ele) {
        double w = cfl_cell_w(cp, f, c, ne);
        puse = fp[c] + w * cp.pe[c];
      }
      if (fn[c] <= 1e-30 || puse <= 1e-30)
        continue; /* zero-density/pressure fluid: no signal speed */
      if (mag && cp.charge[f] > 0.0) {
        /* M4: ion fast-magnetosonic speed, same formula as the host
           cell_signal_sum's M4 branch. Per-ion quantities:
           a_s^2 = gamma P_eff,s / rho_s,
           b_s^2 = w_s |B|^2 / rho_s,
           b_{s,n}^2 = w_s B_n^2 / rho_s.
           Zero-field branch returns a_s^2 as-is. */
        double w = cfl_cell_w(cp, f, c, ne);
        double b2, bn2, a2, cf2v, sig, bn;
        if (w < 0.0)
          w = 0.0;
        if (w > 1.0)
          w = 1.0;
        b2 = w * (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]) / fn[c];
        bn = (d == 0) ? bc[0] : (d == 1) ? bc[1] : bc[2];
        bn2 = w * bn * bn / fn[c];
        a2 = cp.gamma * puse / fn[c];
        if (b2 <= 0.0)
          cf2v = a2;
        else {
          double ssum = a2 + b2;
          double disc = ssum * ssum - 4.0 * a2 * bn2;
          if (disc < 0.0)
            disc = 0.0;
          cf2v = 0.5 * (ssum + sqrt(disc));
        }
        sig = fabs(fv[c]) + sqrt(cf2v);
        if (sig > s)
          s = sig;
      } else {
        /* Neutrals (and the whole gate-closed path): unchanged. */
        cs = sqrt(cp.gamma * puse / fn[c]);
        if (fabs(fv[c]) + cs > s)
          s = fabs(fv[c]) + cs;
      }
    }
    if (rhoe > 1e-30 && pe_use > 1e-30) {
      /* B3 hard gate: the pseudo sweep's own |u_{e,d}| + c_e. */
      double ue_d = (d == 0) ? ue0 : (d == 1) ? ue1 : ue2;
      double se = fabs(ue_d) + sqrt(cp.gamma * pe_use / rhoe);
      if (se > s)
        s = se;
    }
    s = s > 1e-12 ? s : 1e-12;
    inv += s / cp.dx[d][idx];
  }
  return inv;
}

/* Device-side max-reduction over interior cells. Returns inv_max. */
static double compute_dt_device(const XmfRun *run) {
  xmf::DeviceState *dv = xmf::dev(run);
  CflP cp;
  int d;
  cp.fields = dv->fields.data();
  cp.pe = run->pe ? dv->pe.data() : nullptr;
  for (d = 0; d < 3; d++)
    cp.bf[d] = dv->bf[d].data();
  cp.charge = dv->charge.data();
  cp.mass_code = dv->mass_code.data();
  for (d = 0; d < 3; d++)
    cp.dx[d] = dv->dx[d].data();
  cp.nfluid = run->nfluid;
  cp.ng = run->ng;
  cp.nx = run->nx;
  cp.ny = run->ny;
  cp.nz = run->nz;
  cp.nxtot = run->nxtot;
  cp.nytot = run->nytot;
  cp.nztot = run->nztot;
  cp.ncells = run->ncells;
  cp.gamma = run->gamma;
  cp.ele = elec_closure_active(run);
  cp.mag = run->bf[0] != NULL;
  int ng = cp.ng;
  int k0 = ng, k1 = ng + cp.nz;
  int j0 = ng, j1 = ng + cp.ny;
  int i0 = ng, i1 = ng + cp.nx;
  double inv_max = 0.0;
  Kokkos::parallel_reduce(
      "xmf_cfl_reduce",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k0, j0, i0}, {k1, j1, i1}),
      KOKKOS_LAMBDA(const int k, const int j, const int i, double &lmax) {
        double inv = cfl_cell_signal_sum(cp, i, j, k);
        if (inv > lmax)
          lmax = inv;
      },
      Kokkos::Max<double>(inv_max));
  return inv_max;
}

double compute_dt(const XmfRun *run) {
  double inv_max = 0.0, dt;
  /* Port C: dispatch on device presence. With run->dev the interior cells
     are swept by parallel_reduce over the device Views; without it the
     serial host loop runs (setup before attach, or host-only builds).
     One formula, two paths: the per-cell expression is verbatim the same
     (cell_signal_sum host / cfl_cell_signal_sum device) -- M4
     fast-magnetosonic with B face-averaged to cell center on both. */
  if (run->dev) {
    inv_max = compute_dt_device(run);
  } else {
    int i, j, k;
    for (k = run->ng; k < run->ng + run->nz; k++)
      for (j = run->ng; j < run->ng + run->ny; j++)
        for (i = run->ng; i < run->ng + run->nx; i++) {
          double inv = cell_signal_sum(run, i, j, k);
          if (inv > inv_max)
            inv_max = inv;
        }
  }
  dt = inv_max > 0.0 ? run->cfl / inv_max : 0.0;
  if (run->dt_max > 0.0 && dt > run->dt_max)
    dt = run->dt_max;
  return dt;
}

/* ---- Whole-step operator L_f and SSP RK staging --------------------------
   One call of lf_all_dirs advances the in-place primitive fields by one
   complete forward-Euler increment of the direction-split operator:
   fields_out = U_in + dt*L_f(U_in) with U_in = cons(fields_in). Every
   active direction is swept in turn (ghosts refilled before each
   direction); every fluid is swept along every active direction, and each
   sweep writes only that fluid's own five slots, never another fluid's
   buffer. With the B1 electron closure gate off the P_e block is never
   touched either (A2); with the gate on the per-direction P_e
   pseudo-fluid sweep updates it (pressure increment only). All SSP
   stages call this one whole-step operator -- never a per-direction RK,
   never "rk3 along x1 then sweep x2".

   C2 sync contract: the sweeps run on the device field Views, so every
   active direction performs exactly one roundtrip -- pull the mirror,
   run fill_bc on the host mirror, push it back -- before its sweeps.
   That is the locked frequency (one fill_bc per active direction; RK3
   in 1D is three roundtrips per step). The heat deposit runs as a
   device kernel over the whole cell block: ghost entries of the
   zero-initialised scratch never receive anything, so the != 0 gate
   visits exactly the interior cells the old triple loop visited, in no
   particular order (the deposit is a per-cell += with no cross-cell
   coupling).

   C2sup: the face-flux Views are owned by the sweeps (born and freed
   inside each sweep's Kokkos scope); lf_all_dirs only keeps the
   run-wide heat scratch plumbing. Returns 0, or -1 when a sweep's
   allocation fails -- xmf_step then leaves t untouched.

   The RK stage buffers U0/Us hold the conservative state of every fluid's
   five slots over all cells (ghosts included) plus, when the P_e block
   exists, one trailing P_e segment; SSP convex combinations are taken on
   these conservative states and only the result is converted back to
   primitives -- never a weighted average of (rho, v, P). Both buffers are
   DEVICE allocations (xmf_dev_dvec) handed through the opaque void*
   signatures below; the pure-C xmf_step never dereferences them.

   B1 dispatch: with the electron closure gate on, ion fluids (q_s > 0)
   are swept by the two-pass P_eff operator, neutrals keep the plain
   single-pass sweep (their share is w_s = 0, so P_eff = P_s), and after
   every direction's fluid sweeps the P_e pseudo-fluid sweep updates the
   electron pressure (pressure increment only). With the gate off the loop
   below is bit-for-bit the pre-B1 one.

   M4 dispatch: with the MAGNETIC gate open (bf[0] non-NULL), ion fluids
   are swept by the 8-component MHD operator sweep_dir_mhd instead (the
   two-pass closure yields to it; P_eff enters the MHD state directly),
   neutrals keep the plain HD sweep (w_s = 0 => B_s = 0), and the P_e
   pseudo-sweep still runs afterwards (P_e stays on its B1 pseudo-HD
   track). The induction-equation flux rows are discarded inside the
   MHD sweep; B advances by ONE finite-volume induction in xmf_mag.c
   after lf_all_dirs returns (xmf_step orchestrates). With the gate
   closed the loop is bit-for-bit the pre-M4 one -- in particular
   zero-B gate-open hll stays bit-for-bit the gate-closed HD hll.

   B2 heat plumbing: heat is the one-per-step ncells scratch xmf_step
   allocates for the by-charge/electron tiers (NULL for ions-only or with
   the closure gate off). Each ion sweep accumulates its removed share
   f_s*H per interior cell; after the SAME direction's pseudo-fluid sweep --
   never before it, so the pseudo sweep stays tier-blind and the reversible
   PdV part of P_e stays untouched by the split -- the accumulated share is
   added to P_e of every interior cell that received any and the scratch is
   re-zeroed for the next direction. */

int lf_all_dirs(XmfRun *run, double dt, void *heat) {
  int dir, f;
  int ele = elec_closure_active(run);
  /* M4 magnetic gate: ions (charge > 0) ride the 8-component MHD sweep
     (P_eff + B_s inside, three MHD solvers); neutrals keep their HD
     sweep even under an open gate (their share is w_s = 0, i.e. B_s = 0
     -- the plain HD path). The B1 two-pass closure yields to the MHD
     sweep for ions when the gate is open (P_eff enters the MHD state
     directly); P_e still advances by its own pseudo-HD sweep below.
     Gate closed: the dispatch below is bit-for-bit the pre-M4 one. */
  int mag = run->bf[0] != NULL;
  for (dir = 0; dir < 3; dir++) {
    int nline = (dir == 0) ? run->nx : (dir == 1) ? run->ny : run->nz;
    if (nline <= 1)
      continue; /* degenerate direction: no sweep, no CFL contribution */
    /* Port B: ghost refill via the public dispatch (device path: device
       fill only, no roundtrip; host fallback: pull → fill → push). */
    xmf_fill_bc(run);
    for (f = 0; f < run->nfluid; f++) {
      int err;
      if (mag && run->charge[f] > 0.0)
        err = sweep_dir_mhd(run, f, dir, dt);
      else if (ele && run->charge[f] > 0.0)
        err = sweep_dir_pe(run, f, dir, dt, heat);
      else
        err = sweep_dir(run, f, dir, dt);
      if (err)
        return -1;
    }
    if (ele && sweep_elec_dir(run, dir, dt))
      return -1;
    if (heat) {
      /* Deposit this direction's accumulated non-adiabatic share into P_e
         (only cells an ion sweep handed something to can be non-zero here;
         the accumulation itself is gated on cell_ionized) and re-zero the
         scratch for the next direction. Device kernel: the scratch is
         zero everywhere but the interior cells that accumulated, so the
         gate skips ghosts by value. */
      double *ped = xmf::dev(run)->pe.data();
      double *hd = (double *)heat;
      size_t ntot_cells = run->ncells;
      Kokkos::parallel_for(
          "xmf_hydro_heat_deposit", ntot_cells,
          KOKKOS_LAMBDA(const size_t c) {
            if (hd[c] != 0.0) {
              ped[c] += hd[c];
              hd[c] = 0.0;
            }
          });
    }
    /* Port C: re-pin frozen cells after this direction's sweep (the sweep
       may have modified pinned cells; the next direction must see the
       pinned values). No-op when no pin is loaded. B is NOT pinned --
       the CT path's B is not touched here (extending pin to B is a
       future slice; it would need a hook after mag_induce, not here). */
    if (run->dev)
      xmf_pin_apply_device(run);
    else
      xmf_pin_apply_host(run);
  }
  return 0;
}

/* Whole-block primitive -> conservative conversion. Layout mirrors
   run->fields: [fluid][slot][cell], cells including ghosts. When the P_e
   block exists (B1 staging), one trailing segment copies the electron
   pressure as-is -- pressure-only electrons carry no separate conserved
   energy, so the SSP combination rides P_e itself; with electrons off the
   layout and the numerics are bit-for-bit the pre-B1 ones.
   C2: U is a DEVICE buffer (xmf_dev_dvec); the conversion is a device
   kernel reading the device field Views. */
void fields_to_cons(const XmfRun *run, void *U) {
  const double *fld_d = xmf::dev(run)->fields.data();
  double *U_d = (double *)U;
  size_t ncells = run->ncells;
  int nfluid = run->nfluid;
  double gmm1 = run->gamma - 1.0;
  size_t nwork = (size_t)nfluid * ncells;
  if (!run->bf[0]) {
    /* Magnetic gate closed: bit-for-bit the pre-M4 conversion. */
    Kokkos::parallel_for(
        "xmf_fields_to_cons", nwork, KOKKOS_LAMBDA(const size_t idx) {
          size_t f = idx / ncells;
          size_t c = idx - f * ncells;
          const double *slot0 = fld_d + (f * XMF_SLOT_NVAR) * ncells;
          double *Uf = U_d + (f * XMF_SLOT_NVAR) * ncells;
          double q5[5], U5[5];
          int v;
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            q5[v] = slot0[(size_t)v * ncells + c];
          prim5_to_cons(q5, gmm1, U5);
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            Uf[(size_t)v * ncells + c] = U5[v];
        });
  } else {
    /* M4 gate open: the ion energy rows carry the per-ion magnetic
       share |B_s|^2/2 = w_s |B|^2/2 (w_s clamped to [0,1], the same
       cell-centred B the sweeps feed) -- the staging twin of the
       sweep's conservative energy E_s = P_eff/(gamma-1) + ek +
       |B_s|^2/2 with P_eff = P_s (the sweep adds w_s P_e itself; here
       the combination rides the stored P_s). One share per ion, never
       a summed |B|^2/2. Neutrals are unchanged (w_s = 0). */
    HydroP h = hydro_p(run, 0, 0.0, NULL);
    Kokkos::parallel_for(
        "xmf_fields_to_cons", nwork, KOKKOS_LAMBDA(const size_t idx) {
          size_t f = idx / ncells;
          size_t c = idx - f * ncells;
          const double *slot0 = fld_d + (f * XMF_SLOT_NVAR) * ncells;
          double *Uf = U_d + (f * XMF_SLOT_NVAR) * ncells;
          double q5[5], U5[5];
          int v;
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            q5[v] = slot0[(size_t)v * ncells + c];
          prim5_to_cons(q5, gmm1, U5);
          if (h.charge[f] > 0.0) {
            double ne = hcell_ne(h, c);
            double w = hcell_w(h, f, c, ne);
            double bc[3];
            int i = (int)(c % h.nxtot);
            int j = (int)((c / h.nxtot) % h.nytot);
            int k = (int)(c / (h.nxtot * h.nytot));
            if (w < 0.0)
              w = 0.0;
            if (w > 1.0)
              w = 1.0;
            hcell_b_center(h, k, j, i, bc);
            U5[4] +=
                0.5 * w * (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
          }
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            Uf[(size_t)v * ncells + c] = U5[v];
        });
  }
  if (run->pe) {
    const double *ped = xmf::dev(run)->pe.data();
    double *Upe = U_d + (size_t)nfluid * XMF_SLOT_NVAR * ncells;
    Kokkos::parallel_for(
        "xmf_fields_to_cons_pe", ncells,
        KOKKOS_LAMBDA(const size_t c) { Upe[c] = ped[c]; });
  }
}

/* Whole-block conservative -> primitive conversion (inverse of the above;
   the shared cons5_to_prim floors keep an idle all-zero fluid defined). */
void cons_to_fields(const XmfRun *run, const void *U) {
  double *fld_d = xmf::dev(run)->fields.data();
  const double *U_d = (const double *)U;
  size_t ncells = run->ncells;
  int nfluid = run->nfluid;
  double gmm1 = run->gamma - 1.0;
  size_t nwork = (size_t)nfluid * ncells;
  if (!run->bf[0]) {
    /* Magnetic gate closed: bit-for-bit the pre-M4 conversion. */
    Kokkos::parallel_for(
        "xmf_cons_to_fields", nwork, KOKKOS_LAMBDA(const size_t idx) {
          size_t f = idx / ncells;
          size_t c = idx - f * ncells;
          const double *Uf = U_d + (f * XMF_SLOT_NVAR) * ncells;
          double *slot0 = fld_d + (f * XMF_SLOT_NVAR) * ncells;
          double U5[5], q5[5];
          int v;
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            U5[v] = Uf[(size_t)v * ncells + c];
          cons5_to_prim(U5, gmm1, q5);
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            slot0[(size_t)v * ncells + c] = q5[v];
        });
  } else {
    /* M4 gate open: inverse of fields_to_cons -- subtract the per-ion
       magnetic share |B_s|^2/2 before the pressure recovery (the B of
       the same stage; B is frozen during the sweeps and combined
       beside the fluid state, so the share here is the twin of the
       one fields_to_cons added). */
    HydroP h = hydro_p(run, 0, 0.0, NULL);
    Kokkos::parallel_for(
        "xmf_cons_to_fields", nwork, KOKKOS_LAMBDA(const size_t idx) {
          size_t f = idx / ncells;
          size_t c = idx - f * ncells;
          const double *Uf = U_d + (f * XMF_SLOT_NVAR) * ncells;
          double *slot0 = fld_d + (f * XMF_SLOT_NVAR) * ncells;
          double U5[5], q5[5];
          int v;
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            U5[v] = Uf[(size_t)v * ncells + c];
          if (h.charge[f] > 0.0) {
            /* Charge share from U's OWN conserved density rows, never
               from h.fields: U may hold an SSP-combined state the fields
               buffer does not carry yet, and reading the fields would mix
               two different states into one conversion. With U's rows the
               fields_to_cons / cons_to_fields pair closes bit for bit on
               any ion mix (the pack side reads the very densities this
               unpack recovers). The ionisation gate mirrors hcell_w's,
               applied to U's rows. */
            int g, ionized = 0;
            double ne = 0.0, w;
            for (g = 0; g < nfluid; g++) {
              double rg = U_d[((size_t)g * XMF_SLOT_NVAR) * ncells + c];
              ne += h.charge[g] * rg / h.mass_code[g];
              if (h.charge[g] > 0.0 && rg > XMF_RHO_FLOOR)
                ionized = 1;
            }
            w = (!ionized || ne <= 0.0)
                    ? 0.0
                    : h.charge[f] * U5[0] / h.mass_code[f] / ne;
            double bc[3];
            int i = (int)(c % h.nxtot);
            int j = (int)((c / h.nxtot) % h.nytot);
            int k = (int)(c / (h.nxtot * h.nytot));
            if (w < 0.0)
              w = 0.0;
            if (w > 1.0)
              w = 1.0;
            hcell_b_center(h, k, j, i, bc);
            U5[4] -=
                0.5 * w * (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
          }
          cons5_to_prim(U5, gmm1, q5);
          for (v = 0; v < XMF_SLOT_NVAR; v++)
            slot0[(size_t)v * ncells + c] = q5[v];
        });
  }
  if (run->pe) {
    double *ped = xmf::dev(run)->pe.data();
    const double *Upe = U_d + (size_t)nfluid * XMF_SLOT_NVAR * ncells;
    Kokkos::parallel_for(
        "xmf_cons_to_fields_pe", ncells,
        KOKKOS_LAMBDA(const size_t c) { ped[c] = Upe[c]; });
  }
}

/* U <- a*U0 + b*U, element-wise over every fluid/slot/cell (plus the
   trailing P_e segment when the block exists). This is where the SSP
   convex combinations land -- on the conservative state U (device). */
void cons_combine(const XmfRun *run, void *U, double a, const void *U0,
                  double b) {
  double *U_d = (double *)U;
  const double *U0_d = (const double *)U0;
  size_t n = ((size_t)run->nfluid * XMF_SLOT_NVAR + (run->pe ? 1 : 0)) *
             run->ncells;
  Kokkos::parallel_for(
      "xmf_cons_combine", n,
      KOKKOS_LAMBDA(const size_t i) { U_d[i] = a * U0_d[i] + b * U_d[i]; });
}
