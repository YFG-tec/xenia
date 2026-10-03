/* Internal run-state header of the xmf_core library (A4.5 split).

   Everything the split translation units (xmf_core.c, xmf_mesh.c,
   xmf_state.c, xmf_bc.c, xmf_hydro.c) must share but that is NOT part of
   the public surface lives here: the full XmfRun definition, the per-fluid
   slot offsets, the cell/field indexing helpers and the cross-file internal
   function prototypes.

   The public surface stays xmf_core.h alone (four-step API plus the
   xmf_mesh_* / xmf_state_* / xmf_bc_* / xmf_time / xmf_dt / xmf_ncell
   helper symbols). XmfRun remains opaque out there; nothing in this file
   carries XMF_API and nothing here is ever exported by name. */

#ifndef XMF_RUN_H
#define XMF_RUN_H

#include <stddef.h>

#include "xmf_core.h"
#include "xmf_ir.h"

/* Physical constants locked for the A5 conversion boundary (cgs). The
   kernel only uses them at the density/temperature conversion seams --
   LOAD n reads rho/mass, T reads P*unit_P/(n*kB) -- never anywhere
   else; no formula in the kernel carries a hard-coded physical number
   of its own. */
#define XMF_AMU 1.66053886e-24
#define XMF_KB 1.3806505e-16

/* Per-fluid slot offsets. Layout constants only -- not species/component
   names; the kernel never recognises a component name. Slot 0 stores the
   MASS density rho = n*m (code mass units); the IR boundary converts it
   to the number density n = rho/m on LOAD and writes dn back as
   rho += m*dn. The Godunov sweeps always see rho. */
enum {
  XMF_SLOT_RHO = 0,
  XMF_SLOT_VX = 1,
  XMF_SLOT_VY = 2,
  XMF_SLOT_VZ = 3,
  XMF_SLOT_P = 4,
  XMF_SLOT_NVAR = 5
};

/* Density floor the Godunov cons->prim step applies to rho (code mass
   density; the same literal cons5_to_prim / wavespeeds use). A cell whose
   every ion sits at this floor is numerically vacuum: the B1 closure
   treats it exactly like n_e <= 0 (w_s = 0, u_e = 0, P_eff = P_s, P_e
   frozen), so an emptied cell never couples to nor advances the electron
   pressure. Gating on the stored rho (not on n_e = sum q_s rho_s /
   mass_code, which the per-fluid mass_code rescales) keeps the vacuum
   test unit-independent. M4: the induction's u_+ assembly shares the
   same gate. */
#define XMF_RHO_FLOOR 1.0e-30

/* Radiation scan state (A5, RadiationScan mode 1). The beam table
   (sigma/hnu/finf, one entry per beam) and the transmitted flux arrays
   F_b arrive through the xmf_rad_load helper -- never through the flat
   XmfConfig. flux/dflux are laid out [beam][cell] over the full cell
   block (ghosts included); only interior cells are filled by the scan.
   mode/dilution/sense keep the enum room of the card: this slice
   implements mode == grid_line, dilution == none, sense == inward only;
   anything else is refused at load time (setup then fails). */
typedef struct {
  int enabled;   /* 0: FLUX slots read the boundary flux finf (vacuum) */
  int mode;      /* 1 = grid_line (ray = cells + mesh ds) */
  int dilution;  /* 0 = none */
  int sense;     /* 0 = inward: F_infty enters at the x1end outer face */
  int absorber;  /* fluid index whose number density sets the opacity */
  int nbeam;
  double *sigma; /* [nbeam] photo cross sections, cm^2 */
  double *hnu;   /* [nbeam] photon energies, erg */
  double *finf;  /* [nbeam] flux at the outer boundary, erg/cm^2/s */
  double *flux;  /* [nbeam][ncells] transmitted F_b after each cell */
  double *dflux; /* [nbeam][ncells] absorbed increment F_in*(1-e^-tau) */
  /* B3 multi-absorber state, APPENDED auxiliary port
     (xmf_rad_load_absorbers). NULL/0 while only the single absorber
     above is loaded -- the scan then runs the single-absorber walk bit
     for bit as before. When present (nabs >= 2) the per-beam optical
     depth is tau_b = (sum_s sigma_multi[s*nbeam+b] * n_{absorbers[s]})
     * ds; absorber/sigma above keep the flat-port contract (first
     absorber's row) but no longer set the scan. */
  int nabs;
  int *absorbers;    /* [nabs] fluid indices, NULL when nabs == 0 */
  double *sigma_multi; /* [nabs*nbeam] row-major, NULL when nabs == 0 */
} XmfRadiation;

/* One direction's mesh: two point sets (centres x, faces xf) plus three
   metrics (dx, face area A, volume factor dV). 1-D authority only. */
typedef struct {
  int ntot;   /* cells including ghosts: n_d + 2*ng */
  double *x;  /* centres, length ntot */
  double *xf; /* faces, length ntot+1; xl(i)=xf[i], xr(i)=xf[i+1] */
  double *dx; /* widths, length ntot */
  double *A;  /* face-area factor, length ntot+1, at the face coordinate */
  double *dV; /* volume factor, length ntot */
} XmfAxis;

struct XmfRun {
  int nx, ny, nz;             /* interior cell counts (n[0],n[1],n[2]) */
  int nxtot, nytot, nztot;    /* with ghosts: n_d + 2*ng */
  int ng, nfluid, geometry;
  int bc[6];                  /* per-face boundary types (enum XmfBcType),
                                 face order 0=x1beg..5=x3end; copied from
                                 XmfConfig.bc at setup, never re-read from
                                 the config blob afterwards */
  size_t ncells;              /* nxtot*nytot*nztot */
  /* No scalar grid spacings live here (A1 stretch supplement): the mesh[]
     face/width arrays below are the ONLY grid authority. CFL and every
     sweep read mesh[d].dx / A / dV cell by cell -- never a run-wide dx. */
  double gamma, cfl, t, dt;
  int riemann;                /* enum XmfRiemann: hll=0, hllc=1, tvdlf=2,
                                 hlld=3 (M4 MHD solver; the magnetic gate
                                 refuses hllc at xmf_mag_load time) */
  int recon;                  /* enum XmfRecon: pcm=0, plm=1, weno3=2.
                                 sweep_line dispatches on it for every face;
                                 plm/weno3 need the two ghost layers setup
                                 enforces; nothing unknown is ever silently
                                 run as PCM (setup refuses it instead) */
  int rk;                     /* enum XmfRK: rk1=0, rk2=1, rk3=2; rk4 is
                                 refused at setup, never stored here */
  int shock_heat;             /* enum XmfShockHeat (B2): ions-only=0,
                                 by-charge=1, electron=2. Copied from the
                                 former reserved0 slot of the flat blob
                                 (byte position frozen, name moved) and
                                 validated to 0..2 at setup. Read only by
                                 xmf_hydro.c when the electron closure gate
                                 is on; with the gate off it is inert and
                                 every path stays the pre-B1 one. */
  double dt_max;              /* > 0 caps dt from above; <= 0 means no cap */
  XmfAxis mesh[3];            /* one 1-D mesh per direction */
  double *fields;             /* [nfluid][XMF_SLOT_NVAR][ncells] */
  double *pe;                 /* optional electron pressure P_e, length
                                 ncells (ghosts included). NULL when
                                 electron_mode == XMF_ELEC_OFF. Separate
                                 energy slot -- never a fluid, never
                                 counted in nfluid */
  XmfTape *tapes;             /* source tapes (A4.6), loaded through the
                                 xmf_ir_load helper after setup; NULL/0
                                 while no set is loaded. Empty sources
                                 therefore never touch the pure-hydro
                                 path. Owned by the run, freed by
                                 xmf_finalize via xmf_ir_clear */
  int ntape;
  /* ---- A5 physics block ---------------------------------------------
     Per-fluid masses and the unit triplet, loaded through the
     xmf_phys_load helper (same helper level as xmf_ir_load -- never
     part of the flat XmfConfig). mass_code[f] is the mass per particle
     in code mass-density units: mass_amu[f]*XMF_AMU/unit_density, so
     the stored slot-0 rho converts to the IR number density via
     n = rho/mass_code. Defaults (no xmf_phys_load call) are 1.0 for
     every mass_code and 1.0 for every unit, which keeps the pre-A5
     tests bit-for-bit: LOAD n returns slot 0 unchanged and its seeded
     derivative stays exactly 1. */
  double *mass_code;          /* [nfluid] */
  double unit_density, unit_velocity, unit_length, unit_P;
  /* ---- B1 charge table -----------------------------------------------
     Per-fluid charges q_s in elementary-charge multiples, loaded through
     the xmf_charge_load helper (auxiliary load port at the same level as
     xmf_phys_load -- never part of the flat XmfConfig). The kernel's ion
     role is q_s != 0; species names are never consulted. Defaults (no
     xmf_charge_load call) are 0.0 for every fluid, so the electron
     closure gate stays off and the hydro path stays bit-for-bit the
     pre-B1 one. */
  double *charge;             /* [nfluid] */
  XmfRadiation *rad;          /* NULL until xmf_rad_load succeeds */
  /* User boundary floor (A5). Fixed base state in code-unit slots,
     [nfluid][XMF_SLOT_NVAR][nztot*nytot] (transverse index k*nytot+j).
     Every fill of an x1 user face writes it back to that side's ghost
     layers AND the innermost active cell; no function pointers, no
     per-cell callbacks. NULL -> a user face falls back to copy (the
     pre-A5 padding semantics), so old tests stay green. */
  double *user_bc;
  /* Analytic user boundary tapes (A3 supplement): per-face assignment
     tape sets loaded through xmf_bc_load_tapes. Each tape writes ONE
     fluid slot of EVERY ghost cell of its face (assignment, not
     increment); slots without a tape fall back to copy @1. Stored per
     face and NEVER merged into run->tapes -- the source set feeding
     xmf_ir_apply / xmf_ir_jac / the implicit source solve stays exactly
     the loaded sources. Opcodes and packed slots follow xmf_ir.h;
     neighbour reads carry the XMF_IR_NB kind and resolve face-relative
     (@1/@2 = the two innermost active layers). NULL/0 while a face has
     no tapes -- the face then keeps the floor/copy semantics. Owned by
     the run, freed by xmf_finalize via xmf_bc_clear_tapes. */
  XmfTape *bc_tapes[6];
  int nbc_tape[6];
  /* ---- M block 1 magnetic state (M2 staggered storage, M3 B ghosts) ---
     Face-centred B (bf) and edge-centred E (ef), one block per component
     0/1/2, loaded through the xmf_mag_load auxiliary port -- never the
     flat XmfConfig. NULL bf[0] is the magnetic GATE CLOSED state: every
     mag helper refuses (-1) or no-ops, the fluid five slots stay bit for
     bit the pre-M path and no B key appears in the Python read port.
     Layouts (i_d fastest, (k,j,i) convention; d1=(d+1)%3, d2=(d+2)%3):
       bf[d]: length (ntot[d]+1)*ntot[d1]*ntot[d2]; index
              ((c_d2*ntot[d1]) + c_d1)*(ntot[d]+1) + f_d with f_d the
              FACE index along d and c the CELL indices along d1/d2;
       ef[d]: length ntot[d]*(ntot[d1]+1)*(ntot[d2]+1); index
              ((e_d2*(ntot[d1]+1)) + e_d1)*ntot[d] + c_d with e the EDGE
              indices along d1/d2 and c_d the cell index along d.
     Block 1 semantics (superseded by M4): B was a frozen bystander.
     M4: with the gate open the ion sweeps run the 8-component MHD
     Riemann solvers (hll / hlld / tvdlf; hllc is refused at load time)
     fed B_s = sqrt(w_s) B, and after every direction sweep one
     finite-volume induction advances B with
     d_t B = curl(u_+ x B). M5 replaces the finite-volume induction
     with constrained transport: every stage writes edge-centred
     E = -u_+ x B into the E blocks and advances B by the discrete curl
     of E. The E blocks carry physical values from M5 on. Ghost fills register
     in bc_fill_aux_blocks (M3): reflect follows the mag_reflect tier
     (wall / symmetric), user faces prescribe the tangential components
     (xmf_mag_set_user floor or copy @1) and reconstruct the normal one
     from the discrete divB = 0 constraint. mag_reflect is enum
     XmfMagReflect; mag_user_bc is the x1-face tangential floor
     [tcomp][k*nytot+j] (tcomp 0=B2, 1=B3), NULL until loaded, owned by
     the run. */
  double *bf[3];
  double *ef[3];
  int mag_reflect;
  double *mag_user_bc;
  /* M4: charge-weighted ion velocity u_+ = sum q_s n_s u_s / sum q_s n_s
     snapshot (three component blocks of ncells each, ghosts included),
     rebuilt from the STAGE-START ion fields by mag_up_snapshot before
     every lf_all_dirs and consumed by the one-shot induction mag_induce
     after it. Host mirror buffer; allocated by xmf_mag_load with the
     gate, freed by xmf_mag_clear. NULL while the gate is closed. */
  double *mag_up;
  /* Port C: frozen-cell pin mask and values. pin_mask is ncells ints
     (0/1, ghosts included); pin_vals is [nfluid*XMF_SLOT_NVAR*ncells]
     doubles (same flat layout as run->fields). Loaded through
     xmf_bc_set_pin (auxiliary load port, NOT a fifth public verb).
     The step path applies them automatically: after every ghost fill
     in xmf_fill_bc and after every active direction's sweep in
     lf_all_dirs. P_e is never touched; B is never touched (the
     magnetic gate's CT path is NOT pinned -- extending pin to B is a
     future slice, not this one). NULL/NULL while no pin is loaded.
     Owned by the run, freed by xmf_finalize. */
  int *pin_mask;
  double *pin_vals;
  /* Largest relative one-step residual of the linearised implicit
     source solve over the last step (0 when no source ran). Report-only;
     the solve itself never iterates on it. */
  double source_resid;
  /* C2: opaque device state (owned by xmf_device.cpp). NULL until
     xmf_dev_attach completes at the end of setup. Host C code never
     dereferences it -- every device touch goes through the xmf_dev_*
     hooks below (SyncToDevice / SyncFromDevice style). The host blocks
     above (fields / pe / mass_code / charge / rad arrays / tapes) stay
     the host mirrors of the device Views. */
  void *dev;
};

/* Linear index of cell (k,j,i); i fastest. */
static inline size_t cidx(const XmfRun *run, int k, int j, int i) {
  return (((size_t)k * (size_t)run->nytot) + (size_t)j) * (size_t)run->nxtot +
         (size_t)i;
}

/* Pointer to the full 3-D sub-array of fluid f, slot var. */
static inline double *fld(const XmfRun *run, int f, int var) {
  return run->fields +
         (((size_t)f * XMF_SLOT_NVAR + (size_t)var) * run->ncells);
}

/* ---- Internal cross-file helpers (NOT public API) -------------------------
   No XMF_API here and nothing ever enters xmf_core.h: these are the seams
   between the split translation units only. C linkage throughout: the C++
   loop modules (C1a) define symbols the pure-C xmf_core.c calls. */

#ifdef __cplusplus
extern "C" {
#endif

/* Zeroed double vector allocator (xmf_mesh.c); calloc(n ? n : 1, ...). */
double *dvec(size_t n);

/* Per-direction mesh construction / allocation (xmf_mesh.c). */
int axis_alloc(XmfAxis *ax, int ntot);
int axis_fill_faces(XmfAxis *ax, int n, int ng, const double *xf_active);
void axis_fill_metrics(XmfAxis *ax, int geometry, int dir);

/* Per-face ghost fill over all six faces (xmf_bc.c). */
void fill_bc(XmfRun *run);

/* Port C: frozen-cell pin apply (xmf_bc.c). Host path iterates every
   cell and writes pin_vals where pin_mask is set; device path does the
   same on the device Views. Both are no-ops when no pin is loaded
   (pin_mask == NULL). P_e is never touched; B is never touched. */
void xmf_pin_apply_host(XmfRun *run);
void xmf_pin_apply_device(XmfRun *run);

/* Port C: push the host pin mask/vals to the device Views (xmf_device.cpp).
   Called by xmf_bc_set_pin after the host copy has landed in
   run->pin_mask / run->pin_vals. No-op when no device is attached or
   no pin is loaded (extent 0 marks "not loaded" on the device). Returns
   0, or -1 on invalid arguments / allocation failure. */
int xmf_dev_pin_sync(XmfRun *run);

/* Free all six per-face boundary tape sets (NULL-safe; xmf_bc.c).
   Afterwards every bc_tapes[face] is NULL and nbc_tape[face] is 0. */
void xmf_bc_clear_tapes(XmfRun *run);

/* ---- M block 1 magnetic seams (xmf_mag.c) --------------------------------
   Block lengths of the staggered B/E blocks (the XmfRun layout comment
   carries the index formulas). Used by xmf_mag.c itself and the device
   owner (View extents / mirror transfers). */
size_t xmf_mag_len_b(const XmfRun *run, int comp);
size_t xmf_mag_len_e(const XmfRun *run, int comp);

/* Ghost fill of one face of the staggered B blocks (M3), registered in
   bc_fill_aux_blocks -- the single point every auxiliary block goes
   through. reflect != 0 mirrors per the mag_reflect tier; a user face
   prescribes the tangential components and back-solves the normal one
   from discrete divB = 0; copy/outflow/none pad zero-order. No-op when
   the magnetic gate is closed (bf[0] NULL). */
void mag_fill_face(XmfRun *run, int face, int reflect);

/* Free the magnetic blocks and the tangential floor (NULL-safe); every
   pointer is NULL afterwards (gate closed). Used by xmf_finalize. */
void xmf_mag_clear(XmfRun *run);

/* ---- M4/M5 induction seams (xmf_mag.c) -----------------------------------
   M4: one-shot finite-volume induction (superseded by M5).
   M5: constrained transport (CT) -- edge-centred E = -u_+ x B assembled
   every stage, face-centred B advanced by the discrete curl of E.
   mag_up_snapshot rebuilds run->mag_up from the stage-start ion fields
   (the u_e formula, ions only); mag_induce assembles E and advances B
   (CT replaces the M4 finite-volume path; same signature and position
   in the orchestration). Both are no-ops while the magnetic gate is
   closed. mag_induce returns 0, or -1 on allocation failure.
   mag_divb_max returns the maximum normalised discrete divB over the
   interior cells (C-IDF-4 area form, dimensionless, report-only -- the
   threshold 1e-6 is enforced in the test suite, never here). The B SSP
   stage buffers (separate from the fluid U0/Us, same Heun / Shu-Osher
   coefficients): alloc/free own three face-centred device blocks via
   xmf_dev_dvec (never allocated gate closed); snapshot copies bf -> dst
   beside fields_to_cons(U0); combine applies bf <- a*src0 + b*bf beside
   every cons_combine. */
void mag_up_snapshot(XmfRun *run);
int mag_induce(XmfRun *run, double dt);
double xmf_divb_max(const XmfRun *run);
int mag_b_stage_alloc(const XmfRun *run, void **buf);
void mag_b_stage_free(void **buf);
int mag_b_snapshot(const XmfRun *run, void **dst);
int mag_b_combine(const XmfRun *run, double a, void **src0, double b);

/* CFL time step and the whole-step hydro operator pieces (xmf_hydro.cpp).
   C1a: the line workspaces (primitive gather q, pass-input/pseudo-fluid
   qe, conservative scratch Uc, scatter Un) moved into the sweep itself --
   one row per transverse line, allocated inside the Kokkos loop scope and
   freed before it returns, so concurrent lines never share state.
   lf_all_dirs returns 0, or -1 when a sweep's workspace allocation fails.
   heat (B2 shock-heating share) is still one ncells-long scratch buffer
   owned by xmf_step that accumulates the per-direction non-adiabatic
   pressure increment share f_s*H destined for P_e; NULL exactly when the
   share is ions-only or the closure gate is off, in which case every
   sweep is bit-for-bit the B1 path and P_e receives only its own
   pseudo-sweep increment.

   C2: the sweeps, the heat deposit and the three conservative-state
   helpers run as device kernels over the device Views; the stage buffers
   U0/Us and heat are device allocations handed through as opaque void*
   (xmf_dev_dvec / xmf_dev_dfree in the hooks below), so the pure-C
   xmf_core.c orchestration never touches a device pointer itself.
   compute_dt stays a serial HOST loop over the freshly pulled mirror
   (xmf_step pulls before calling it), keeping the max order bit for bit.
   lf_all_dirs keeps the locked sync frequency: one device<->mirror
   roundtrip per ACTIVE direction (pull, fill_bc on the mirror, push). */
double compute_dt(const XmfRun *run);
int lf_all_dirs(XmfRun *run, double dt, void *heat);
void fields_to_cons(const XmfRun *run, void *U);
void cons_to_fields(const XmfRun *run, const void *U);
void cons_combine(const XmfRun *run, void *U, double a, const void *U0,
                  double b);

/* B1 electron-closure gate (xmf_hydro.c): true exactly when the optional
   P_e block exists AND at least one fluid carries a loaded non-zero
   charge q_s. Everything the B1 closure adds -- the P_eff sharing in the
   ion sweeps, the P_e pseudo-sweep, the P_e ghost fill -- hangs off this
   gate; with it off the hydro path stays bit-for-bit the pre-B1 one (the
   empty-table pcm Sod pins). */
int elec_closure_active(const XmfRun *run);

/* Radiation scan (xmf_scan.c, A5): fills run->rad->flux/dflux for the
   interior cells from the current absorber density. No-op when no
   radiation state is loaded or it is disabled. xmf_rad_clear frees the
   beam state (NULL-safe, used by xmf_finalize). */
void xmf_rad_scan(XmfRun *run);
void xmf_rad_clear(XmfRun *run);

/* Implicit source step (xmf_source.c, A5): advances the in-place fields
   by one semi-implicit source increment U^{n+1}-U* = dt*R(U^{n+1}),
   R = T(w, L_c), solved by ONE linearised backward-Euler step with a
   single dense N x N LU, N = xmf_ir_nstate(run) (B2: 5*nfluid plus one
   trailing P_e unknown when the electron block exists; the P_e rate
   enters as-is, pressure-only). No-op returning 0 when no source tape
   is loaded (the pure-hydro path stays bit-for-bit). Returns 0, or -1
   on invalid arguments / tape-evaluation failure. */
int xmf_source_step(XmfRun *run, double dt);

/* ---- C2 device-state hooks (xmf_device.cpp) ------------------------------
   The fields / P_e / read-only tables live as Kokkos device Views with
   the host blocks of XmfRun as their mirrors (Idefix SyncToDevice /
   SyncFromDevice style). Host C (bc / state / ir) only ever touches the
   mirrors and calls these hooks at the sync points; device kernels only
   ever touch the Views. Every hook is a no-op when run->dev == NULL
   (before attach / after detach), so load ports called on a bare run
   stay valid. Attach happens once at the end of xmf_setup, after the
   initial fill_bc + compute_dt; detach is xmf_finalize's first act. */

/* Allocate the device state (fields / pe / mass_code / charge / mesh
   arrays) and push the current mirrors. Returns 0, or -1 on allocation
   failure (setup then fails). Idempotent. */
int xmf_dev_attach(XmfRun *run);

/* Free the device state (NULL-safe); run->dev is NULL afterwards.
   Never calls Kokkos::finalize. */
void xmf_dev_detach(XmfRun *run);

/* Fields + P_e mirror -> device / device -> mirror. */
void xmf_dev_push(const XmfRun *run);
void xmf_dev_pull(const XmfRun *run);

/* Single fluid/slot block or the P_e block, both directions. */
void xmf_dev_push_field(const XmfRun *run, int f, int var);
void xmf_dev_pull_field(const XmfRun *run, int f, int var);
void xmf_dev_push_pe(const XmfRun *run);
void xmf_dev_pull_pe(const XmfRun *run);

/* Read-only tables: mass_code + charge mirror -> device. */
void xmf_dev_push_tables(const XmfRun *run);

/* Transmitted beam flux device -> mirror (xmf_rad_flux_copy and the
   host IR readers need it after a device scan). */
void xmf_dev_pull_flux(const XmfRun *run);

/* (Re)allocate every device radiation View from run->rad and push the
   current host content (flux/dflux arrive zero at xmf_rad_load time).
   Called by xmf_rad_load. Returns 0, or -1 on allocation failure. */
int xmf_dev_rad_alloc(XmfRun *run);

/* (Re)allocate + push only the B3 multi-absorber device Views
   (sigma_multi / absorbers); flux/dflux are left untouched. Called by
   xmf_rad_load_absorbers. Returns 0, or -1 on allocation failure. */
int xmf_dev_rad_sync_absorbers(XmfRun *run);

/* Rebuild the flattened device tape set (ops / consts / descriptors)
   from run->tapes. Called by xmf_ir_load. Returns 0, or -1 on
   allocation failure. */
int xmf_dev_tapes_sync(XmfRun *run);

/* Magnetic-block device mirrors (M block 1). xmf_dev_mag_alloc
   (re)allocates the six bf/ef Views from the host blocks and pushes the
   current host content; called by xmf_mag_load (returns 0 when no
   device is attached -- host-only stage, the rad_alloc precedent). The
   push/pull pair syncs all six blocks; xmf_dev_push / xmf_dev_pull call
   them whenever the gate is open, so the locked fluid sync points cover
   the magnetic mirrors too. No-ops when the gate is closed or no device
   is attached. */
int xmf_dev_mag_alloc(XmfRun *run);
void xmf_dev_push_b(const XmfRun *run);
void xmf_dev_pull_b(const XmfRun *run);

/* M4: freeze the stage-start electron density n_e into the device
   ne_stage snapshot (xmf_dev_mag_alloc allocates it beside the magnetic
   blocks). Called at every RK stage start, before lf_all_dirs, while the
   gate is open -- the ion sweeps' charge shares then all read the same
   stage-start n_e. No-op returning 0 when the gate is closed or no
   device is attached; -1 only on a launch failure. */
int xmf_dev_ne_stage_fill(const XmfRun *run);

/* Port D: device-side induction path. xmf_dev_mag_up_snapshot fills
   dv->mag_up from dv->fields (the charge-weighted ion velocity u_+ with
   the same formula, density floor, ionized fallback and degenerate-axis
   active-slice remap as the host mag_up_snapshot). xmf_dev_mag_induce
   performs the full CT induction on device Views: cell-centred B from
   upper-face averages, G = u_+ x B, edge E = -G (4-cell average with
   periodic wrapping), discrete curl update of dv->bf, and the trailing
   xmf_bc_fill_e_ghosts_device call. Both are no-ops returning 0 when the
   gate is closed or no device is attached; -1 only on a launch failure. */
int xmf_dev_mag_up_snapshot(const XmfRun *run);
int xmf_dev_mag_induce(XmfRun *run, double dt);

/* M4 B SSP stage hooks: snapshot deep-copies the three device bf Views
   into the stage buffers; combine applies bf <- a*src0 + b*bf on the
   device (the fluid cons_combine never touches the magnetic blocks).
   Both return 0, or -1 on invalid arguments; no-ops returning 0 when no
   device is attached (host-only stage) or the gate is closed. */
int xmf_dev_mag_stage_snapshot(const XmfRun *run, void **dst);
int xmf_dev_mag_stage_combine(const XmfRun *run, double a, void **src0,
                              double b);

/* Zeroed device double vector (RK stage buffers U0/Us and the heat
   scratch): the device-side dvec. Freed with xmf_dev_dfree. */
void *xmf_dev_dvec(size_t n);
void xmf_dev_dfree(void *p);

/* ---- Port B: device-side BC state & skin fill -----------------------------
   xmf_dev_bc_sync pushes the user-BC floor, per-face BC tapes and the
   magnetic tangential floor to the device Views (called after load ports
   and at setup). xmf_bc_fill_device fills the fluid + P_e + B ghost skin
   on the device (no host roundtrip); called from xmf_fill_bc and
   lf_all_dirs when run->dev is attached. xmf_bc_fill_e_ghosts_device
   fills the non-periodic E ghost edges on device; called from mag_induce
   after the interior E fill and push. All three are no-ops when no device
   is attached. xmf_dev_bc_sync returns 0, or -1 on invalid arguments /
   allocation failure. */
int xmf_dev_bc_sync(XmfRun *run);
void xmf_bc_fill_device(XmfRun *run);
void xmf_bc_fill_e_ghosts_device(XmfRun *run);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
