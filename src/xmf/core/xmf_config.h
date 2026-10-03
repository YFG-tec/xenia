#ifndef XMF_CONFIG_H
#define XMF_CONFIG_H

#include <stdint.h>

#define XMF_SCHEMA 1

enum XmfGeometry { XMF_GEOM_CARTESIAN = 0, XMF_GEOM_SPHERICAL = 1, XMF_GEOM_CYLINDRICAL = 2 };
enum XmfBackend { XMF_BACKEND_SERIAL = 0, XMF_BACKEND_HOST = 1, XMF_BACKEND_CUDA = 2 };
enum XmfElectron { XMF_ELEC_OFF = 0, XMF_ELEC_PRESSURE_ONLY = 1 };
/* Riemann solvers (A4): four are supported -- hll, hllc, tvdlf (local
   Lax-Friedrichs / Rusanov) and hlld. No Roe / exact / two-shock; the
   kernel dispatches on this integer only, never on names. M4: hlld is
   the Miyoshi-Kusano MHD solver -- xmf_setup accepts it; with the
   magnetic gate open the ion sweeps dispatch to the 8-component MHD
   branch, with the gate closed it runs the zero-field limit of the very
   same solver (never a silent fallback to another solver). The magnetic
   gate itself refuses hllc at xmf_mag_load time: hllc's wave fan
   carries no magnetic waves and is never silently swapped. */
enum XmfRiemann {
  XMF_RIEMANN_HLL = 0,
  XMF_RIEMANN_HLLC = 1,
  XMF_RIEMANN_TVDLF = 2,
  XMF_RIEMANN_HLLD = 3
};
enum XmfRecon { XMF_RECON_PCM = 0, XMF_RECON_PLM = 1, XMF_RECON_WENO3 = 2 };
/* SSP Runge-Kutta time integrators (A4 supplement): rk1 (forward Euler),
   rk2 (Heun) and rk3 (Shu-Osher) are implemented; every stage calls the
   whole-step direction-split operator L_f, never a per-direction RK.
   rk4 is enumerated so configs can name it, but the kernel does NOT
   implement it -- xmf_setup refuses it (no silent fallback to rk1).
   Values must match the Python-side table config.RK. */
enum XmfRK { XMF_RK1 = 0, XMF_RK2 = 1, XMF_RK3 = 2, XMF_RK4 = 3 };
enum XmfStretch { XMF_STRETCH_NONE = 0, XMF_STRETCH_X1 = 1 };
/* Shock-heating share (B2): how the Godunov sweep's non-adiabatic
   (entropy-producing) pressure increment is split between the ion
   pressure P_s and the electron pressure P_e. Tóth's warning is the
   contract: T_e/T_i is NOT unique at a shock, so none of the three is
   physical truth -- they are switchable model options the tube tests
   contrast. ions-only keeps every increment in P_s (the B1 two-pass
   path, bit for bit); by-charge hands the charge share w_s of each
   ion's increment to P_e; electron hands the whole increment to P_e.
   Values must match the Python-side table config.SHOCK_HEAT. The
   integer occupies the former reserved0 slot of the flat blob: the
   byte position is frozen, only the name moved. */
enum XmfShockHeat {
  XMF_SHOCK_HEAT_IONS_ONLY = 0,
  XMF_SHOCK_HEAT_BY_CHARGE = 1,
  XMF_SHOCK_HEAT_ELECTRON = 2
};

/* Per-face boundary condition types. Values must match the Python-side
   lookup config.BC_ENUM, which is derived from config.BC_TYPES (same
   order); the kernel only ever sees these integers, never names. */
enum XmfBcType {
  XMF_BC_REFLECT = 0,
  XMF_BC_COPY = 1,
  XMF_BC_OUTFLOW = 2,
  XMF_BC_USER = 3,
  XMF_BC_NONE = 4,
  XMF_BC_PERIODIC = 5
};

/* Magnetic reflect tier (M block 1, M3): how a reflect face mirrors the
   face-centred B block. wall (PLUTO REFLECTIVE / AthenaK): the normal
   component flips sign and the wall face itself is pinned to 0 -- the
   conducting-wall image; symmetric (PLUTO EQTSYMMETRIC / Idefix): the
   tangential components flip (axial-vector mirroring) and the normal one
   copies, the wall face untouched. The fluid five slots use the very same
   reflect fill in both tiers (normal velocity flips); only the magnetic
   block reads this selector. It rides the auxiliary xmf_mag_load port --
   never the flat blob, whose byte layout stays frozen. Values must match
   the Python-side table config.MHD_REFLECT. */
enum XmfMagReflect { XMF_MAG_REFLECT_WALL = 0, XMF_MAG_REFLECT_SYMMETRIC = 1 };

/* Flat, versioned blob. Named fields only — no g_inputParam[GAMMA] indices.
   Layout: 22×int32 (88 bytes) then doubles. Keep in sync with Python ctypes. */
typedef struct XmfConfig {
  int32_t schema_version;
  int32_t ndim;
  int32_t n[3];
  int32_t ng;
  int32_t geometry;
  int32_t nfluid;
  int32_t sources_enabled;
  int32_t electron_mode;
  int32_t backend;
  int32_t riemann;
  int32_t recon;
  int32_t rk;
  int32_t stretch;
  /* Shock-heating share tier (enum XmfShockHeat). This field used to be
     the unused reserved0 slot; the byte position is FROZEN (no field may
     be inserted after electron_mode and shift the blob), only the name
     changed. Validated at setup to 0..2. */
  int32_t shock_heat;
  int32_t bc[6]; /* per-face types (enum XmfBcType); face order is fixed:
                    0=x1beg, 1=x1end, 2=x2beg, 3=x2end, 4=x3beg, 5=x3end */
  double x1min, x1max, x2min, x2max, x3min, x3max;
  double gamma;
  double cfl;
  double dt_max;
  double tstop;
  double x_disc;
  double rho_l, v_l, p_l;
  double rho_r, v_r, p_r;
} XmfConfig;

#endif
