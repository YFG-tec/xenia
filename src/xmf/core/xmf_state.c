/* Multi-fluid state queries and the optional electron-pressure block of the
   ideal-Euler kernel (A4.5 split out of xmf_core.c; the public surface
   stays xmf_core.h).

   Optional electron pressure (A2): when electron_mode ==
   XMF_ELEC_PRESSURE_ONLY one extra block P_e, ncells doubles long (ghosts
   included, same length as every fluid slot), is allocated separately and
   calloc'd to zero. Electrons are not a fluid: P_e is never a sixth slot
   and never counted in nfluid. When electron_mode == XMF_ELEC_OFF the
   block is not allocated at all (pointer stays NULL) and the P_e helpers
   refuse to touch it. */
#include "xmf_run.h"

#include <math.h>
#include <string.h>

/* ---- Multi-fluid state queries (A2, helper style like xmf_mesh_copy_*) ---
   Fields are laid out [fluid][slot][cell], slot order (n, vx, vy, vz, P)
   == offsets 0..4 (layout constants above, not component names); fluids
   are addressed only by integer index f in [0, nfluid). Every fluid/slot
   block is ncells = nxtot*nytot*nztot doubles long, ghosts included --
   this is NOT the interior x1 count returned by xmf_ncell.

   xmf_state_pack copies one whole block from the caller buffer INTO the
   kernel; xmf_state_unpack copies one whole block OUT of the kernel into
   the caller buffer. Writing one slot never touches any other slot. The
   optional electron pressure P_e (electron_mode == pressure-only) is one
   separate block of the same length: xmf_state_copy_pe copies it out,
   xmf_state_write_pe copies into it. When electrons are off there is no
   P_e block; the P_e helpers then return -1 and touch nothing.

   All functions below return the number of doubles moved, or -1 (moving
   nothing) on invalid arguments. */

int xmf_state_ncells(const XmfRun *run) {
  if (!run)
    return 0;
  return (int)run->ncells; /* nxtot*nytot*nztot, ghosts included */
}

int xmf_state_nfluid(const XmfRun *run) {
  if (!run)
    return 0;
  return run->nfluid;
}

int xmf_state_pack(XmfRun *run, int f, int var, const double *in) {
  if (!run || !in || f < 0 || f >= run->nfluid || var < 0 ||
      var >= XMF_SLOT_NVAR)
    return -1;
  memcpy(fld(run, f, var), in, run->ncells * sizeof(double));
  xmf_dev_push_field(run, f, var); /* C2 sync point: mirror -> device */
  return (int)run->ncells;
}

int xmf_state_unpack(const XmfRun *run, int f, int var, double *out) {
  if (!run || !out || f < 0 || f >= run->nfluid || var < 0 ||
      var >= XMF_SLOT_NVAR)
    return -1;
  xmf_dev_pull_field(run, f, var); /* C2 sync point: device -> mirror */
  memcpy(out, fld(run, f, var), run->ncells * sizeof(double));
  return (int)run->ncells;
}

int xmf_state_copy_pe(const XmfRun *run, double *out) {
  if (!run || !out || !run->pe) /* electrons off: no P_e block exists */
    return -1;
  xmf_dev_pull_pe(run); /* C2 sync point: device -> mirror */
  memcpy(out, run->pe, run->ncells * sizeof(double));
  return (int)run->ncells;
}

int xmf_state_write_pe(XmfRun *run, const double *in) {
  if (!run || !in || !run->pe) /* electrons off: no P_e block exists */
    return -1;
  memcpy(run->pe, in, run->ncells * sizeof(double));
  xmf_dev_push_pe(run); /* C2 sync point: mirror -> device */
  return (int)run->ncells;
}

/* Physics constants + per-fluid masses (A5), auxiliary load port -- see
   xmf_core.h. Converts the amu masses to code mass density per number
   density and records the unit triplet (unit_P = density*velocity^2 for
   the T closure). mass_code/unit defaults are set in xmf_setup to 1.0,
   so a run that never calls this port keeps the pre-A5 numerics. */
int xmf_phys_load(XmfRun *run, const double *mass_amu, double unit_density,
                  double unit_velocity, double unit_length) {
  int f;
  if (!run || !mass_amu)
    return -1;
  if (unit_density <= 0.0 || unit_velocity <= 0.0 || unit_length <= 0.0)
    return -1;
  for (f = 0; f < run->nfluid; f++) {
    if (mass_amu[f] <= 0.0)
      return -1;
    run->mass_code[f] = mass_amu[f] * XMF_AMU / unit_density;
  }
  run->unit_density = unit_density;
  run->unit_velocity = unit_velocity;
  run->unit_length = unit_length;
  run->unit_P = unit_density * unit_velocity * unit_velocity;
  xmf_dev_push_tables(run); /* C2 sync point: mass table mirror -> device */
  return 0;
}

/* Per-fluid charges (B1), auxiliary load port -- see xmf_core.h. charge_e
   carries nfluid charges in elementary-charge multiples q_s; the kernel
   stores them as-is (the ion role is q_s != 0, names never consulted).
   Charge defaults are set in xmf_setup to 0.0, so a run that never calls
   this port keeps the pre-B1 hydro path bit for bit (the electron
   closure gate stays off). */
int xmf_charge_load(XmfRun *run, const double *charge_e) {
  int f;
  if (!run || !charge_e)
    return -1;
  for (f = 0; f < run->nfluid; f++) {
    if (!isfinite(charge_e[f]))
      return -1;
  }
  for (f = 0; f < run->nfluid; f++)
    run->charge[f] = charge_e[f];
  xmf_dev_push_tables(run); /* C2 sync point: charge table mirror -> device */
  return 0;
}
