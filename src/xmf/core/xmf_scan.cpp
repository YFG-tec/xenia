/* Radiation line scan (A5, RadiationScan mode grid_line) -- own module,
   never merged back into xmf_core.c.

   A ray is nothing but the sequence of grid cells along an x1 line plus
   the mesh spacing ds of each cell; there is no separate ray object and
   no characteristic tracing. The beam table (per-beam cross section,
   photon energy and boundary flux) arrives through the xmf_rad_load
   auxiliary LOAD PORT -- never through the flat XmfConfig and never as a
   formula: the kernel recognises no species name and carries no
   photo-ionisation model of its own, only the loaded numbers.

   Scan semantics (mode == grid_line, dilution == none, sense == inward):
   for every interior x1 line (loop order k, j, i) and every beam b the
   optical depth of cell i is
       dtau_b = sigma_b * n_abs * ds,      ds = dx[i] * unit_length  [cm]
   where n_abs = rho_abs/mass_code[absorber] is the absorber's number
   density in cm^-3 (rho_abs floored at zero), and the transmitted flux
   walks inward from the x1end outer face:
       F_{b,i}   = F_{b,i+1} * exp(-dtau_b)        (flux after cell i)
       dF_{b,i}  = F_in * (1 - exp(-dtau_b))       (absorbed in cell i)
   with F_in = F_infty at the outer boundary. Both arrays are stored:
   flux[b][c] = F_{b,i}, dflux[b][c] = dF_{b,i}, laid out [beam][cell]
   over the full cell block (ghosts stay zero; only interior cells are
   filled). When the radiation is disabled (enabled == 0) the scan never
   runs and the IR FLUX slots read the boundary value finf instead --
   the vacuum limit; the Python factory passes finf == 0 for an
   off-radiation run, so F_b vanishes there while the source tapes stay
   loaded.

   Public helpers (declared in xmf_core.h, auxiliary load/read port level
   like xmf_ir_load -- NOT a fifth public verb):
       xmf_rad_load           -- load the beam table, allocate F_b state;
       xmf_rad_load_absorbers -- B3 appended port: multi-absorber index
                                 table + sigma[n_abs x n_beam];
       xmf_rad_nbeam          -- loaded beam count;
       xmf_rad_flux_copy      -- copy one beam's transmitted flux out.
   xmf_rad_scan is the internal seam called by xmf_step (via xmf_run.h);
   it is a no-op when no radiation state is loaded or it is disabled.

   C2: the scan runs on the device radiation Views (sigma/hnu/finf/flux/
   dflux/sigma_multi/absorbers mirrored by xmf_dev_rad_alloc /
   xmf_dev_rad_sync_absorbers at the load ports). The absorber density
   comes off the device field View directly -- the hydro sweeps left it
   there -- so a scan never pulls a mirror. xmf_rad_flux_copy pulls the
   transmitted flux back before copying out. */

#include "xmf_run.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "xmf_device.h"
#include "xmf_kokkos.h"

static void rad_free(XmfRadiation *rad) {
  if (!rad)
    return;
  free(rad->sigma);
  free(rad->hnu);
  free(rad->finf);
  free(rad->flux);
  free(rad->dflux);
  free(rad->absorbers);
  free(rad->sigma_multi);
  free(rad);
}

/* Free the run's beam state (NULL-safe); afterwards run->rad == NULL.
   Internal seam for xmf_finalize, declared in xmf_run.h. */
void xmf_rad_clear(XmfRun *run) {
  if (!run)
    return;
  rad_free(run->rad);
  run->rad = NULL;
}

/* Copy a beam table out of the caller into freshly allocated storage.
   Returns 0, or -1 on allocation failure (nothing partially kept: the
   caller frees on failure). */
static int copy_vec(double **dst, const double *src, int n) {
  double *buf = (double *)malloc(sizeof(double) * (size_t)n);
  if (!buf)
    return -1;
  memcpy(buf, src, sizeof(double) * (size_t)n);
  *dst = buf;
  return 0;
}

/* Beam-table load port (A5), see the file header and xmf_core.h. This
   slice implements mode == 1 (grid_line), dilution == 0 (none) and
   sense == 0 (inward) only; any other value is refused with -1 and no
   state stored (the enum room stays reserved for later slices). A later
   successful call replaces the previous state. Returns nbeam, or -1 on
   invalid arguments / unsupported mode / allocation failure.

   C2: after the host state is swapped in, xmf_dev_rad_alloc rebuilds and
   pushes the device radiation Views (the host flux/dflux arrive zero).
   A device-side failure restores the previous state -- the new block is
   freed and the old one reinstated -- so the contract "nothing partially
   kept" covers the mirror too. */
int xmf_rad_load(XmfRun *run, int enabled, int mode, int dilution,
                 int sense, int absorber, int nbeam, const double *sigma,
                 const double *hnu, const double *finf) {
  XmfRadiation *rad, *old;
  if (!run || !sigma || !hnu || !finf)
    return -1;
  if (enabled != 0 && enabled != 1)
    return -1;
  if (mode != 1 || dilution != 0 || sense != 0)
    return -1; /* only grid_line / none / inward exist this slice; the
                  enums are kept but everything else is refused */
  if (nbeam < 1 || absorber < 0 || absorber >= run->nfluid)
    return -1;
  rad = (XmfRadiation *)calloc(1, sizeof(XmfRadiation));
  if (!rad)
    return -1;
  if (copy_vec(&rad->sigma, sigma, nbeam) ||
      copy_vec(&rad->hnu, hnu, nbeam) ||
      copy_vec(&rad->finf, finf, nbeam)) {
    rad_free(rad);
    return -1;
  }
  rad->flux = dvec((size_t)nbeam * run->ncells);
  rad->dflux = dvec((size_t)nbeam * run->ncells);
  if (!rad->flux || !rad->dflux) {
    rad_free(rad);
    return -1;
  }
  rad->enabled = enabled;
  rad->mode = mode;
  rad->dilution = dilution;
  rad->sense = sense;
  rad->absorber = absorber;
  rad->nbeam = nbeam;
  /* Swap in the new state, then (re)build the device mirror from it. On
     mirror failure restore the previous state verbatim. */
  old = run->rad;
  run->rad = rad;
  if (xmf_dev_rad_alloc(run)) {
    run->rad = old;
    rad_free(rad);
    return -1;
  }
  rad_free(old); /* replace any previously loaded beam state */
  return nbeam;
}

/* Multi-absorber load port (B3), appended after xmf_rad_load -- see the
   contract in xmf_core.h. Requires a beam table already loaded (its
   nbeam fixes the row length); replaces any previous multi table. The
   single-absorber fields (absorber/sigma) are left untouched: they keep
   the flat-port contract but stop steering the scan once this table is
   present. Returns nabs, or -1 with the previous state untouched.

   C2: after the host swap, xmf_dev_rad_sync_absorbers rebuilds and pushes
   ONLY the sigma_multi / absorbers device Views (flux/dflux untouched --
   a scan between loads must never see stale flux); on mirror failure the
   previous host + device state is restored. */
int xmf_rad_load_absorbers(XmfRun *run, int nabs, const int *absorbers,
                           const double *sigma) {
  XmfRadiation *rad;
  int *abuf, *old_absorbers;
  double *sbuf, *old_sigma_multi;
  int s, old_nabs;
  if (!run || !run->rad || !absorbers || !sigma || nabs < 2)
    return -1;
  rad = run->rad;
  for (s = 0; s < nabs; s++)
    if (absorbers[s] < 0 || absorbers[s] >= run->nfluid)
      return -1;
  abuf = (int *)malloc(sizeof(int) * (size_t)nabs);
  sbuf = (double *)malloc(sizeof(double) * (size_t)nabs *
                          (size_t)rad->nbeam);
  if (!abuf || !sbuf) {
    free(abuf);
    free(sbuf);
    return -1;
  }
  memcpy(abuf, absorbers, sizeof(int) * (size_t)nabs);
  memcpy(sbuf, sigma,
         sizeof(double) * (size_t)nabs * (size_t)rad->nbeam);
  old_nabs = rad->nabs;
  old_absorbers = rad->absorbers;
  old_sigma_multi = rad->sigma_multi;
  rad->nabs = nabs;
  rad->absorbers = abuf;
  rad->sigma_multi = sbuf;
  if (xmf_dev_rad_sync_absorbers(run)) {
    /* Mirror failure: restore the previous multi table, drop the new. */
    rad->nabs = old_nabs;
    rad->absorbers = old_absorbers;
    rad->sigma_multi = old_sigma_multi;
    free(abuf);
    free(sbuf);
    return -1;
  }
  free(old_absorbers);
  free(old_sigma_multi);
  return nabs;
}

int xmf_rad_nbeam(const XmfRun *run) {
  return (run && run->rad) ? run->rad->nbeam : 0;
}

/* Auxiliary READ port for one beam's transmitted flux (ghosts included,
   xmf_state_ncells doubles). Ghost cells carry whatever the last scan
   left there (zero before the first scan; the scan fills interior cells
   only). Returns the number of doubles written, or -1.
   C2: the scan writes the device flux View, so the mirror is pulled back
   before the copy. */
int xmf_rad_flux_copy(const XmfRun *run, int b, double *out) {
  if (!run || !run->rad || !out || b < 0 || b >= run->rad->nbeam)
    return -1;
  xmf_dev_pull_flux(run);
  memcpy(out, run->rad->flux + (size_t)b * run->ncells,
         run->ncells * sizeof(double));
  return (int)run->ncells;
}

/* Fill flux/dflux from the current absorber density (called from
   xmf_step before the hydro operator of each RK stage; a no-op when no
   radiation state is loaded or it is disabled). The walk is strictly
   local in each beam: cell i only needs the flux transmitted by cell
   i+1, so the loop runs i from the outermost interior cell down to the
   innermost -- never IEND->IBEG style index wraparound, just the natural
   inward direction of the incoming radiation.

   B3: with a multi-absorber table loaded (xmf_rad_load_absorbers) the
   per-beam optical depth is tau_b = (sum_s sigma_{s,b} n_s) * ds over
   the table's absorber fluids; with no table loaded the single-absorber
   walk below runs bit for bit the A5 one.

   C1a: the MUTUALLY INDEPENDENT rays (b, k, j) ride one Kokkos
   parallel_for -- different rays never share state (each starts from its
   own F_infty and writes only its own beam's [beam][cell] rows). On each
   ray the i walk stays SEQUENTIAL, outermost cell inward, exactly the
   prefix product above: the i direction is never folded into the
   parallel iteration space.

   C2: the ray space rides the default execution space; every array the
   kernel touches (beam tables, flux/dflux, absorber density, dx, mass
   table) is a device View of the run's DeviceState. The absorber density
   is the post-source/post-stage device field state -- the scan is called
   before lf_all_dirs of each stage, so it sees exactly U^n of that
   stage. */
void xmf_rad_scan(XmfRun *run) {
  XmfRadiation *rad;
  xmf::DeviceState *dv;
  int ng, nx, ny, nz, nrays;
  if (!run || !run->rad || !run->rad->enabled)
    return; /* disabled / not loaded: FLUX slots read finf at eval time */
  rad = run->rad;
  dv = xmf::dev(run);
  if (!dv)
    return; /* no device state attached: nothing to scan into */
  ng = run->ng;
  nx = run->nx;
  ny = run->ny;
  nz = run->nz;
  nrays = rad->nbeam * nz * ny;
  {
    const int nxtot = run->nxtot, nytot = run->nytot;
    const size_t ncells = run->ncells;
    const double unit_length = run->unit_length;
    const double *dx = dv->dx[0].data();
    const double *fields = dv->fields.data();
    const double *mass_code = dv->mass_code.data();
    const double *finf = dv->finf.data();
    double *flux = dv->flux.data();
    double *dflux = dv->dflux.data();
    if (rad->nabs >= 2 && rad->absorbers && rad->sigma_multi) {
      /* Multi-absorber walk (B3): same ray space, direction and storage
         as the single-absorber walk below; only tau picks up the sum over
         the table's absorber fluids. */
      const int nabs = rad->nabs;
      const int nbeam = rad->nbeam;
      const double *sigma_multi = dv->sigma_multi.data();
      const int *absorbers = dv->absorbers.data();
      Kokkos::parallel_for(
          "xmf_rad_scan_multi", nrays, KOKKOS_LAMBDA(const int ray) {
            const int b = ray / (nz * ny);
            const int rem = ray % (nz * ny);
            const int k = ng + rem / ny;
            const int j = ng + rem % ny;
            double f = finf[b]; /* F_in at the x1end outer face */
            /* Sequential inward prefix product on this ray. */
            for (int i = ng + nx - 1; i >= ng; i--) {
              size_t c = (((size_t)k * (size_t)nytot) + (size_t)j) *
                             (size_t)nxtot +
                         (size_t)i;
              double kappa = 0.0; /* sum_s sigma_{s,b} n_s, cm^-1 */
              double tau, e, ds;
              for (int s = 0; s < nabs; s++) {
                int fa = absorbers[s];
                double n_abs =
                    fields[((size_t)fa * XMF_SLOT_NVAR) * ncells + c] /
                    mass_code[fa];
                if (n_abs < 0.0)
                  n_abs = 0.0; /* never let a negative density amplify */
                kappa += sigma_multi[(size_t)s * nbeam + b] * n_abs;
              }
              ds = dx[i] * unit_length; /* cell width in cm */
              tau = kappa * ds;
              e = exp(-tau);
              dflux[(size_t)b * ncells + c] =
                  f * (1.0 - e); /* absorbed inside cell i */
              f *= e;            /* transmitted onward (inward) */
              flux[(size_t)b * ncells + c] = f;
            }
          });
      return;
    }
    {
      const double *rho_abs =
          fields + ((size_t)rad->absorber * XMF_SLOT_NVAR) * ncells;
      const double m_abs = run->mass_code[rad->absorber];
      const double *sigma = dv->sigma.data();
      Kokkos::parallel_for(
          "xmf_rad_scan", nrays, KOKKOS_LAMBDA(const int ray) {
            const int b = ray / (nz * ny);
            const int rem = ray % (nz * ny);
            const int k = ng + rem / ny;
            const int j = ng + rem % ny;
            double f = finf[b]; /* F_in at the x1end outer face */
            /* Sequential inward prefix product on this ray. */
            for (int i = ng + nx - 1; i >= ng; i--) {
              size_t c = (((size_t)k * (size_t)nytot) + (size_t)j) *
                             (size_t)nxtot +
                         (size_t)i;
              double n_abs = rho_abs[c] / m_abs; /* number density, cm^-3 */
              double tau, e, ds;
              if (n_abs < 0.0)
                n_abs = 0.0; /* never let a negative density amplify flux */
              ds = dx[i] * unit_length;    /* cell width in cm */
              tau = sigma[b] * n_abs * ds; /* dimensionless depth */
              e = exp(-tau);
              dflux[(size_t)b * ncells + c] =
                  f * (1.0 - e); /* absorbed inside cell i */
              f *= e;            /* transmitted onward (inward) */
              flux[(size_t)b * ncells + c] = f;
            }
          });
    }
  }
}
