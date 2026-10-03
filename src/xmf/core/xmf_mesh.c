/* Per-direction mesh construction and the read-only xmf_mesh_* queries of
   the ideal-Euler kernel (A4.5 split out of xmf_core.c; the public surface
   stays xmf_core.h).

   Mesh (A1): one authoritative 1-D mesh per direction d = 0,1,2. Each mesh
   stores exactly two point sets and three metrics -- no corner, edge-centre
   or volume-centroid (xgc) points, and no 3-D coordinate field:
       x   -- cell-centre coordinates, length n_d + 2*ng (ghosts included);
       xf  -- one face lattice, length n_d + 1 + 2*ng. Face f sits at xf[f];
              cell i's left face is xf[i], its right face is xf[i+1]. No
              separate xl/xr arrays are stored;
       dx  -- cell widths, same length as x, dx[i] = xf[i+1] - xf[i]. A
              non-uniform lattice is the default; a uniform grid is just
              the special case of equal widths (no separate code path);
       A   -- geometric face-area factor, same length as xf. A[f] is
              evaluated at the face coordinate xf[f] itself (spherical x1:
              xf^2 -- never at the cell centre);
       dV  -- geometric cell-volume factor, same length as x. The 3-D cell
              volume is the product dV_x1[i]*dV_x2[j]*dV_x3[k]; no separate
              3-D volume array exists. */
#include "xmf_run.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

double *dvec(size_t n) {
  return (double *)calloc(n ? n : 1, sizeof(double));
}

/* ---- Mesh construction (A1) ------------------------------------------- */

/* Allocate the five arrays of one direction's mesh. Every pointer starts
   NULL (the run is calloc'd), so the shared cleanup path stays safe if a
   later allocation fails. Returns 1 on success. */
int axis_alloc(XmfAxis *ax, int ntot) {
  ax->ntot = ntot;
  ax->x = dvec((size_t)ntot);
  ax->xf = dvec((size_t)ntot + 1);
  ax->dx = dvec((size_t)ntot);
  ax->A = dvec((size_t)ntot + 1);
  ax->dV = dvec((size_t)ntot);
  return ax->x && ax->xf && ax->dx && ax->A && ax->dV;
}

/* Fill one direction from the caller's ACTIVE face lattice (A1 stretch
   supplement). xf_active carries the n+1 absolute positions of the active
   interfaces (ghosts not included); the Python front end builds them
   (uniform or stretched alike -- the kernel never sees a recipe, only
   faces). The kernel is the authoritative monotonicity validator:
   xf_active[i+1] > xf_active[i] for every i, else the fill (and setup)
   fails. This is a can-compute check, never grid scoring: no stretch-ratio
   or smoothness measure exists here. Ghosts take the ADJACENT active width
   (left ghosts all Delta_L = xf_active[1]-xf_active[0], right ghosts all
   Delta_R = xf_active[n]-xf_active[n-1]); no geometric extrapolation.
   Centres and widths are then derived self-consistently:
       x[i]  = 0.5*(xf[i]+xf[i+1]),   dx[i] = xf[i+1]-xf[i].
   Returns 1 on success, 0 on a non-finite or non-increasing lattice. */
int axis_fill_faces(XmfAxis *ax, int n, int ng, const double *xf_active) {
  int i, ntot = ax->ntot; /* n + 2*ng, set by axis_alloc */
  double dl, dr;
  if (!xf_active)
    return 0;
  for (i = 0; i <= n; i++)
    if (!isfinite(xf_active[i]))
      return 0; /* a non-finite interface is not computable */
  for (i = 0; i < n; i++)
    if (!(xf_active[i + 1] > xf_active[i]))
      return 0; /* strictly increasing active faces (NaN fails too) */
  dl = xf_active[1] - xf_active[0];     /* first active width */
  dr = xf_active[n] - xf_active[n - 1]; /* last active width */
  for (i = 0; i < ng; i++)
    ax->xf[i] = xf_active[0] - (double)(ng - i) * dl;
  for (i = 0; i <= n; i++)
    ax->xf[ng + i] = xf_active[i]; /* active lattice copied verbatim */
  for (i = 1; i <= ng; i++)
    ax->xf[ng + n + i] = xf_active[n] + (double)i * dr;
  for (i = 0; i < ntot; i++) {
    ax->x[i] = 0.5 * (ax->xf[i] + ax->xf[i + 1]);
    ax->dx[i] = ax->xf[i + 1] - ax->xf[i];
  }
  return 1;
}

/* Geometric face-area and cell-volume factors. Metric contract follows the
   old tree's set_geometry.c (read-only reference, nothing copied): areas
   are evaluated at the face coordinate itself -- never at the cell centre
   -- and the 3-D cell volume is the product of the three per-direction
   factors. A is a geometric factor only (no 4*pi prefactor). */
void axis_fill_metrics(XmfAxis *ax, int geometry, int dir) {
  int i, f, ntot = ax->ntot;
  /* A[f] at face xf[f]; the leftmost face f=0 is part of the lattice, so no
     special-casing beyond the formula itself. */
  for (f = 0; f <= ntot; f++) {
    double s = ax->xf[f];
    double a;
    if (geometry == XMF_GEOM_SPHERICAL && dir == 0)
      a = s * s; /* spherical r: A = r_face^2 */
    else if (geometry == XMF_GEOM_SPHERICAL && dir == 1)
      a = fabs(sin(s)); /* spherical theta: A = |sin theta_face| */
    else if (geometry == XMF_GEOM_CYLINDRICAL && dir == 0)
      a = fabs(s); /* cylindrical R: A = |R_face| */
    else
      a = 1.0; /* cartesian; spherical/cylindrical azimuthal and z */
    ax->A[f] = a;
  }
  for (i = 0; i < ntot; i++) {
    double xl = ax->xf[i], xr = ax->xf[i + 1];
    double v;
    if (geometry == XMF_GEOM_SPHERICAL && dir == 0)
      v = fabs(xr * xr * xr - xl * xl * xl) / 3.0;
    else if (geometry == XMF_GEOM_SPHERICAL && dir == 1)
      v = fabs(cos(xl) - cos(xr));
    else if (geometry == XMF_GEOM_CYLINDRICAL && dir == 0)
      v = fabs(ax->x[i]) * ax->dx[i];
    else
      v = ax->dx[i]; /* cartesian; angular/z directions */
    ax->dV[i] = v;
  }
}

/* ---- Read-only mesh queries (helper style, like xmf_ncell/xmf_time) ----
   Direction d is 0=x1, 1=x2, 2=x3. The copy functions write into
   caller-owned storage and return the number of doubles written; they
   return -1 (writing nothing) on invalid arguments. Lengths: x, dx and dV
   have xmf_mesh_ncell(run,d) entries; xf and A have xmf_mesh_nface(run,d)
   entries. Coordinates carry the same code units as xdmin/xdmax of the
   config; A and dV are geometric factors (the full 3-D cell volume is the
   product dV_x1[i]*dV_x2[j]*dV_x3[k]). */

int xmf_mesh_ncell(const XmfRun *run, int d) {
  if (!run || d < 0 || d > 2)
    return 0;
  return run->mesh[d].ntot; /* cells including ghosts: n_d + 2*ng */
}

int xmf_mesh_nface(const XmfRun *run, int d) {
  if (!run || d < 0 || d > 2)
    return 0;
  return run->mesh[d].ntot + 1;
}

static int axis_copy(const double *src, int len, double *out) {
  if (!src || !out)
    return -1;
  memcpy(out, src, (size_t)len * sizeof(double));
  return len;
}

int xmf_mesh_copy_x(const XmfRun *run, int d, double *out) {
  if (!run || d < 0 || d > 2)
    return -1;
  return axis_copy(run->mesh[d].x, run->mesh[d].ntot, out);
}

int xmf_mesh_copy_xf(const XmfRun *run, int d, double *out) {
  if (!run || d < 0 || d > 2)
    return -1;
  return axis_copy(run->mesh[d].xf, run->mesh[d].ntot + 1, out);
}

int xmf_mesh_copy_dx(const XmfRun *run, int d, double *out) {
  if (!run || d < 0 || d > 2)
    return -1;
  return axis_copy(run->mesh[d].dx, run->mesh[d].ntot, out);
}

int xmf_mesh_copy_A(const XmfRun *run, int d, double *out) {
  if (!run || d < 0 || d > 2)
    return -1;
  return axis_copy(run->mesh[d].A, run->mesh[d].ntot + 1, out);
}

int xmf_mesh_copy_dV(const XmfRun *run, int d, double *out) {
  if (!run || d < 0 || d > 2)
    return -1;
  return axis_copy(run->mesh[d].dV, run->mesh[d].ntot, out);
}
