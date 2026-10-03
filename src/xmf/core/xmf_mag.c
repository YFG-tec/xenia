/* Magnetic-field module of the ideal-MHD kernel (M block 1: M1 IR/config
   name space on the Python/header side, M2 staggered storage, M3 B
   boundary conditions; split out as its own translation unit -- the
   public surface stays xmf_core.h, the shared state xmf_run.h).

   Staggered storage (M2). The magnetic field lives face-centred, one
   block per component (bf[0..2]); the electric field lives edge-centred
   (ef[0..2]), allocated zero and untouched this slice (no induction yet
   -- the M4/M5 blocks own it). Layouts follow the XmfRun comment in
   xmf_run.h, always (k,j,i) with the block's own direction fastest:
       bf[d]: (ntot[d]+1) faces along d, cells across d1=(d+1)%3 and
              d2=(d+2)%3; length (ntot[d]+1)*ntot[d1]*ntot[d2];
       ef[d]: ntot[d] cells along d, (ntot[d1]+1)*(ntot[d2]+1) edges
              across; length ntot[d]*(ntot[d1]+1)*(ntot[d2]+1).
   1D runs are the degeneration nx2=nx3=1: the transverse directions
   still carry ghost cells, so the blocks keep their full transverse
   extent and every loop below walks all three indices.

   Gate. xmf_mag_load allocates the blocks and opens the gate; bf[0]
   NULL is gate closed, the pre-M state: every helper refuses (-1) or
   no-ops and the fluid five slots stay bit for bit untouched.

   Block 1 step semantics (superseded by M4): B was a frozen
   bystander. M4 (this file's induction section): with the gate open
   xmf_mag_load refuses hllc and any run without a charge > 0 fluid;
   each RK stage snaps the charge-weighted ion velocity u_+ before the
   sweeps and advances B once afterwards by the finite-volume induction
   d_t B = curl(u_+ x B); the ion sweeps themselves live in
   xmf_hydro.cpp (8-component MHD Riemann, frozen stage B). B ghosts
   are refilled at every fill_bc; the E blocks stay zero until M5,
   which replaces the arithmetic-average interface EMF with constrained
   transport without touching the storage contract here.

   Boundary conditions (M3), registered in bc_fill_aux_blocks:
   - reflect faces follow run->mag_reflect (enum XmfMagReflect): the
     WALL tier flips the normal component and pins the wall face to 0
     (PLUTO REFLECTIVE / AthenaK conducting-wall image); the SYMMETRIC
     tier flips the tangential components and copies the normal one,
     leaving the wall face untouched (PLUTO EQTSYMMETRIC / Idefix
     axial-vector mirroring). The fluid five slots use the same reflect
     fill either way;
   - copy / outflow / none pad zero-order (boundary face value for the
     normal ghosts, outermost active cell for the tangential ghosts);
   - user faces prescribe the tangential components -- from the
     xmf_mag_set_user floor on x1 faces (layout [tcomp][k*nytot+j],
     tcomp 0=B2, 1=B3; the buffer is cell-row sized, so a component's
     top face row has no floor entry and falls back to copy @1), else
     copy @1 -- and reconstruct the normal component's ghost faces from
     the discrete divB = 0 constraint,
     marching outward from the boundary face (an ACTIVE value, never
     overwritten): for tensor-product metrics div = 0 per cell is
         sum_e (A_e[i_e+1] B_e[i_e+1] - A_e[i_e] B_e[i_e]) / dV_e[i_e] = 0
     with A the face-area factor and dV the volume factor of the mesh
     (xmf_mesh.c), so one ghost face per cell follows from the other
     five faces and the transverse metrics of its own row. */
#include "xmf_run.h"

#include <stdlib.h>
#include <string.h>

/* ---- Block geometry ------------------------------------------------------ */

static int mag_ntot(const XmfRun *run, int d) {
  return (d == 0) ? run->nxtot : (d == 1) ? run->nytot : run->nztot;
}

static int mag_nint(const XmfRun *run, int d) {
  return (d == 0) ? run->nx : (d == 1) ? run->ny : run->nz;
}

size_t xmf_mag_len_b(const XmfRun *run, int comp) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  return (size_t)(mag_ntot(run, comp) + 1) * (size_t)mag_ntot(run, d1) *
         (size_t)mag_ntot(run, d2);
}

size_t xmf_mag_len_e(const XmfRun *run, int comp) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  return (size_t)mag_ntot(run, comp) * (size_t)(mag_ntot(run, d1) + 1) *
         (size_t)(mag_ntot(run, d2) + 1);
}

/* Flat index into bf[comp] with the full coordinate triple c: c[comp]
   is the FACE index along the component's own direction, the other two
   entries are CELL indices (xmf_run.h layout comment). */
static size_t mag_bidx(const XmfRun *run, int comp, const int c[3]) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  return (((size_t)c[d2] * (size_t)mag_ntot(run, d1)) + (size_t)c[d1]) *
             (size_t)(mag_ntot(run, comp) + 1) +
         (size_t)c[comp];
}

/* Flat index into ef[comp] with the full coordinate triple c: c[comp]
   is the CELL index along the component's own direction (range
   0..ntot[comp]-1), the other two entries are EDGE indices (range
   0..ntot[d1] and 0..ntot[d2]). Layout from the xmf_run.h comment:
   ef[d] length ntot[d]*(ntot[d1]+1)*(ntot[d2]+1), fastest index d. */
static size_t mag_eidx(const XmfRun *run, int comp, const int c[3]) {
  int d1 = (comp + 1) % 3, d2 = (comp + 2) % 3;
  return (((size_t)c[d2] * (size_t)(mag_ntot(run, d1) + 1)) +
          (size_t)c[d1]) *
             (size_t)mag_ntot(run, comp) +
         (size_t)c[comp];
}

/* ---- Load port and pack/unpack helpers ----------------------------------- */

int xmf_mag_load(XmfRun *run, int reflect_mode, const double *b1,
                 const double *b2, const double *b3) {
  const double *bin[3];
  double *nb[3] = {NULL, NULL, NULL};
  double *ne[3] = {NULL, NULL, NULL};
  int d;
  if (!run || !b1 || !b2 || !b3)
    return -1;
  if (reflect_mode != XMF_MAG_REFLECT_WALL &&
      reflect_mode != XMF_MAG_REFLECT_SYMMETRIC)
    return -1;
  /* M4 gate checks -- the magnetic gate only opens when the ion MHD
     machinery has something to work on (setup-level refusal lands here
     because the charge table and this port load after xmf_setup; the
     Python setup() raises on the -1):
     - hllc's wave fan carries no magnetic waves, so the gate refuses
       it outright -- no silent fallback to hlld/hll, no rewriting the
       solver integer;
     - with no charge > 0 fluid at all, u_+ is undefined and every
       B_s = 0 while the gate hangs open -- refused. */
  if (run->riemann == XMF_RIEMANN_HLLC)
    return -1;
  {
    int any_ion = 0, f;
    for (f = 0; f < run->nfluid; f++)
      if (run->charge[f] > 0.0)
        any_ion = 1;
    if (!any_ion)
      return -1;
  }
  bin[0] = b1;
  bin[1] = b2;
  bin[2] = b3;
  for (d = 0; d < 3; d++) {
    nb[d] = dvec(xmf_mag_len_b(run, d)); /* zeroed; the memcpy below
                                             overwrites the whole B block */
    ne[d] = dvec(xmf_mag_len_e(run, d)); /* E stays calloc'd zero */
    if (!nb[d] || !ne[d])
      goto fail;
  }
  /* Complete: swap the new state in, drop the previous one (NULL-safe
     frees), record the reflect tier, mirror to the device. */
  for (d = 0; d < 3; d++) {
    free(run->bf[d]);
    free(run->ef[d]);
    run->bf[d] = nb[d];
    run->ef[d] = ne[d];
    memcpy(run->bf[d], bin[d], xmf_mag_len_b(run, d) * sizeof(double));
  }
  run->mag_reflect = reflect_mode;
  if (xmf_dev_mag_alloc(run)) {
    /* Device mirror failed: roll back to gate closed rather than leave
       a host state with a stale / missing device twin. */
    for (d = 0; d < 3; d++) {
      free(run->bf[d]);
      free(run->ef[d]);
      run->bf[d] = NULL;
      run->ef[d] = NULL;
    }
    return -1;
  }
  /* M4: the stage-start u_+ snapshot buffer (three component blocks of
     ncells each, ghosts included), allocated with the gate and freed by
     xmf_mag_clear. A reload with the gate already open keeps the
     existing buffer. */
  if (!run->mag_up) {
    run->mag_up = dvec(3 * run->ncells);
    if (!run->mag_up) {
      for (d = 0; d < 3; d++) {
        free(run->bf[d]);
        free(run->ef[d]);
        run->bf[d] = NULL;
        run->ef[d] = NULL;
      }
      return -1;
    }
  }
  return 0;

fail:
  for (d = 0; d < 3; d++) {
    free(nb[d]);
    free(ne[d]);
  }
  return -1;
}

int xmf_mag_pack(XmfRun *run, int comp, const double *in) {
  if (!run || !in || comp < 0 || comp > 2 || !run->bf[comp])
    return -1; /* gate closed (or bad arguments): move nothing */
  memcpy(run->bf[comp], in, xmf_mag_len_b(run, comp) * sizeof(double));
  xmf_dev_push_b(run); /* C2 sync point: mirrors -> device */
  return (int)xmf_mag_len_b(run, comp);
}

int xmf_mag_unpack(const XmfRun *run, int comp, double *out) {
  if (!run || !out || comp < 0 || comp > 2 || !run->bf[comp])
    return -1;
  xmf_dev_pull_b(run); /* C2 sync point: device -> mirrors */
  memcpy(out, run->bf[comp], xmf_mag_len_b(run, comp) * sizeof(double));
  return (int)xmf_mag_len_b(run, comp);
}

int xmf_mag_unpack_e(const XmfRun *run, int comp, double *out) {
  if (!run || !out || comp < 0 || comp > 2 || !run->ef[comp])
    return -1;
  xmf_dev_pull_b(run); /* E rides the same mirror pair (block 1 keeps it
                          zero; the read seam exists for the M2 tests) */
  memcpy(out, run->ef[comp], xmf_mag_len_e(run, comp) * sizeof(double));
  return (int)xmf_mag_len_e(run, comp);
}

/* Tangential-B floor of the x1 user faces (M3), auxiliary load port --
   see xmf_core.h. The buffer is copied into the run and owned by it
   from here on (freed by xmf_finalize via xmf_mag_clear). */
int xmf_mag_set_user(XmfRun *run, const double *data) {
  size_t nt, bytes;
  double *buf;
  if (!run || !data || !run->bf[0])
    return -1; /* a B floor only ever exists with the gate open */
  nt = (size_t)run->nytot * (size_t)run->nztot;
  bytes = 2 * nt * sizeof(double); /* tcomp 0 = B2, tcomp 1 = B3 */
  buf = (double *)malloc(bytes);
  if (!buf)
    return -1;
  memcpy(buf, data, bytes);
  free(run->mag_user_bc); /* replace any previously loaded floor */
  run->mag_user_bc = buf;
  /* Port B: push the new magnetic floor to the device Views. The B
     ghost fill kernel reads from the device copy; the host pointer is
     never dereferenced by device kernels. */
  xmf_dev_bc_sync(run);
  return 0;
}

void xmf_mag_clear(XmfRun *run) {
  int d;
  if (!run)
    return;
  for (d = 0; d < 3; d++) {
    free(run->bf[d]);
    free(run->ef[d]);
    run->bf[d] = NULL;
    run->ef[d] = NULL;
  }
  free(run->mag_user_bc);
  run->mag_user_bc = NULL;
  free(run->mag_up); /* M4 u_+ snapshot buffer; NULL-safe */
  run->mag_up = NULL;
}

/* ---- Per-face ghost fills (M3) -------------------------------------------
   All three fillers walk the whole transverse plane like the fluid
   fill_face_block does, so 1D (degenerate transverse directions) stays
   self-consistent. Only this face's ghost entries are written; active
   cells and the shared boundary face are never touched (the user-face
   normal reconstruction reads the boundary face as its marching seed). */

static void mag_fill_reflect(XmfRun *run, int face) {
  int d = face / 2, side = face % 2;
  int ng = run->ng;
  int nint = mag_nint(run, d);
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int wall = (run->mag_reflect == XMF_MAG_REFLECT_WALL);
  int w = (side == 0) ? ng : ng + nint; /* boundary face */
  double sgn_n, sgn_t;
  int a, b, delta, t;
  /* Mirror parities per tier: wall (PLUTO REFLECTIVE / AthenaK) is the
     conducting-wall image -- B_n odd (ghost faces flipped, wall face
     pinned to 0), B_t even; symmetric (PLUTO EQTSYMMETRIC / Idefix) is
     the axial-vector symmetry plane -- B_n even (copied, wall face
     untouched), B_t odd (flipped).

     Staggered extents: the normal component is face-indexed along d and
     cell-indexed across (both transverse coordinates walk cells). Each
     tangential component is cell-indexed along d but FACE-indexed along
     its own direction -- its transverse walk covers ntot[t]+1 face rows
     times ntot[u] cells, u the remaining direction. */
  sgn_n = wall ? -1.0 : 1.0;
  sgn_t = wall ? 1.0 : -1.0;
  for (a = 0; a < mag_ntot(run, t1); a++)
    for (b = 0; b < mag_ntot(run, t2); b++) {
      int c[3];
      c[t1] = a;
      c[t2] = b;
      /* Normal component (face-indexed along d). */
      if (wall) {
        c[d] = w;
        run->bf[d][mag_bidx(run, d, c)] = 0.0;
      }
      for (delta = 1; delta <= ng; delta++) {
        int f = (side == 0) ? w - delta : w + delta;
        int src = (side == 0) ? w + delta : w - delta;
        double v;
        c[d] = src;
        v = run->bf[d][mag_bidx(run, d, c)];
        c[d] = f;
        run->bf[d][mag_bidx(run, d, c)] = sgn_n * v;
      }
    }
  for (t = 0; t < 3; t++) {
    int u, nf, nc;
    if (t == d)
      continue;
    u = 3 - d - t; /* remaining transverse direction (cell-indexed) */
    nf = mag_ntot(run, t) + 1;
    nc = mag_ntot(run, u);
    for (a = 0; a < nf; a++)
      for (b = 0; b < nc; b++) {
        int c[3];
        c[t] = a; /* face row along the component's own direction */
        c[u] = b;
        for (delta = 1; delta <= ng; delta++) {
          int g = (side == 0) ? ng - delta : ng + nint - 1 + delta;
          int src = (side == 0) ? ng + delta - 1 : ng + nint - delta;
          double v;
          c[d] = src;
          v = run->bf[t][mag_bidx(run, t, c)];
          c[d] = g;
          run->bf[t][mag_bidx(run, t, c)] = sgn_t * v;
        }
      }
  }
}

/* Zero-order pad for copy / outflow / none faces: normal ghost faces
   take the boundary face value (transverse walk over cells), tangential
   ghost cells the outermost active cell of the same row -- the row walk
   covers the component's own-direction FACE rows (ntot[t]+1 of them)
   times cells of the remaining transverse direction. */
static void mag_fill_copy(XmfRun *run, int face) {
  int d = face / 2, side = face % 2;
  int ng = run->ng;
  int nint = mag_nint(run, d);
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  int a, b, m, t;
  for (a = 0; a < mag_ntot(run, t1); a++)
    for (b = 0; b < mag_ntot(run, t2); b++) {
      int c[3];
      int w = (side == 0) ? ng : ng + nint;
      double bw;
      c[t1] = a;
      c[t2] = b;
      c[d] = w;
      bw = run->bf[d][mag_bidx(run, d, c)];
      for (m = 1; m <= ng; m++) {
        int f = (side == 0) ? ng - m : ng + nint + m;
        c[d] = f;
        run->bf[d][mag_bidx(run, d, c)] = bw;
      }
    }
  for (t = 0; t < 3; t++) {
    int u, nf, nc;
    if (t == d)
      continue;
    u = 3 - d - t;
    nf = mag_ntot(run, t) + 1;
    nc = mag_ntot(run, u);
    for (a = 0; a < nf; a++)
      for (b = 0; b < nc; b++) {
        int c[3], src;
        double v0;
        c[t] = a;
        c[u] = b;
        src = (side == 0) ? ng : ng + nint - 1;
        c[d] = src;
        v0 = run->bf[t][mag_bidx(run, t, c)];
        for (m = 1; m <= ng; m++) {
          int g = (side == 0) ? ng - m : ng + nint - 1 + m;
          c[d] = g;
          run->bf[t][mag_bidx(run, t, c)] = v0;
        }
      }
  }
}

/* Normal-component reconstruction on a user face: march the ghost cells
   outward from the boundary face (whose value stays the active one) and
   solve each cell's discrete divB = 0 for its OUTER ghost face. beg:
   cell i = ng-m, known face i+1, unknown face i,
       B[i] = (A[i+1] B[i+1] + dV[i] S) / A[i];
   end: cell i = ng+nint-1+m, known face i, unknown face i+1,
       B[i+1] = (A[i] B[i] - dV[i] S) / A[i+1];
   S is the transverse flux sum over e != d of the ghost cell itself:
       (A_e[c_e+1] B_e[c_e+1] - A_e[c_e] B_e[c_e]) / dV_e[c_e].
   The B_e values at the ghost cell were just filled by the tangential
   pass of mag_fill_user. A zero face area (spherical pole) falls back
   to copying the known face rather than dividing by zero. */
static void mag_fill_user_normal(XmfRun *run, int d, int side) {
  int ng = run->ng;
  int nint = mag_nint(run, d);
  int t1 = (d == 0) ? 1 : 0;
  int t2 = (d == 2) ? 1 : 2;
  const double *Ad = run->mesh[d].A;
  const double *dVd = run->mesh[d].dV;
  int a, b, m, e;
  for (a = 0; a < mag_ntot(run, t1); a++)
    for (b = 0; b < mag_ntot(run, t2); b++) {
      int c[3];
      c[d] = 0;
      c[t1] = a;
      c[t2] = b;
      for (m = 1; m <= ng; m++) {
        int i = (side == 0) ? ng - m : ng + nint - 1 + m;
        int fk, fu;
        double s, bk;
        c[d] = i;
        s = 0.0;
        for (e = 0; e < 3; e++) {
          const double *Ae, *dVe;
          int ce;
          double vin, vout;
          if (e == d)
            continue;
          Ae = run->mesh[e].A;
          dVe = run->mesh[e].dV;
          ce = c[e];
          if (dVe[ce] == 0.0)
            continue; /* zero-volume slice: no transverse flux term */
          c[e] = ce + 1;
          vout = run->bf[e][mag_bidx(run, e, c)];
          c[e] = ce;
          vin = run->bf[e][mag_bidx(run, e, c)];
          s += (Ae[ce + 1] * vout - Ae[ce] * vin) / dVe[ce];
        }
        if (side == 0) {
          fk = i + 1;
          fu = i;
        } else {
          fk = i;
          fu = i + 1;
        }
        c[d] = fk;
        bk = run->bf[d][mag_bidx(run, d, c)];
        c[d] = fu;
        if (Ad[fu] != 0.0) {
          double num = Ad[fk] * bk + (side == 0 ? dVd[i] * s : -dVd[i] * s);
          run->bf[d][mag_bidx(run, d, c)] = num / Ad[fu];
        } else {
          run->bf[d][mag_bidx(run, d, c)] = bk;
        }
      }
    }
}

/* User-face fill (M3): tangential components from the floor (x1 faces
   only; every other user face falls back to copy @1), all ghost layers,
   whole transverse plane; then the normal ghost faces from divB = 0.
   The boundary face itself is never overwritten -- it is the active
   value the normal march seeds from. The tangential walk covers each
   component's own-direction FACE rows (ntot[t]+1); the floor buffer is
   cell-row sized ([tcomp][k*nytot+j]), so the component's TOP face row
   (a == ntot[t]) has no floor entry and takes the copy @1 fallback. */
static void mag_fill_user(XmfRun *run, int face) {
  int d = face / 2, side = face % 2;
  int ng = run->ng;
  int nint = mag_nint(run, d);
  int has_floor = (d == 0) && run->mag_user_bc;
  size_t nt = (size_t)run->nytot * (size_t)run->nztot;
  int a, b, m, t;
  for (t = 0; t < 3; t++) {
    int u, nf, nc;
    if (t == d)
      continue;
    u = 3 - d - t;
    nf = mag_ntot(run, t) + 1;
    nc = mag_ntot(run, u);
    for (a = 0; a < nf; a++)
      for (b = 0; b < nc; b++) {
        int c[3];
        double v0;
        c[t] = a; /* face row along the component's own direction */
        c[u] = b;
        if (has_floor && a < mag_ntot(run, t)) {
          int tcomp = (t == 1) ? 0 : 1; /* x1 faces: B2 -> 0, B3 -> 1 */
          v0 = run->mag_user_bc[(size_t)tcomp * nt +
                                (size_t)c[2] * (size_t)run->nytot +
                                (size_t)c[1]];
        } else {
          int src = (side == 0) ? ng : ng + nint - 1;
          c[d] = src;
          v0 = run->bf[t][mag_bidx(run, t, c)];
        }
        for (m = 1; m <= ng; m++) {
          int g = (side == 0) ? ng - m : ng + nint - 1 + m;
          c[d] = g;
          run->bf[t][mag_bidx(run, t, c)] = v0;
        }
      }
  }
  mag_fill_user_normal(run, d, side);
}

/* Periodic face fill (M5 supplement): face-centred B ghosts wrap from
   the opposite active domain. For direction d, face index f:
   - beg ghost (f < ng): read from f + nint[d] (end interior)
   - end ghost (f >= ng + nint[d]): read from f - nint[d] (beg interior)
   No sign flips -- periodic is strict wrap, not mirror. The normal
   component B_d on the beg/end ghost faces takes the value of the
   corresponding end/beg interior face, preserving discrete ∇·B (the
   face values are periodic, so the divergence sum telescopes correctly
   across the periodic boundary). Tangential components also wrap with
   no special treatment. */
static void mag_fill_periodic(XmfRun *run, int face) {
  int d = face / 2;
  int side = face % 2;
  int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
  int nint_d = (d == 0) ? run->nx : (d == 1) ? run->ny : run->nz;
  int ng = run->ng;
  int f1, f2, f_d;
  int comp;
  for (comp = 0; comp < 3; comp++) {
    /* For the face's own direction d, only B_d has face ghosts; the
       tangential components B_d1, B_d2 are cell-centred along d and
       their ghosts are filled by the transverse loops below. */
    for (f2 = 0; f2 < mag_ntot(run, d2) + (comp == d2 ? 1 : 0); f2++)
      for (f1 = 0; f1 < mag_ntot(run, d1) + (comp == d1 ? 1 : 0); f1++) {
        int cc[3], cc_src[3];
        cc[d1] = f1;
        cc[d2] = f2;
        cc_src[d1] = f1;
        cc_src[d2] = f2;
        if (comp == d) {
          /* B_d: face-centred along d, range 0..ntot[d] */
          for (f_d = 0; f_d <= mag_ntot(run, d); f_d++) {
            int is_ghost = (f_d < ng) || (f_d > ng + nint_d);
            if (!is_ghost)
              continue;
            cc[d] = f_d;
            cc_src[d] = (f_d < ng) ? f_d + nint_d : f_d - nint_d;
            run->bf[comp][mag_bidx(run, comp, cc)] =
                run->bf[comp][mag_bidx(run, comp, cc_src)];
          }
        } else {
          /* B_d1 or B_d2: cell-centred along d, range 0..ntot[d]-1 */
          for (f_d = 0; f_d < mag_ntot(run, d); f_d++) {
            int is_ghost = (f_d < ng) || (f_d >= ng + nint_d);
            if (!is_ghost)
              continue;
            cc[d] = f_d;
            cc_src[d] = (f_d < ng) ? f_d + nint_d : f_d - nint_d;
            run->bf[comp][mag_bidx(run, comp, cc)] =
                run->bf[comp][mag_bidx(run, comp, cc_src)];
          }
        }
      }
  }
  (void)side; /* side unused: periodic fills both beg and end in one pass */
}

void mag_fill_face(XmfRun *run, int face, int reflect) {
  if (!run || !run->bf[0])
    return; /* magnetic gate closed: nothing to fill */
  if (reflect)
    mag_fill_reflect(run, face);
  else if (run->bc[face] == XMF_BC_USER)
    mag_fill_user(run, face);
  else if (run->bc[face] == XMF_BC_PERIODIC)
    mag_fill_periodic(run, face);
  else
    mag_fill_copy(run, face); /* copy / outflow / none: zero-order */
}

/* ---- M4 one-shot finite-volume induction ---------------------------------
   Every RK stage (xmf_step orchestrates): BEFORE lf_all_dirs the
   stage-start ion velocities assemble u_+ (mag_up_snapshot); AFTER
   lf_all_dirs and BEFORE the source step ONE induction advances B,
       d_t B = curl(u_+ x B),
   the same PDE the card's 1D line discretises (with the interface EMF
   understood as G = u_+ x B there). Interface values are arithmetic
   neighbour averages -- NOT upwind-constrained transport (UCT is M5,
   together with the edge-centred E, which this slice leaves zero).
   Staggering: the EMF G~_{d2} entering the d1 derivative of the B_d
   row sits at the d1 FACE -- the arithmetic average over the cells
   sharing that face AND B_d's own face plane (a 2x2 cell average, the
   1D pair of neighbours in a 1D run), so the difference below is a
   centred divergence (a cell-centre EMF would be a half-cell-shifted
   forward difference -- anti-diffusive, unstable under every RK).
   Geometry rides the mesh[].A / dV factors of the M3 discrete
   divergence: the derivative along direction e of the face values G~
   at faces c_e and c_e+1 is
       D_e = (A_e[c_e+1] G~[c_e+1] - A_e[c_e] G~[c_e]) / dV_e[c_e]
   (cartesian A = 1, dV = dx, so a uniform 1D grid reduces to the
   card's (G~_{i+1} - G~_i)/dx with G~ the two-neighbour face average).
   All loops walk k,j,i; ndim >= 2 runs
   the very same curl -- no hard-coded 1D indices. A purely 1D
   cartesian run keeps the normal component B1 fixed: its two transverse
   derivative terms vanish identically on the transversely uniform
   state, so the update adds exactly 0.0. Host-mirror C loops, same
   layer as the M3 ghost fills: pull -> snapshot/induce -> push. */

/* Charge-weighted ion velocity snapshot u_+ = sum q_s n_s u_s /
   sum q_s n_s -- the very u_e formula of the B1 closure (sweep_elec_line
   / cell_signal_sum), assembled from the STAGE-START ion fields. Cells
   without an ion above the density floor (the n_e <= 0 fallback once
   the cons->prim floor is accounted for) carry u_+ = 0, so their
   induction increment vanishes. */
void mag_up_snapshot(XmfRun *run) {
  int f;
  size_t c;
  double *upx, *upy, *upz;
  if (!run || !run->bf[0] || !run->mag_up)
    return; /* magnetic gate closed: no u_+, no induction */
  /* Port D: device path computes u_+ on device Views from dv->fields;
     no pull needed. The host mirror run->mag_up becomes "last pulled"
     state (unpack helpers pull on read). */
  if (run->dev) {
    xmf_dev_mag_up_snapshot(run);
    return;
  }
  xmf_dev_pull(run); /* stage-start fields: device -> mirror */
  upx = run->mag_up;
  upy = upx + run->ncells;
  upz = upy + run->ncells;
  for (c = 0; c < run->ncells; c++) {
    double ne = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
    int ionized = 0;
    /* Degenerate directions (n_d == 1) are transversely uniform by
       construction, but their fluid ghost slices are never swept and a
       BC-none face never refills them, so they lag the active slice
       after the first step. The induction's transverse derivatives read
       exactly those ghost slices; reading the stale values there would
       fabricate a spurious curl (a 1D run would write its normal B).
       Read the ACTIVE slice instead -- the physical state of a
       degenerate direction. The B ghosts do not need this mirror: the
       magnetic ghost fills (mag_fill_*) pad them every fill_bc. */
    int i = (int)(c % (size_t)run->nxtot);
    int j = (int)((c / (size_t)run->nxtot) % (size_t)run->nytot);
    int k = (int)(c / ((size_t)run->nxtot * (size_t)run->nytot));
    int ir = (run->nx == 1) ? run->ng : i;
    int jr = (run->ny == 1) ? run->ng : j;
    int kr = (run->nz == 1) ? run->ng : k;
    size_t cr = cidx(run, kr, jr, ir);
    for (f = 0; f < run->nfluid; f++) {
      double ns;
      if (run->charge[f] <= 0.0)
        continue;
      if (fld(run, f, XMF_SLOT_RHO)[cr] > XMF_RHO_FLOOR)
        ionized = 1;
      ns =
          run->charge[f] * fld(run, f, XMF_SLOT_RHO)[cr] / run->mass_code[f];
      ne += ns;
      mx += ns * fld(run, f, XMF_SLOT_VX)[cr];
      my += ns * fld(run, f, XMF_SLOT_VY)[cr];
      mz += ns * fld(run, f, XMF_SLOT_VZ)[cr];
    }
    if (ionized && ne > 0.0) {
      upx[c] = mx / ne;
      upy[c] = my / ne;
      upz[c] = mz / ne;
    } else {
      upx[c] = 0.0;
      upy[c] = 0.0;
      upz[c] = 0.0;
    }
  }
}

/* G_comp at the intersection of B_d's face plane (face index cc[d]
   along d) with the DERIVATIVE direction de's face (face index cc[de]
   along de): the arithmetic average of the four cells sharing both
   faces -- (cc[d]-1, cc[d]) along d times (cc[de]-1, cc[de]) along de,
   the third coordinate at cc. The average along d interpolates the
   cell-centred G onto B_d's face plane (the card's neighbour-average
   interface EMF); the average along de places it ON the derivative
   face, so the transverse difference below is a genuine centred
   divergence in the mesh[].A / dV factors -- averaging along d only
   would leave the flux at cell-centre positions and turn the face
   difference into a half-cell-shifted forward difference (positive
   Fourier real part: anti-diffusion, unstable under every RK). cc's d
   entry is the FACE index, the other two entries are cell indices; the
   caller keeps cc[d] in [1, ntot[d]-1] and cc[de] in [1, ntot[de]-1]
   so all four cells exist. */
static double mag_g_edge(const double *G, const XmfRun *run, int d,
                         const int cc[3], int de, int comp) {
  int lo[3] = {cc[0], cc[1], cc[2]};
  size_t n = run->ncells;
  double s = 0.0;
  int a, b;
  for (a = 0; a < 2; a++) {
    lo[d] = cc[d] - 1 + a;
    for (b = 0; b < 2; b++) {
      lo[de] = cc[de] - 1 + b;
      s += G[(size_t)comp * n + cidx(run, lo[2], lo[1], lo[0])];
    }
  }
  return 0.25 * s;
}

/* Host E ghost edge fill (Port B extraction from mag_induce). Fills the
   non-periodic ghost edges by zero-order copy from the nearest interior
   edge. Three passes per component maintain the d1 → d2 → corner
   dependency order. Periodic ghost edges are already filled by the
   extended interior loop in mag_induce (wrapped 4-cell G average). */
static void mag_fill_e_ghosts_host(XmfRun *run) {
  int d, k, i, j;
  for (d = 0; d < 3; d++) {
    int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
    int nd = mag_ntot(run, d);
    int nd1 = mag_ntot(run, d1);
    int nd2 = mag_ntot(run, d2);
    int beg_periodic = (run->bc[2 * d1] == XMF_BC_PERIODIC);
    int end_periodic = (run->bc[2 * d1 + 1] == XMF_BC_PERIODIC);
    int beg_periodic_d2 = (run->bc[2 * d2] == XMF_BC_PERIODIC);
    int end_periodic_d2 = (run->bc[2 * d2 + 1] == XMF_BC_PERIODIC);
    for (k = 0; k < nd; k++) {
      int cc_ghost[3], cc_src[3];
      cc_ghost[d] = k;
      cc_src[d] = k;
      /* Pass 1: d1 ghost edges over interior d2 range (1..nd2-1). */
      for (i = 1; i < nd2; i++) {
        cc_ghost[d2] = i;
        cc_src[d2] = i;
        if (!beg_periodic) {
          cc_ghost[d1] = 0;
          cc_src[d1] = 1;
          run->ef[d][mag_eidx(run, d, cc_ghost)] =
              run->ef[d][mag_eidx(run, d, cc_src)];
        }
        if (!end_periodic) {
          cc_ghost[d1] = nd1;
          cc_src[d1] = nd1 - 1;
          run->ef[d][mag_eidx(run, d, cc_ghost)] =
              run->ef[d][mag_eidx(run, d, cc_src)];
        }
      }
      /* Pass 2: d2 ghost edges over interior d1 range (1..nd1-1). */
      for (j = 1; j < nd1; j++) {
        cc_ghost[d1] = j;
        cc_src[d1] = j;
        if (!beg_periodic_d2) {
          cc_ghost[d2] = 0;
          cc_src[d2] = 1;
          run->ef[d][mag_eidx(run, d, cc_ghost)] =
              run->ef[d][mag_eidx(run, d, cc_src)];
        }
        if (!end_periodic_d2) {
          cc_ghost[d2] = nd2;
          cc_src[d2] = nd2 - 1;
          run->ef[d][mag_eidx(run, d, cc_ghost)] =
              run->ef[d][mag_eidx(run, d, cc_src)];
        }
      }
      /* Pass 3: corner ghost edges. */
      {
        int d1_ghost[2] = {0, nd1};
        int d2_ghost[2] = {0, nd2};
        int di, dj;
        for (di = 0; di < 2; di++) {
          for (dj = 0; dj < 2; dj++) {
            int d1_is_beg = (d1_ghost[di] == 0);
            int d2_is_beg = (d2_ghost[dj] == 0);
            int d1_per = d1_is_beg ? beg_periodic : end_periodic;
            int d2_per = d2_is_beg ? beg_periodic_d2 : end_periodic_d2;
            if (d1_per && d2_per)
              continue;
            cc_ghost[d1] = d1_ghost[di];
            cc_ghost[d2] = d2_ghost[dj];
            cc_src[d1] = d1_is_beg ? 1 : nd1 - 1;
            cc_src[d2] = d2_is_beg ? 1 : nd2 - 1;
            run->ef[d][mag_eidx(run, d, cc_ghost)] =
                run->ef[d][mag_eidx(run, d, cc_src)];
          }
        }
      }
    }
  }
}

/* mag_g_edge with periodic wrapping of cell indices (M6 E-ghost fix).
   When per[dir] is nonzero, out-of-range indices along that direction
   wrap by ntot[dir] (the total array size including ghosts). For a
   periodic box the ghost-edge E value must be computed from the 4 cells
   that STRADDLE the ghost face, wrapping across the periodic boundary;
   a simple copy from an interior edge misses the wrap and introduces an
   O(dx) error. cc[d3] is a FACE index in mag_eidx (range [0, ntot[d3]]),
   but the G array uses CELL indices (range [0, ntot[d3]-1]); the far
   face c[d3]=ntot[d3] is clamped to ntot[d3]-1 (the last ghost cell,
   which mirrors the first interior cell for periodic boundaries). */
static double mag_g_edge_p(const double *G, const XmfRun *run, int d,
                           const int cc[3], int de, int comp,
                           const int per[3]) {
  int d3 = 3 - d - de;
  size_t n = run->ncells;
  double s = 0.0;
  int a, b;
  int ntot_de = (de == 0) ? (int)run->nxtot
                          : (de == 1) ? (int)run->nytot : (int)run->nztot;
  int ntot_d = (d == 0) ? (int)run->nxtot
                        : (d == 1) ? (int)run->nytot : (int)run->nztot;
  int ntot_d3 = (d3 == 0) ? (int)run->nxtot
                          : (d3 == 1) ? (int)run->nytot : (int)run->nztot;
  for (a = 0; a < 2; a++) {
    int id = cc[d] - 1 + a;
    if (per[d] && id < 0)
      id += ntot_d;
    else if (per[d] && id >= ntot_d)
      id -= ntot_d;
    for (b = 0; b < 2; b++) {
      int ide = cc[de] - 1 + b;
      int lo[3];
      int id3;
      if (per[de] && ide < 0)
        ide += ntot_de;
      else if (per[de] && ide >= ntot_de)
        ide -= ntot_de;
      id3 = cc[d3];
      if (id3 >= ntot_d3)
        id3 = ntot_d3 - 1;
      lo[d] = id;
      lo[de] = ide;
      lo[d3] = id3;
      s += G[(size_t)comp * n + cidx(run, lo[2], lo[1], lo[0])];
    }
  }
  return 0.25 * s;
}

/* ---- M5 constrained transport (CT) ----------------------------------------
   Replaces the M4 one-shot finite-volume induction. The numerical
   structure is identical to M4 (cell-centred G = u_+ x B averaged to
   edges via the 4-cell mag_g_edge stencil): the arithmetic tier's
   stability requires averaging the NONLINEAR PRODUCT G at cells, not
   the separate u_+ and B factors (averaging separately introduces
   O(dx) cross-terms near discontinuities that cause exponential
   growth). M5 adds: edge-centred E blocks filled with E = -G (the
   ideal-MHD interface EMF) for diagnostics and future UCT upgrade,
   and a divB diagnostic (xmf_divb_max). Returns 0, or -1 on
   allocation failure.

   Port C B-pin ban: this function advances B by CT. The fluid pin mask
   (xmf_bc_set_pin / xmf_pin_apply_*) pins the fluid five slots ONLY;
   it does NOT and MUST NOT pin B here or after this call. Extending pin
   to B is a future slice that needs its own hook AFTER mag_induce, not
   a drive-by addition to the fluid pin apply. Do not add B writes to
   xmf_pin_apply_device / xmf_pin_apply_host. */
int mag_induce(XmfRun *run, double dt) {
  int d, i, j, k;
  double *G;
  if (!run || !run->bf[0] || !run->mag_up)
    return 0; /* magnetic gate closed: nothing to induce */
  if (dt == 0.0)
    return 0;
  /* Port D: device path performs full CT induction on device Views
     (G computation, edge E fill, curl update, E ghost fill) without any
     pull/push roundtrip. The host mirror run->bf / run->ef becomes "last
     pulled" state; unpack helpers pull on read. The host fallback path
     below keeps the pull -> compute -> push cadence for run->dev == NULL. */
  if (run->dev)
    return xmf_dev_mag_induce(run, dt);
  xmf_dev_pull_b(run); /* B: device -> mirror */
  xmf_dev_pull(run);   /* fields / mag_up: device -> mirror */
  G = (double *)malloc(3 * run->ncells * sizeof(double));
  if (!G)
    return -1;
  /* Cell-centred B and G = u_+ x B over every cell (k,j,i). */
  for (k = 0; k < run->nztot; k++)
    for (j = 0; j < run->nytot; j++)
      for (i = 0; i < run->nxtot; i++) {
        size_t c = cidx(run, k, j, i);
        int cc[3];
        int dd;
        double bcx, bcy, bcz, ux, uy, uz;
        cc[0] = i;
        cc[1] = j;
        cc[2] = k;
        bcx = 0.0;
        bcy = 0.0;
        bcz = 0.0;
        for (dd = 0; dd < 3; dd++) {
          int hi[3] = {cc[0], cc[1], cc[2]};
          double v;
          hi[dd] += 1; /* upper face of the cell along dd */
          v = 0.5 * (run->bf[dd][mag_bidx(run, dd, cc)] +
                     run->bf[dd][mag_bidx(run, dd, hi)]);
          if (dd == 0)
            bcx = v;
          else if (dd == 1)
            bcy = v;
          else
            bcz = v;
        }
        ux = run->mag_up[c];
        uy = run->mag_up[run->ncells + c];
        uz = run->mag_up[2 * run->ncells + c];
        G[c] = uy * bcz - uz * bcy;
        G[run->ncells + c] = uz * bcx - ux * bcz;
        G[2 * run->ncells + c] = ux * bcy - uy * bcx;
      }
  /* M5: fill edge-centred E blocks with E = -G averaged to the edge
     position (4-cell average matching the derivative stencil). Edges
     at e = 0 or e = ntot stay zero until the ghost pad below. */
  for (d = 0; d < 3; d++) {
    int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
    int nd = mag_ntot(run, d);
    int nd1 = mag_ntot(run, d1);
    int nd2 = mag_ntot(run, d2);
    /* Interior E fill. The 4-cell edge stencil reads cells at
       cc[dir]-1 and cc[dir] for each direction. At positions where
       cc[dir]=0 or cc[dir]=ntot[dir]-1, one of the cells is out of
       range. For periodic boundaries, the out-of-range cell wraps
       across the periodic boundary; for non-periodic, the ghost cell
       is filled by the BC fill. The periodic-wrap version
       (mag_g_edge_p) handles all cases correctly (non-periodic
       directions with per[dir]=0 simply don't wrap, and the cell
       access stays in range for interior positions). */
    {
      int per[3] = {0, 0, 0};
      per[0] = (run->bc[0] == XMF_BC_PERIODIC);
      per[1] = (run->bc[2] == XMF_BC_PERIODIC);
      per[2] = (run->bc[4] == XMF_BC_PERIODIC);
      for (k = 0; k < nd; k++)
        for (j = 1; j < nd1; j++)
          for (i = 1; i < nd2; i++) {
            int cc[3];
            cc[d] = k;
            cc[d1] = j;
            cc[d2] = i;
            run->ef[d][mag_eidx(run, d, cc)] =
                -mag_g_edge_p(G, run, d, cc, d1, d, per);
          }
    }
    /* M6 fix: for periodic ghost edges, compute E from the wrapped
       4-cell average instead of copying an interior edge. The copy
       misses the periodic wrap of cell indices and introduces an O(dx)
       bias (the uniform-field E2 deviation). */
    {
      int per[3] = {0, 0, 0};
      per[0] = (run->bc[0] == XMF_BC_PERIODIC);
      per[1] = (run->bc[2] == XMF_BC_PERIODIC);
      per[2] = (run->bc[4] == XMF_BC_PERIODIC);
      if (per[d1] || per[d2]) {
        for (k = 0; k < nd; k++) {
          int cc[3];
          cc[d] = k;
          /* d1 ghost edges over interior d2 range */
          if (per[d1]) {
            for (i = 1; i < nd2; i++) {
              cc[d2] = i;
              cc[d1] = 0;
              run->ef[d][mag_eidx(run, d, cc)] =
                  -mag_g_edge_p(G, run, d, cc, d1, d, per);
              cc[d1] = nd1;
              run->ef[d][mag_eidx(run, d, cc)] =
                  -mag_g_edge_p(G, run, d, cc, d1, d, per);
            }
          }
          /* d2 ghost edges over interior d1 range */
          if (per[d2]) {
            for (j = 1; j < nd1; j++) {
              cc[d1] = j;
              cc[d2] = 0;
              run->ef[d][mag_eidx(run, d, cc)] =
                  -mag_g_edge_p(G, run, d, cc, d1, d, per);
              cc[d2] = nd2;
              run->ef[d][mag_eidx(run, d, cc)] =
                  -mag_g_edge_p(G, run, d, cc, d1, d, per);
            }
          }
          /* Corner ghost edges (both d1 and d2 at ghost positions) */
          if (per[d1] && per[d2]) {
            int jd, id;
            for (jd = 0; jd <= 1; jd++) {
              for (id = 0; id <= 1; id++) {
                cc[d1] = jd * nd1; /* 0 or nd1 */
                cc[d2] = id * nd2; /* 0 or nd2 */
                run->ef[d][mag_eidx(run, d, cc)] =
                    -mag_g_edge_p(G, run, d, cc, d1, d, per);
              }
            }
          }
        }
      }
    }
    /* M5 E ghost pad: moved to Port B device dispatch below. The
       non-periodic ghost edges (zero-order copy from nearest interior)
       are filled by mag_fill_e_ghosts_host on the host path or by
       xmf_bc_fill_e_ghosts_device on the device path. Periodic ghost
       edges were already filled by the extended interior loop above. */
  }
  /* Port B: fill non-periodic E ghost edges. Host path fills on the
     mirror; device path fills after push_b (the curl reads G, not E,
     so the order relative to the curl doesn't matter). */
  if (!run->dev)
    mag_fill_e_ghosts_host(run);
  /* Discrete curl into every component's faces (k,j,i). */
  for (d = 0; d < 3; d++) {
    int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
    const double *A1 = run->mesh[d1].A;
    const double *dV1 = run->mesh[d1].dV;
    const double *A2 = run->mesh[d2].A;
    const double *dV2 = run->mesh[d2].dV;
    for (k = 0; k < run->nztot; k++)
      for (j = 0; j < run->nytot; j++)
        for (i = 0; i < run->nxtot; i++) {
          int cc[3] = {i, j, k};
          double D1 = 0.0, D2 = 0.0;
          if (cc[d] < 1 || cc[d] > mag_ntot(run, d) - 1)
            continue; /* face range of the d average stencil */
          if (cc[d1] < 1 || cc[d1] > mag_ntot(run, d1) - 2)
            continue;
          if (cc[d2] < 1 || cc[d2] > mag_ntot(run, d2) - 2)
            continue;
          if (dV1[cc[d1]] != 0.0) {
            int up[3] = {cc[0], cc[1], cc[2]};
            double g_lo = mag_g_edge(G, run, d, cc, d1, d2);
            up[d1] += 1;
            D1 = (A1[cc[d1] + 1] * mag_g_edge(G, run, d, up, d1, d2) -
                  A1[cc[d1]] * g_lo) /
                 dV1[cc[d1]];
          }
          if (dV2[cc[d2]] != 0.0) {
            int up[3] = {cc[0], cc[1], cc[2]};
            double g_lo = mag_g_edge(G, run, d, cc, d2, d1);
            up[d2] += 1;
            D2 = (A2[cc[d2] + 1] * mag_g_edge(G, run, d, up, d2, d1) -
                  A2[cc[d2]] * g_lo) /
                 dV2[cc[d2]];
          }
          run->bf[d][mag_bidx(run, d, cc)] += dt * (D1 - D2);
        }
  }
  free(G);
  xmf_dev_push_b(run); /* B + E interior: mirror -> device */
  /* Port B: fill non-periodic E ghost edges on device (the interior +
     periodic ghost E values were just pushed). */
  if (run->dev)
    xmf_bc_fill_e_ghosts_device(run);
  return 0;
}

/* ---- M5 divB diagnostic (report-only, C-IDF-4 form) ----------------------
   Maximum normalised discrete divergence over the interior cells. Per
   cell: (div B)_c = sum_d (A_d[f_d+1] B_d[f_d+1] - A_d[f_d] B_d[f_d])
   / dV_c, normalised by the largest single absolute term in the sum
   (dimensionless, 0 for a constructively divergence-free field). The
   threshold 1e-6 is enforced in the test suite, not here -- this
   function is pure report, never aborts. Returns 0 when the gate is
   closed or all cells have zero-scale divergence. Public API:
   xmf_core.h declares it as xmf_divb_max; the implementation lives here
   because it reads the staggered B blocks directly. */
double xmf_divb_max(const XmfRun *run) {
  int i, j, k, d;
  double max_ratio = 0.0;
  if (!run || !run->bf[0])
    return 0.0;
  /* Port D mirror contract: the device path writes only device Views;
     the host mirror run->bf is "last pulled" state. Pull B here so the
     diagnostic reads fresh values. This is a report path (not the step
     hot loop), same tier as the xmf_fields entry pull (xmf_core.c:477). */
  if (run->dev)
    xmf_dev_pull_b(run);
  for (k = run->ng; k < run->ng + run->nz; k++)
    for (j = run->ng; j < run->ng + run->ny; j++)
      for (i = run->ng; i < run->ng + run->nx; i++) {
        int c[3] = {i, j, k};
        double div_sum = 0.0, scale = 0.0;
        for (d = 0; d < 3; d++) {
          const double *Ad = run->mesh[d].A;
          const double *dVd = run->mesh[d].dV;
          int c_out[3];
          double bout, bin_, term, aterm;
          if (dVd[c[d]] == 0.0)
            continue;
          c_out[0] = c[0];
          c_out[1] = c[1];
          c_out[2] = c[2];
          c_out[d] = c[d] + 1;
          bout = run->bf[d][mag_bidx(run, d, c_out)];
          bin_ = run->bf[d][mag_bidx(run, d, c)];
          term = (Ad[c[d] + 1] * bout - Ad[c[d]] * bin_) / dVd[c[d]];
          div_sum += term;
          aterm = term < 0.0 ? -term : term;
          if (aterm > scale)
            scale = aterm;
        }
        if (scale > 0.0) {
          double ratio = (div_sum < 0.0 ? -div_sum : div_sum) / scale;
          if (ratio > max_ratio)
            max_ratio = ratio;
        }
      }
  return max_ratio;
}

/* ---- M4 B SSP stage helpers ----------------------------------------------
   The face-centred B needs its own stage buffers: the fluid U0/Us
   length stays untouched and cons_combine never carries the magnetic
   blocks. Three device blocks (one per component, ghosts included,
   xmf_dev_dvec lengths xmf_mag_len_b) hold B^n; every convex
   combination lands beside the fluid cons_combine with the very same
   coefficients (Heun 1/2-1/2, Shu-Osher 3/4-1/4 and 1/3-2/3). Gate
   closed: nothing is allocated and every helper below is a no-op.
   RK1 stays in-place (xmf_step never allocates the buffers there). */

int mag_b_stage_alloc(const XmfRun *run, void **buf) {
  int d;
  if (!buf)
    return -1;
  buf[0] = NULL;
  buf[1] = NULL;
  buf[2] = NULL;
  if (!run || !run->bf[0])
    return 0; /* gate closed: no buffers needed */
  for (d = 0; d < 3; d++) {
    buf[d] = xmf_dev_dvec(xmf_mag_len_b(run, d));
    if (!buf[d]) {
      mag_b_stage_free(buf);
      return -1;
    }
  }
  return 0;
}

void mag_b_stage_free(void **buf) {
  int d;
  if (!buf)
    return;
  for (d = 0; d < 3; d++) {
    xmf_dev_dfree(buf[d]);
    buf[d] = NULL;
  }
}

/* Copy the current device bf into the stage buffers (B^n beside
   fields_to_cons(U0)). */
int mag_b_snapshot(const XmfRun *run, void **dst) {
  if (!run || !run->bf[0] || !dst || !dst[0])
    return 0; /* gate closed (or nothing allocated): nothing to snapshot */
  return xmf_dev_mag_stage_snapshot(run, dst);
}

/* bf <- a*src0 + b*bf on the device, beside cons_combine(src0 = the
   B^n buffers). */
int mag_b_combine(const XmfRun *run, double a, void **src0, double b) {
  if (!run || !run->bf[0] || !src0 || !src0[0])
    return 0; /* gate closed (or nothing allocated): nothing to combine */
  return xmf_dev_mag_stage_combine(run, a, src0, b);
}
