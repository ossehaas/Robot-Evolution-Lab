"""Numerical cross-check of the TPU ankle spring and the TPU sole spring of the dog (linear elastic FE, numpy/scipy).

    lab -m tools.dog.ankle_fe [prototypes/puppy/genome.json] [--out runs/exp2_dog/1_design/ankle_fe.md] [--quick]

No FE package is installed in the lab environment, so this is a small self-contained solver on structured grids:

    3D      8-node hexahedra on voxels of the real CAD profiles from core/morphology/dog_paw.py (the spring's
            I-profile extruded along y; the sole's arm layer extruded through its thickness)
    axi     4-node axisymmetric quads (r-z), used for the rejected alternative (a hollow round paw)

Nearly incompressible TPU (nu = 0.45) locks plain low-order elements, so all models use selective reduced integration
(deviatoric part 2x2(x2) Gauss points, volumetric part 1 point). Linear analysis: stiffness at small deflection; the
large-strain softening of TPU (secant modulus drops with strain) is discussed in the README. The calibration constants
in dog_paw.py (SPRING_K_FE, SPRING_EPS_FE, SOLE_K_REF, SOLE_K_EXP, SOLE_EPS) come from these models.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E_TPU = 33.6  # MPa, Polymaker PolyFlex TPU95 TDS V5.4, Young's modulus (X-Y), ISO 37
NU_TPU = 0.45

# ---------------------------------------------------------------------- material matrices


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def split_D(E, nu, n):
    """Deviatoric and volumetric parts of the isotropic elasticity matrix (Voigt, n normal components)."""
    lam, mu = lame(E, nu)
    K = lam + 2 * mu / 3  # bulk modulus
    m = np.zeros(n + (3 if n == 3 else 1))
    m[:n] = 1.0
    Dvol = K * np.outer(m, m)
    Dfull = np.zeros((len(m), len(m)))
    Dfull[:n, :n] = lam
    Dfull += np.diag([2 * mu] * n + [mu] * (len(m) - n))
    return Dfull - Dvol, Dvol


# ---------------------------------------------------------------------- 2D plane / axisymmetric quads


def _q4_shape(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                          [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    return N, dN


def grid2d(mask: np.ndarray, h: float, x0: float, z0: float):
    """Nodes and quads of a masked grid. mask[i, k]: cell between x0 + i*h and x0 + (i+1)*h, z likewise."""
    nx, nz = mask.shape
    ii, kk = np.nonzero(mask)
    nid = np.full((nx + 1, nz + 1), -1)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for di, dk in corners:
        nid[ii + di, kk + dk] = 0
    used = np.argwhere(nid == 0)
    nid[used[:, 0], used[:, 1]] = np.arange(len(used))
    xy = np.column_stack([x0 + used[:, 0] * h, z0 + used[:, 1] * h])
    conn = np.column_stack([nid[ii + di, kk + dk] for di, dk in corners])
    return xy, conn


def assemble2d(xy, conn, h, E, nu, mode: str):
    """Global stiffness; mode = 'strain' | 'stress' | 'axi' (axisymmetric, thickness = 2*pi*r per radian... full ring)."""
    ne = len(conn)
    gp = [(-1 / math.sqrt(3), -1 / math.sqrt(3)), (1 / math.sqrt(3), -1 / math.sqrt(3)),
          (1 / math.sqrt(3), 1 / math.sqrt(3)), (-1 / math.sqrt(3), 1 / math.sqrt(3))]
    J = h / 2.0  # square cells
    if mode == "stress":
        D = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        Ddev, Dvol, n_norm = D, None, 2
    elif mode == "strain":
        Ddev3, Dvol3 = split_D(E, nu, 3)  # use the 3D split, keep xx, zz, (yy = 0), xz
        idx = [0, 2, 4]  # (xx, zz, xz) out of (xx, yy, zz, xy, yz, xz) ordering below
        Ddev, Dvol, n_norm = None, None, 2
        lam, mu = lame(E, nu)
        Kb = lam + 2 * mu / 3
        Dfull = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
        Dvol = Kb * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        Ddev = Dfull - Dvol
    else:  # axi: strains (rr, zz, tt, rz)
        lam, mu = lame(E, nu)
        Kb = lam + 2 * mu / 3
        Dfull = np.zeros((4, 4))
        Dfull[:3, :3] = lam
        Dfull += np.diag([2 * mu, 2 * mu, 2 * mu, mu])
        m = np.array([1.0, 1.0, 1.0, 0.0])
        Dvol = Kb * np.outer(m, m)
        Ddev = Dfull - Dvol
    X = xy[conn]  # ne x 4 x 2
    Ke = np.zeros((ne, 8, 8))

    def Bmat(xi, eta):
        N, dN = _q4_shape(xi, eta)
        dNx = dN / J  # d/dx, d/dz (square cells, axis aligned)
        nrow = 4 if mode == "axi" else 3
        B = np.zeros((ne, nrow, 8))
        B[:, 0, 0::2] = dNx[0]
        B[:, 1, 1::2] = dNx[1]
        if mode == "axi":
            r = X[:, :, 0] @ N  # ne
            r = np.maximum(r, 1e-6)
            B[:, 2, 0::2] = N[None, :] / r[:, None]
            B[:, 3, 0::2] = dNx[1]
            B[:, 3, 1::2] = dNx[0]
            w = 2 * math.pi * r * J * J
        else:
            B[:, 2, 0::2] = dNx[1]
            B[:, 2, 1::2] = dNx[0]
            w = np.full(ne, J * J)
        return B, w

    for xi, eta in gp:
        B, w = Bmat(xi, eta)
        Ke += np.einsum("e,eji,jk,ekl->eil", w, B, Ddev, B)
    if Dvol is not None:
        B, w = Bmat(0.0, 0.0)
        Ke += np.einsum("e,eji,jk,ekl->eil", 4 * w, B, Dvol, B)
    dofs = np.stack([2 * conn, 2 * conn + 1], axis=2).reshape(ne, 8)
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    n = 2 * len(xy)
    return sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)), Ke, dofs


def solve_prescribed(K, fixed_dofs, fixed_vals):
    n = K.shape[0]
    u = np.zeros(n)
    fixed_dofs = np.asarray(fixed_dofs)
    u[fixed_dofs] = fixed_vals
    free = np.setdiff1d(np.arange(n), fixed_dofs)
    Kff = K[free][:, free].tocsc()
    rhs = -K[free][:, fixed_dofs] @ u[fixed_dofs]
    u[free] = spla.spsolve(Kff, rhs)
    f = K @ u
    return u, f


# ---------------------------------------------------------------------- hollow paw, axisymmetric


def paw_axi(t_shell, R=21.0, zc=2.0, sole=-18.0, apex=-1.25, pocket_r=11.45, pocket_z0=1.25, top=16.0, h=0.15,
            E=E_TPU, nu=NU_TPU, friction=True, t_side=None):
    """Vertical stiffness of the hollow paw on a flat floor (body of revolution about the vertical axis; paw frame:
    paw centre at z = 0, sole at z = -18). Outer: sphere R about (0, zc) cut flat at the sole and the top. Cavity:
    sphere R - t_shell (same centre) below a 45 deg cone roof with its apex at ``apex``. Pocket for the spring's
    bottom block (radius of equal area) from pocket_z0 up; its floor and wall are held (the spring block sits there).
    The flat sole is pushed up uniformly (rigid floor, sticking if friction else sliding)."""
    rmax = R + 0.5
    nr, nz = int(math.ceil(rmax / h)), int(math.ceil((top - sole) / h))
    rc = (np.arange(nr) + 0.5) * h
    zcc = sole + (np.arange(nz) + 0.5) * h
    Rr, Zz = np.meshgrid(rc, zcc, indexing="ij")
    outer = (Rr ** 2 + (Zz - zc) ** 2 <= R ** 2)
    Ri = R - t_shell
    cav = (Rr ** 2 + (Zz - zc) ** 2 <= Ri ** 2) & (Zz <= apex - Rr)  # 45 deg cone, apex on the axis
    if t_side is not None:  # thicker side walls: inner sphere radius reduced at the sides (ellipsoid-like)
        a, c = R - t_side, R - t_shell
        cav &= (Rr / a) ** 2 + ((Zz - zc) / c) ** 2 <= 1.0
    pocket = (Rr <= pocket_r) & (Zz >= pocket_z0)
    mask = outer & ~cav & ~pocket
    xy, conn = grid2d(mask, h, 0.0, sole)
    K, _, _ = assemble2d(xy, conn, h, E, nu, "axi")
    r, z = xy[:, 0], xy[:, 1]
    eps = 1e-6
    tol = 0.6 * h  # nodes on the pocket boundary (the grid need not line up with it)
    held = (r <= pocket_r + tol) & (z >= pocket_z0 - tol)
    soleN = np.abs(z - sole) < eps
    axis = r < eps
    nh, ns, na = np.nonzero(held)[0], np.nonzero(soleN)[0], np.nonzero(axis & ~soleN & ~held)[0]
    d = 1e-3
    fd = [2 * nh, 2 * nh + 1, 2 * ns + 1, 2 * na]
    fv = [np.zeros(2 * len(nh)), np.full(len(ns), d), np.zeros(len(na))]
    if friction:
        fd.append(2 * ns)
        fv.append(np.zeros(len(ns)))
    else:
        on_axis = ns[r[ns] < eps]
        fd.append(2 * on_axis)
        fv.append(np.zeros(len(on_axis)))
    fd, fv = np.concatenate(fd), np.concatenate(fv)
    fd, idx = np.unique(fd, return_index=True)
    u, f = solve_prescribed(K, fd, fv[idx])
    F = float(f[2 * ns + 1].sum())
    flat_r = math.sqrt(max(R * R - (sole - zc) ** 2, 0.0))
    return dict(k=F / d, flat_r=flat_r, n_dof=2 * len(xy))


def reissner(t, R=21.0, E=E_TPU, nu=NU_TPU):
    """Point load on a thin spherical shell (Reissner): k = 4 E t^2 / (R sqrt(3 (1 - nu^2)))."""
    return 4 * E * t * t / (R * math.sqrt(3 * (1 - nu * nu)))


# ---------------------------------------------------------------------- CAD profiles from core/morphology/dog_paw.py


def raster(cs, h, hy=None, anchor=(0.0, 0.0)):
    """Cells inside a manifold3d CrossSection: mask[i, j], x0, y0. Cell size h (x) by hy (y, default h); the grid lines
    pass through ``anchor`` so that faces placed on grid lines (e.g. the faces of the spring's blade) are exact."""
    from matplotlib.path import Path as MPath

    hy = h if hy is None else hy
    bx0, by0, x1, y1 = cs.bounds()
    x0 = anchor[0] - math.ceil((anchor[0] - bx0) / h - 1e-9) * h
    y0 = anchor[1] - math.ceil((anchor[1] - by0) / hy - 1e-9) * hy
    nx, ny = int(math.ceil((x1 - x0) / h - 1e-9)), int(math.ceil((y1 - y0) / hy - 1e-9))
    xc = x0 + (np.arange(nx) + 0.5) * h
    yc = y0 + (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.zeros(len(pts), bool)
    for poly in cs.to_polygons():  # outer loops and holes do not overlap: even-odd = xor
        inside ^= MPath(np.asarray(poly)).contains_points(pts)
    return inside.reshape(nx, ny), x0, y0


def hex_model(cells, size, origin, E=E_TPU, nu=NU_TPU):
    """Voxel model: cells (n x 3 int), cell size (hx, hy, hz), origin -> K, node coordinates, connectivity."""
    Ke, corners = _hex_ke_box(size, E, nu)
    cells = np.asarray(cells, int)
    cidx = cells[:, None, :] + corners[None, :, :].astype(int)
    keys = cidx.reshape(-1, 3)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    conn = inv.reshape(len(cells), 8)
    P = np.asarray(origin, float) + uniq * np.asarray(size, float)
    dofs = np.stack([3 * conn, 3 * conn + 1, 3 * conn + 2], axis=2).reshape(len(conn), 24)
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    n = 3 * len(P)
    K = sp.csr_matrix((np.tile(Ke.ravel(), len(conn)), (rows, cols)), shape=(n, n))
    return K, P, conn


def _hex_ke_box(size, E, nu):
    Ddev, Dvol = split_D(E, nu, 3)
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    s = 2 * corners - 1
    size = np.asarray(size, float)

    def B(xi):
        dN = np.zeros((3, 8))
        for a in range(8):
            for d in range(3):
                o = [1 + s[a, k] * xi[k] for k in range(3) if k != d]
                dN[d, a] = 0.125 * s[a, d] * o[0] * o[1]
        dNx = dN / (size[:, None] / 2)
        Bm = np.zeros((6, 24))
        Bm[0, 0::3], Bm[1, 1::3], Bm[2, 2::3] = dNx[0], dNx[1], dNx[2]
        Bm[3, 0::3], Bm[3, 1::3] = dNx[1], dNx[0]
        Bm[4, 1::3], Bm[4, 2::3] = dNx[2], dNx[1]
        Bm[5, 0::3], Bm[5, 2::3] = dNx[2], dNx[0]
        return Bm

    detJ = np.prod(size / 2)
    g = 1 / math.sqrt(3)
    Ke = np.zeros((24, 24))
    for xi in [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]:
        Bm = B(xi)
        Ke += Bm.T @ Ddev @ Bm * detJ
    Bm = B((0.0, 0.0, 0.0))
    Ke += Bm.T @ Dvol @ Bm * detJ * 8
    return Ke, corners


def _strain_max_hex(P, conn, u, size):
    s = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 2 - 1
    dN = 0.125 * s.T  # derivative at the cell centre: dN_a/dxi_d = s_ad / 8
    dNx = dN / (np.asarray(size, float)[:, None] / 2)
    U = np.stack([u[3 * conn], u[3 * conn + 1], u[3 * conn + 2]], axis=2)
    G = np.einsum("da,eac->ecd", dNx, U)
    Eps = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    w = np.linalg.eigvalsh(Eps)
    return np.abs(w).max(axis=1)


def solve_hex(K, fixed, values):
    n = K.shape[0]
    u = np.zeros(n)
    u[fixed] = values
    free = np.setdiff1d(np.arange(n), fixed)
    lu = spla.splu(K[free][:, free].tocsc(), permc_spec="COLAMD")
    u[free] = lu.solve(-(K[free][:, fixed] @ u[fixed]))
    return u, K @ u


def spring_cad_3d(t, L, n_t=14, n_L=26, hy=1.0, bc="faces", cases=("pitch", "roll", "yaw", "axial", "shear_x", "shear_y")):
    """The dog_paw spring profile (with its fillets and bolt holes), extruded along y: pitch/roll/yaw/axial/shear.
    The voxels are t/n_t by L/n_L, with grid lines on the blade's faces (x = +-t/2) and roots (z = +-L/2), so the blade
    is modelled at its true size (k ~ t^3: a 0.25 mm error in t is a 10 % error in k)."""
    from core.morphology import dog_paw as DP

    hx, hz = t / n_t, L / n_L
    mask, x0, z0 = raster(DP.spring_profile(t, L), hx, hz, anchor=(t / 2, L / 2))
    h = (hx, hz)
    ny = int(round(DP.SPRING_W / hy))
    ii, kk = np.nonzero(mask)
    cells = np.array([(i, j, k) for j in range(ny) for i, k in zip(ii, kk)])
    K, P, conn = hex_model(cells, (hx, hy, hz), (x0, -DP.SPRING_W / 2, z0))
    x, y, z = P.T
    eps = 1e-6
    zt, zb = z.max(), z.min()
    top = z >= zt - eps
    bot = z <= zb + eps
    if bc == "faces":
        side = np.abs(x) >= np.abs(x).max() - hx - eps  # outermost node columns of the blocks (rounded corners)
        top |= side & (z >= L / 2 + eps)
        bot |= side & (z <= -L / 2 - eps)
    nt, nb = np.nonzero(top)[0], np.nonzero(bot)[0]
    fixed = np.concatenate([3 * nt, 3 * nt + 1, 3 * nt + 2, 3 * nb, 3 * nb + 1, 3 * nb + 2])
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    lu = spla.splu(K[free][:, free].tocsc(), permc_spec="COLAMD")
    Kfp = K[free][:, fixed]
    out = {"n_dof": K.shape[0]}
    pb = P[nb]
    for name, axis in (("pitch", np.array([0, 1.0, 0])), ("roll", np.array([1.0, 0, 0])), ("yaw", np.array([0, 0, 1.0])),
                       ("axial", None), ("shear_x", None), ("shear_y", None)):
        if name not in cases:
            continue
        if name == "axial":
            ub = np.tile([0, 0, -1e-3], (len(nb), 1))
        elif name == "shear_x":
            ub = np.tile([1e-3, 0, 0], (len(nb), 1))
        elif name == "shear_y":
            ub = np.tile([0, 1e-3, 0], (len(nb), 1))
        else:
            ub = 1e-3 * np.cross(axis, pb)
        u = np.zeros(K.shape[0])
        u[3 * nb], u[3 * nb + 1], u[3 * nb + 2] = ub.T
        u[free] = lu.solve(-(Kfp @ u[fixed]))
        f = K @ u
        fb = np.column_stack([f[3 * nb], f[3 * nb + 1], f[3 * nb + 2]])
        if axis is None:
            out[name] = float(np.abs(fb.sum(axis=0)).max() / 1e-3)
        else:
            out[name] = float(np.cross(pb, fb).sum(axis=0) @ axis / 1e-3)
        if name == "pitch":
            e = _strain_max_hex(P, conn, u, (hx, hy, hz))
            out["strain_per_rad"] = float(e.max() / 1e-3)
    return out


def sole_cad_3d(t_arm, h=0.2, nz=6, slide=True, **profile):
    """The dog_paw sole disc (arms layer, t_arm thick): rim held, pad pushed up rigidly. k [N/mm], peak strain per mm."""
    from core.morphology import dog_paw as DP

    mask, x0, y0 = raster(DP.sole_arms_profile(**profile), h)
    hz = t_arm / nz
    ii, jj = np.nonzero(mask)
    cells = np.array([(i, j, k) for k in range(nz) for i, j in zip(ii, jj)])
    K, P, conn = hex_model(cells, (h, h, hz), (x0, y0, 0.0))
    r = np.hypot(P[:, 0], P[:, 1])
    rim = np.nonzero(r >= DP.SOLE_RIM_R[0] + 0.3)[0]
    pad = np.nonzero(r <= DP.SOLE_PAD_R - 0.6)[0]
    d = 1e-3
    fixed = [3 * rim, 3 * rim + 1, 3 * rim + 2, 3 * pad + 2]
    vals = [np.zeros(3 * len(rim)), np.full(len(pad), d)]
    if not slide:  # tread sticks to the floor: the pad cannot turn or shift
        fixed += [3 * pad, 3 * pad + 1]
        vals += [np.zeros(2 * len(pad))]
    else:  # pad rigid but free to turn about z (the three arcs make it twist as it rises): tie it in-plane
        pass
    fixed = np.concatenate(fixed)
    u, f = solve_hex(K, fixed, np.concatenate(vals))
    F = float(f[3 * pad + 2].sum())
    e = _strain_max_hex(P, conn, u, (h, h, hz))
    # ignore the cells touching the prescribed pad region (artificial corner of the rigid pad)
    cc = P[conn].mean(axis=1)
    rc = np.hypot(cc[:, 0], cc[:, 1])
    arm = (rc > DP.SOLE_PAD_R + 0.2) & (rc < DP.SOLE_RIM_R[0] - 0.1)
    return dict(k=F / d, strain_per_mm=float(e[arm].max() / d), n_dof=K.shape[0])


# ---------------------------------------------------------------------- report


def main():
    from core.morphology import dog_paw as DP
    from core.morphology.genome import Genome

    ap = argparse.ArgumentParser(prog="python -m tools.dog.ankle_fe")
    ap.add_argument("genome", type=Path, nargs="?", default=Path("prototypes/puppy/genome.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="coarser meshes, skip the sweeps")
    args = ap.parse_args()
    lines = []

    def log(s=""):
        print(s)
        sys.stdout.flush()
        lines.append(s)

    g = Genome.load(args.genome)
    d = DP.design(g, 1000 * g.legs.paw_radius)
    E, w = DP.TPU_E, DP.SPRING_W
    log(f"# Ankle spring and soft paw: FE cross-check ({args.genome})")
    log("")
    log(f"Material: E = {E} MPa (Polymaker PolyFlex TPU95 TDS V5.4, Young's modulus X-Y, ISO 37), nu = {NU_TPU}; "
        "linear elastic, small strain. Solver: this file (structured voxel/quad meshes, selective reduced integration).")
    log("")
    log(f"## Ankle spring: blade t = {d.t} mm, L = {d.L} mm, width w = {w} mm, pivot {d.zp:.1f} mm above the paw centre")
    kb = E * w * d.t ** 3 / (12 * d.L)
    eb = d.limit * d.t / (2 * d.L)
    log(f"Beam formula (Euler-Bernoulli, pure moment): k = E w t^3 / (12 L) = {kb:.0f} N*mm/rad; "
        f"surface strain at the limit {d.limit} rad: limit * t / (2 L) = {eb:.3f}")
    n_t = 10 if args.quick else 14
    r_f = spring_cad_3d(d.t, d.L, n_t=n_t, bc="faces")
    r_e = spring_cad_3d(d.t, d.L, n_t=n_t, bc="ends")
    log("3D FE of the printed profile (fillets, bolt holes), blocks held on their far + front/back faces (tight socket) / "
        "far faces only (soft pocket):")
    log(f"  pitch {r_f['pitch']:.0f} / {r_e['pitch']:.0f} N*mm/rad  (beam x {r_f['pitch'] / kb:.2f} / {r_e['pitch'] / kb:.2f})")
    log(f"  roll  {r_f['roll']:.0f} / {r_e['roll']:.0f} N*mm/rad  ({r_f['roll'] / r_f['pitch']:.1f} x pitch)")
    log(f"  yaw   {r_f['yaw']:.0f} / {r_e['yaw']:.0f} N*mm/rad  ({r_f['yaw'] / r_f['pitch']:.1f} x pitch)")
    log(f"  axial {r_f['axial']:.0f} N/mm, shear along x {r_f['shear_x']:.0f} N/mm, along y {r_f['shear_y']:.0f} N/mm")
    log(f"  peak strain at the limit: {d.limit * r_f['strain_per_rad']:.3f} / {d.limit * r_e['strain_per_rad']:.3f}")
    log(f"Design value used by dog_paw.py: {d.k_est:.0f} N*mm/rad (calibration {DP.SPRING_K_FE}), "
        f"peak strain {d.eps_limit:.3f} at {d.limit} rad")
    if not args.quick:
        log("")
        log("Tuning at fixed L (k ~ t^3), dog_paw.spring_k:")
        log("| t [mm] | k [N*m/rad] | strain at limit |")
        log("|---|---|---|")
        for dt in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
            t = d.t + dt
            log(f"| {t:.2f} | {DP.spring_k(t, d.L) / 1000:.2f} | {DP.SPRING_EPS_FE * d.limit * t / (2 * d.L):.3f} |")
    log("")
    log(f"## Sole disc: {DP.SOLE_ARMS} arc arms ({math.degrees(DP.SOLE_ARM_SPAN):.0f} deg), arms {d.t_sole} mm thick, "
        f"travel to the bump stop {DP.SOLE_TRAVEL} mm")
    hs = 0.25 if args.quick else 0.2
    s_sl = sole_cad_3d(d.t_sole, h=hs, nz=5, slide=True)
    s_st = sole_cad_3d(d.t_sole, h=hs, nz=5, slide=False)
    log(f"3D FE (rim held, pad pushed up): k = {s_sl['k']:.1f} N/mm with the tread sliding, {s_st['k']:.1f} N/mm sticking; "
        f"peak strain {s_sl['strain_per_mm'] * DP.SOLE_TRAVEL:.3f} at the stop ({DP.SOLE_TRAVEL} mm), "
        f"{s_sl['strain_per_mm'] * 1.0:.3f} at 1 mm (~10 N)")
    log(f"Design value used by dog_paw.py: {d.k_sole_est:.1f} N/mm, peak strain {d.eps_sole:.3f} at the stop")
    if not args.quick:
        log("")
        log("Tuning (k ~ t^2.2: short curved arms bend and twist), dog_paw.sole_k:")
        log("| arms t [mm] | k [N/mm] | strain at the stop |")
        log("|---|---|---|")
        for t in (3.0, 3.5, 4.0, d.t_sole, 4.5, 5.0):
            log(f"| {t:.2f} | {DP.sole_k(t):.1f} | {DP.sole_strain(t):.3f} |")
        log("")
        log("## Rejected: a hollow round paw printed sole-down (axisymmetric FE, sphere R = 21 mm, 45 deg cavity roof)")
        log("| flat sole radius | wall 1.0 | 1.2 | 1.4 | 1.6 mm | [N/mm] |")
        log("|---|---|---|---|---|---|")
        for a in (6.4, 4.5, 3.0):
            sole = 2.0 - math.sqrt(21.0 ** 2 - a * a)
            ks = [paw_axi(t, sole=sole, apex=4.0, pocket_z0=6.5, top=18.0, h=0.12)["k"] for t in (1.0, 1.2, 1.4, 1.6)]
            log(f"| {a:.1f} mm | " + " | ".join(f"{k:.0f}" for k in ks) + " | |")
        log(f"Reissner (point load on a thin sphere): {reissner(1.0):.1f}, {reissner(1.2):.1f}, {reissner(1.4):.1f}, "
            f"{reissner(1.6):.1f} N/mm. A sole-down print needs a flat of r >= 6 mm (steeper overhangs fail in TPU), "
            "which makes a 1.0-1.6 mm shell 3-5x stiffer than 10 N/mm (thinner walls would not last).")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
