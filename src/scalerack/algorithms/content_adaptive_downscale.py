import math
from typing import Protocol

import numpy as np

from scalerack.algorithms.registry import register
from scalerack.image_io import ImageInput, as_image_input


class ColorModule(Protocol):
    def rgb2lab(self, rgb: np.ndarray) -> np.ndarray: ...

    def lab2rgb(self, lab: np.ndarray) -> np.ndarray: ...


def _skimage_color() -> ColorModule:
    try:
        from skimage import color
    except ImportError as exc:
        raise RuntimeError(
            "content_adaptive_downscale requires the optional dependency scikit-image"
        ) from exc
    return color


def _rgb_to_lab01(rgb: np.ndarray) -> np.ndarray:
    lab = _skimage_color().rgb2lab(np.clip(rgb, 0.0, 1.0))
    out = np.empty_like(lab, dtype=np.float64)
    out[..., 0] = lab[..., 0] / 100.0
    out[..., 1] = (lab[..., 1] + 128.0) / 255.0
    out[..., 2] = (lab[..., 2] + 128.0) / 255.0
    return out


def _lab01_to_rgb(lab01: np.ndarray) -> np.ndarray:
    lab = np.empty_like(lab01, dtype=np.float64)
    lab[..., 0] = lab01[..., 0] * 100.0
    lab[..., 1] = lab01[..., 1] * 255.0 - 128.0
    lab[..., 2] = lab01[..., 2] * 255.0 - 128.0
    return np.clip(_skimage_color().lab2rgb(lab), 0.0, 1.0)


def _resize_alpha(alpha: np.ndarray, wo: int, ho: int) -> np.ndarray:
    hi, wi = alpha.shape

    xs = (np.arange(wo) + 0.5) * wi / wo - 0.5
    ys = (np.arange(ho) + 0.5) * hi / ho - 0.5

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)

    x1 = np.clip(x0 + 1, 0, wi - 1)
    y1 = np.clip(y0 + 1, 0, hi - 1)

    x0 = np.clip(x0, 0, wi - 1)
    y0 = np.clip(y0, 0, hi - 1)

    wx = xs - x0
    wy = ys - y0

    top = (
        alpha[y0[:, None], x0[None, :]] * (1.0 - wx)[None, :]
        + alpha[y0[:, None], x1[None, :]] * wx[None, :]
    )
    bot = (
        alpha[y1[:, None], x0[None, :]] * (1.0 - wx)[None, :]
        + alpha[y1[:, None], x1[None, :]] * wx[None, :]
    )

    return top * (1.0 - wy)[:, None] + bot * wy[:, None]


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    c = abs(float(np.dot(a, b)) / (na * nb))
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def _neighbors4(k: int, wo: int, ho: int) -> list[int]:
    x = k % wo
    y = k // wo
    out: list[int] = []

    if x > 0:
        out.append(k - 1)
    if x + 1 < wo:
        out.append(k + 1)
    if y > 0:
        out.append(k - wo)
    if y + 1 < ho:
        out.append(k + wo)

    return out


def _neighbors8(k: int, wo: int, ho: int) -> list[int]:
    x = k % wo
    y = k // wo
    out: list[int] = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            xx = x + dx
            yy = y + dy

            if 0 <= xx < wo and 0 <= yy < ho:
                out.append(yy * wo + xx)

    return out


def _edge_orientation_vector(
    k: int,
    n: int,
    *,
    supports: list[np.ndarray],
    gamma: list[np.ndarray],
    wi: int,
    hi: int,
    rx: float,
    ry: float,
    eps: float,
) -> np.ndarray:
    idx_k = supports[k]
    idx_n = supports[n]
    union = np.union1d(idx_k, idx_n)

    if union.size == 0:
        return np.zeros(2, dtype=np.float64)

    uy = union // wi
    ux = union % wi

    y0 = max(0, int(uy.min()) - 1)
    y1 = min(hi - 1, int(uy.max()) + 1)
    x0 = max(0, int(ux.min()) - 1)
    x1 = min(wi - 1, int(ux.max()) + 1)

    gh = y1 - y0 + 1
    gw = x1 - x0 + 1

    gk = np.zeros((gh, gw), dtype=np.float64)
    gn = np.zeros((gh, gw), dtype=np.float64)

    yk = idx_k // wi
    xk = idx_k % wi
    m = (y0 <= yk) & (yk <= y1) & (x0 <= xk) & (xk <= x1)
    gk[yk[m] - y0, xk[m] - x0] = gamma[k][m]

    yn = idx_n // wi
    xn = idx_n % wi
    m = (y0 <= yn) & (yn <= y1) & (x0 <= xn) & (xn <= x1)
    gn[yn[m] - y0, xn[m] - x0] = gamma[n][m]

    den = gk + gn
    q = np.full_like(den, 0.5)
    np.divide(gk, den, out=q, where=den > eps)

    grad_y, grad_x = np.gradient(q)

    return np.array(
        [
            float(grad_x.sum()) * rx,
            float(grad_y.sum()) * ry,
        ],
        dtype=np.float64,
    )


@register
def content_adaptive_downscale(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    staircase_constraint: bool = True,
    locality_constraint: bool = True,
    spatial_constraints: bool = True,
    variance_constraints: bool = True,
    max_iter: int = 30,
    color_sigma: float = 0.04,
    tol: float = 5e-4,
    strict_pseudocode_init: bool = False,
    locality_threshold: float | None = None,
    edge_strength_threshold: float | None = None,
) -> ImageInput:
    image_input = as_image_input(image)
    rgba = image_input.rgba().astype(np.float64)
    rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
    alpha = np.clip(rgba[:, :, 3], 0.0, 1.0)

    hi, wi = rgb.shape[:2]
    wo, ho = image_input.get_target_dimensions(width, height, factor)
    if wo > wi or ho > hi:
        raise ValueError("This implementation only downscales.")

    if (wo, ho) == (wi, hi):
        return image_input.from_numpy(image_input.numpy().copy())

    if color_sigma <= 0:
        raise ValueError("color_sigma must be positive.")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    eps = 1e-300
    rx = wi / wo
    ry = hi / ho

    if locality_threshold is None:
        locality_threshold = 0.2 * rx

    if edge_strength_threshold is None:
        edge_strength_threshold = 0.08 * rx * ry

    lab = _rgb_to_lab01(rgb)
    colors = lab.reshape(-1, 3)

    yy, xx = np.indices((hi, wi), dtype=np.float64)
    pos = np.stack((xx.ravel(), yy.ravel()), axis=1)

    centers = np.array(
        [((ox + 0.5) * rx, (oy + 0.5) * ry) for oy in range(ho) for ox in range(wo)],
        dtype=np.float64,
    )

    centers_out = np.array(
        [(ox + 0.5, oy + 0.5) for oy in range(ho) for ox in range(wo)],
        dtype=np.float64,
    )

    k_count = wo * ho

    supports: list[np.ndarray] = []

    for cx, cy in centers:
        x0 = max(0, int(np.floor(cx - 2.0 * rx)))
        x1 = min(wi - 1, int(np.ceil(cx + 2.0 * rx)))
        y0 = max(0, int(np.floor(cy - 2.0 * ry)))
        y1 = min(hi - 1, int(np.ceil(cy + 2.0 * ry)))

        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)

        X, Y = np.meshgrid(xs, ys)
        idx = (Y * wi + X).ravel().astype(np.int32)

        p = pos[idx]
        keep = (np.abs(p[:, 0] - cx) < 2.0 * rx) & (np.abs(p[:, 1] - cy) < 2.0 * ry)

        supports.append(idx[keep])

    support_pos = [pos[s] for s in supports]
    support_col = [colors[s] for s in supports]

    mu = centers.copy()

    cov = np.zeros((k_count, 2, 2), dtype=np.float64)
    cov[:, 0, 0] = rx / 3.0
    cov[:, 1, 1] = ry / 3.0

    sigma = np.full(k_count, color_sigma, dtype=np.float64)

    if strict_pseudocode_init:
        nu = np.full((k_count, 3), 0.5, dtype=np.float64)
    else:
        nu = np.empty((k_count, 3), dtype=np.float64)

        for k in range(k_count):
            d = support_pos[k] - mu[k]
            inv = np.linalg.inv(cov[k] + np.eye(2) * 1e-8)
            logw = -0.5 * np.einsum("ij,jk,ik->i", d, inv, d)
            w = np.exp(logw - logw.max())
            w /= max(float(w.sum()), eps)
            nu[k] = w @ support_col[k]

    nbr4 = [_neighbors4(k, wo, ho) for k in range(k_count)]
    nbr8 = [_neighbors8(k, wo, ho) for k in range(k_count)]

    for _ in range(max_iter):
        weights: list[np.ndarray] = []

        for k in range(k_count):
            p = support_pos[k]
            c = support_col[k]

            d = p - mu[k]
            inv = np.linalg.inv(cov[k] + np.eye(2) * 1e-8)

            spatial = -0.5 * np.einsum("ij,jk,ik->i", d, inv, d)

            dc = c - nu[k]
            chroma = -np.sum(dc * dc, axis=1) / (2.0 * sigma[k] * sigma[k])

            logw = spatial + chroma
            m = float(logw.max())

            w = np.exp(logw - m) if np.isfinite(m) else np.ones_like(logw)
            w /= max(float(w.sum()), eps)

            weights.append(w)

        denom = np.zeros(hi * wi, dtype=np.float64)

        for s, w in zip(supports, weights, strict=True):
            np.add.at(denom, s, w)

        gamma = [w / np.maximum(denom[s], eps) for s, w in zip(supports, weights, strict=True)]

        old_mu = mu.copy()
        old_cov = cov.copy()
        old_nu = nu.copy()
        old_sigma = sigma.copy()

        # M-step.
        for k in range(k_count):
            g = gamma[k]
            gsum = float(g.sum())

            if gsum <= 1e-12:
                continue

            p = support_pos[k]
            c = support_col[k]

            d_old = p - mu[k]
            cov[k] = (d_old * g[:, None]).T @ d_old / gsum + np.eye(2) * 1e-8

            mu[k] = (g[:, None] * p).sum(axis=0) / gsum
            nu[k] = (g[:, None] * c).sum(axis=0) / gsum

        # C-step 1: spatial mean smoothing and clampBox.
        if spatial_constraints:
            mu_bar = mu.copy()

            for k in range(k_count):
                if nbr4[k]:
                    mu_bar[k] = mu[nbr4[k]].mean(axis=0)

            mu = 0.5 * mu + 0.5 * mu_bar

            mu[:, 0] = np.clip(
                mu[:, 0],
                centers[:, 0] - rx / 4.0,
                centers[:, 0] + rx / 4.0,
            )
            mu[:, 1] = np.clip(
                mu[:, 1],
                centers[:, 1] - ry / 4.0,
                centers[:, 1] + ry / 4.0,
            )

        # C-step 2: spatial covariance/eigenvalue clamp.
        if variance_constraints:
            for k in range(k_count):
                vals, vecs = np.linalg.eigh(0.5 * (cov[k] + cov[k].T))

                # Literal pseudocode values. They are in the paper's normalized
                # working coordinate scale, but retained here for paper fidelity.
                vals = np.clip(vals, 0.05, 0.10)

                cov[k] = (vecs * vals) @ vecs.T + np.eye(2) * 1e-8
                cov[k] = 0.5 * (cov[k] + cov[k].T)

        # C-step 3: locality and staircase constraints.
        # Both constraints increase sigma, making color kernels less restrictive
        # on the next E-step.
        if locality_constraint or staircase_constraint:
            sigma_update = np.ones(k_count, dtype=np.float64)

            mu_out = np.column_stack(
                (
                    mu[:, 0] / rx,
                    mu[:, 1] / ry,
                )
            )

            for k in range(k_count):
                pk_out = np.column_stack(
                    (
                        support_pos[k][:, 0] / rx,
                        support_pos[k][:, 1] / ry,
                    )
                )

                gk = gamma[k]
                idx_k = supports[k]

                for n in nbr8[k]:
                    direction = centers_out[n] - centers_out[k]
                    trigger = False

                    if locality_constraint:
                        proj = (pk_out - mu_out[k]) @ direction
                        locality_score = float(np.sum(gk * np.maximum(0.0, proj) ** 2))

                        if locality_score > locality_threshold:
                            trigger = True

                    if staircase_constraint and not trigger:
                        common, ia, ib = np.intersect1d(
                            idx_k,
                            supports[n],
                            return_indices=True,
                            assume_unique=True,
                        )

                        f = float(np.dot(gamma[k][ia], gamma[n][ib])) if common.size else 0.0

                        if f < edge_strength_threshold:
                            o = _edge_orientation_vector(
                                k,
                                n,
                                supports=supports,
                                gamma=gamma,
                                wi=wi,
                                hi=hi,
                                rx=rx,
                                ry=ry,
                                eps=eps,
                            )

                            if _angle_degrees(direction, o) > 25.0:
                                trigger = True

                    if trigger:
                        sigma_update[k] = max(sigma_update[k], 1.1)
                        sigma_update[n] = max(sigma_update[n], 1.1)

            sigma *= sigma_update

        delta = max(
            float(np.max(np.abs(mu - old_mu))),
            float(np.max(np.abs(cov - old_cov))),
            float(np.max(np.abs(nu - old_nu))),
            float(np.max(np.abs(sigma - old_sigma))),
        )

        if delta < tol:
            break

    out_rgb = _lab01_to_rgb(nu.reshape(ho, wo, 3))
    out_alpha = _resize_alpha(alpha, wo, ho)
    return image_input.from_numpy(np.dstack([out_rgb, out_alpha]))
