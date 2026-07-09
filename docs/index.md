---
title: Scalerack Python API
layout: default
---

# Scalerack

Python image resizing, resampling, downscaling, and pixel-art upscaling for NumPy arrays
and Pillow images.

## Install

~~~bash
pip install scalerack
pip install scalerack[cli]
pip install scalerack[all]
~~~

## Usage

~~~python
import scalerack

big = scalerack.mitchell(image, factor=2.5)
thumb = scalerack.box(photo, width=320)
sprite = scalerack.scale3x(pil_sprite)
out = scalerack.resize("lanczos", image, factor=2)
~~~

## Algorithms

### [`bicubic`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/bicubic.py#L7)

Scale with the classic Keys bicubic kernel.

Sharper than bilinear with mild overshoot; the general-purpose default when Lanczos rings too much.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.
- `a` (`float`, default `-0.5`): Keys sharpness coefficient (-0.5 standard; -0.75 sharper, more halo-prone).

---


### [`bilinear`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/bilinear.py#L7)

Scale with linear interpolation over the triangle kernel.

A fast, artifact-light baseline; visibly softer than the cubic and sinc families.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.

---


### [`box`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/box.py#L8)

Scale by averaging each output pixel's exact source footprint.

The best-behaved choice for downscaling (area interpolation); upscaling degenerates to nearest-neighbor blocks.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.

---


### [`catmull_rom`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/catmull_rom.py#L7)

Scale with the Catmull-Rom interpolating spline.

Sharper than Mitchell; a good pick when extra acutance is worth a little more ringing.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.

---


### [`content_adaptive_downscale`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/content_adaptive_downscale.py#L178)

Downscale with detail-preserving content-adaptive kernels (Kopf et al. 2013).

Each output pixel is a bilateral Gaussian kernel in joint space/color, fit by EM. Kernels start color-crisp; the locality and staircase constraints selectively smooth misbehaving kernels by raising their color variance. For the paper's pixel-art mode pass ``staircase_constraint=False``.
Sharpness accumulates with iterations; the ``max_iter`` default is calibrated to reproduce the sharpness of the paper's published results (running to full convergence over-sharpens). Pixel art converges early on its own.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.
- `staircase_constraint` (`bool`, default `True`):
- `locality_constraint` (`bool`, default `True`):
- `spatial_constraints` (`bool`, default `True`):
- `variance_constraints` (`bool`, default `True`):
- `max_iter` (`int`, default `30`):
- `color_sigma` (`float`, default `0.0001`):
- `tol` (`float`, default `0.0005`):
- `strict_pseudocode_init` (`bool`, default `False`):
- `locality_threshold` (`float | None`, default `None`):
- `edge_strength_threshold` (`float | None`, default `None`):
- `variance_bounds` (`tuple`, default `(0.05, 0.1)`):

---


### [`depixelize`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/depixelize/depixelize.py#L15)

Vectorize pixel art into smooth outlines (Kopf-Lischinski 2011) at any target size.

Intended for hard-edged, limited-palette pixel art at enlarging factors; factors at or below 1 are accepted but produce odd-looking results. The paper's optional harmonic-map cell relaxation is omitted.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.
- `iterations` (`int`, default `2`): Curve-relaxation sweeps; higher values smooth staircases more aggressively at the cost of drifting from the source shapes, 0 disables.
- `supersample` (`int`, default `2`): Antialiasing oversampling factor per output axis; higher values give softer edges and cost proportionally more time and memory.

---


### [`eagle2x`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/eagle.py#L14)

Enlarge pixel art exactly 2x with the classic Eagle corner-rounding rules.



---


### [`eagle3x`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/eagle.py#L20)

Enlarge pixel art exactly 3x with the community Eagle3x extension of the Eagle rules.



---


### [`lanczos`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/lanczos.py#L7)

Scale with a Lanczos windowed-sinc kernel.

The high-detail standard for photographic content; may ring at hard edges.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.
- `taps` (`int`, default `3`): Number of sinc lobes; more is sharper, slower, and more ring-prone.

---


### [`magic_kernel_sharp`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/magic_kernel_sharp.py#L7)

Scale with Costella's Magic Kernel Sharp.

A crisp modern alternative to plain cubic kernels.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.
- `version` (`int`, default `2021`): 2013 (slightly sharpening, cheaper) or 2021 (spectrally flatter).

---


### [`mitchell`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/mitchell.py#L7)

Scale with the Mitchell-Netravali BC-spline.

The safe offline default balancing blur, anisotropy, and ringing.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.
- `b` (`float`, default `0.3333333333333333`): BC-spline B parameter.
- `c` (`float`, default `0.3333333333333333`): BC-spline C parameter.

---


### [`nearest`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/nearest.py#L7)

Scale by copying the nearest source pixel.

Preserves exact pixel values; the right choice for masks, label maps, and deliberately blocky display of low-resolution art.


- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.

---


### [`sai2x`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/sai.py#L14)

Enlarge pixel art exactly 2x with Kreed's 2xSaI edge-aware interpolation.



---


### [`scale2x`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/epx.py#L14)

Enlarge pixel art exactly 2x with the Scale2x (EPX) neighborhood rules.



---


### [`scale3x`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/epx.py#L20)

Enlarge pixel art exactly 3x with the Scale3x neighborhood rules.



---


### [`scale4x`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/epx.py#L26)

Enlarge pixel art exactly 4x by applying Scale2x twice.



---


### [`super2xsai`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/sai.py#L20)

Enlarge pixel art exactly 2x with Kreed's Super 2xSaI, the family's strongest blur.



---


### [`supereagle`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/algorithms/sai.py#L26)

Enlarge pixel art exactly 2x with Kreed's SuperEagle edge detection and blending.



---


## API

### [`resize`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/__init__.py#L66)

Scale an image with the algorithm named by ``method``.

Translates the request into whatever sizing the algorithm exposes: a factor becomes width/height (and vice versa) where needed.


- `method` (`str`): Algorithm name (a key of ``ALGORITHMS``).
- `factor` (`float | None`, default `None`): Scale factor.
- `width` (`int | None`, default `None`): Target output width.
- `height` (`int | None`, default `None`): Target output height.
- `**opts` (`object`): Algorithm-specific options.

---


## Exceptions

- [`ScalerackError`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/exceptions.py#L1) - Base class for every error raised by scalerack.
- [`UnknownAlgorithmError`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/exceptions.py#L5) - Raised when an algorithm name is not a known scalerack algorithm.
- [`InvalidFactorError`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/exceptions.py#L17) - Raised when a sizing request violates the algorithm's constraints.
- [`UnsupportedImageError`](https://github.com/Luke100000/scalerack/blob/master/src/scalerack/exceptions.py#L13) - Raised when an input image has an unsupported type, shape, mode, or dtype.
