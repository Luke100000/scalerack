# Scalerack

[![PyPI](https://img.shields.io/pypi/v/scalerack)](https://pypi.org/project/scalerack/)
[![Python](https://img.shields.io/pypi/pyversions/scalerack)](https://www.python.org/)
[![License](https://img.shields.io/pypi/l/scalerack)](LICENSE)
[![Tests](https://github.com/Luke100000/scalerack/actions/workflows/ci.yml/badge.svg)](https://github.com/Luke100000/scalerack/actions/workflows/ci.yml)
[![Typing](https://img.shields.io/badge/typing-typed-blue)](https://typing.python.org/en/latest/spec/distributing.html)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-222222?logo=github)](https://luke100000.github.io/scalerack/)

Scalerack is a Python image resizing, resampling, upscaling, downscaling, and pixel-art scaling library with a CLI. It
offers many image scaling algorithms through one unified interface, supports NumPy arrays and Pillow images, and
preserves the input format on output. Included methods range from classical filters such as nearest, box, bilinear,
bicubic, Mitchell, Catmull-Rom, Lanczos, Magic Kernel Sharp, and EWA/Jinc to gamma-correct resampling, edge-aware
pixel-art scalers such as Scale2x/3x/4x, hq2x/hq3x/hq4x, xBRZ, super-xBR, EASU/RCAS, depixelization, vectorizing, Seam
Carving, and perceptual downscaling with L0 Gradient Minimization.

- NumPy arrays: `(H, W)`, `(H, W, 3)`, `(H, W, 4)` in `uint8`, `float32`, `float64`
- Pillow images: `L`, `RGB`, `RGBA`

## Install

```bash
pip install scalerack          # core resampler
pip install scalerack[cli]     # command line interface
pip install scalerack[all]     # everything
```

## Usage

```python
import scalerack

big = scalerack.mitchell(image, factor=2.5)
thumb = scalerack.box(photo, width=320)  # height inferred
crisp = scalerack.magic_kernel_sharp(image, width=100, height=100, version=2021)
sprite = scalerack.scale3x(pil_sprite)  # fixed 3x

# generic dispatch
out = scalerack.resize("catmull_rom", image, 2)

# all algorithm names
print(*scalerack.ALGORITHMS)
```

Invalid parameters (unsupported factor, wrong input type, missing extra) raise exceptions.

CLI (with the `cli` extra):

```bash
scalerack scale input.png output.png --method lanczos --factor 2
scalerack scale photo.png thumb.png --method box --width 320   # height inferred from aspect ratio
scalerack scale sprite.png big.png --method scale4x --factor 4
scalerack list
```

## Algorithms

Check the code documentation for algorithm details.
Named methods link to their original work where available.

### Classical resamplers

Downscale previews are generated as 4x reductions (0.25x output size). Photo reconstruction previews
first degrade the source with Lanczos at 0.25x, then upscale the degraded image back toward the
original size. Sprite downscale previews reduce the pixel-art source directly, and sprite upscale
previews enlarge the pixel-art source directly.

| Algorithm                                                                                                   | Photo downscale                                                                                                                            | Sprite downscale                                                                                                                                                                | Photo reconstruction                                                                                                                     | Sprite upscale                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original                                                                                                    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/photo_downscale.jpg" width="160">                     | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/sprite_downscale.png" width="192" style="image-rendering: pixelated;">                     | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/photo_upscale.jpg" width="160">                     | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/sprite_upscale.png" width="48" style="image-rendering: pixelated;">                      |
| `nearest`                                                                                                   | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/nearest_downscale_photo.png" width="160">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/nearest_downscale_sprite.png" width="192" style="image-rendering: pixelated;">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/nearest_upscale_photo.png" width="160">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/nearest_upscale_sprite.png" width="192" style="image-rendering: pixelated;">            |
| `box`                                                                                                       | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/box_downscale_photo.png" width="160">                | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/box_downscale_sprite.png" width="192" style="image-rendering: pixelated;">                | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/box_upscale_photo.png" width="160">                | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/box_upscale_sprite.png" width="192" style="image-rendering: pixelated;">                |
| `bilinear`                                                                                                  | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bilinear_downscale_photo.png" width="160">           | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bilinear_downscale_sprite.png" width="192" style="image-rendering: pixelated;">           | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bilinear_upscale_photo.png" width="160">           | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bilinear_upscale_sprite.png" width="192" style="image-rendering: pixelated;">           |
| `bicubic`<br>[Cubic convolution interpolation](https://doi.org/10.1109/TASSP.1981.1163711)                  | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bicubic_downscale_photo.png" width="160">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bicubic_downscale_sprite.png" width="192" style="image-rendering: pixelated;">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bicubic_upscale_photo.png" width="160">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/bicubic_upscale_sprite.png" width="192" style="image-rendering: pixelated;">            |
| `mitchell`<br>[Mitchell-Netravali filter](https://dl.acm.org/doi/10.1145/378456.378514)                     | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/mitchell_downscale_photo.png" width="160">           | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/mitchell_downscale_sprite.png" width="192" style="image-rendering: pixelated;">           | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/mitchell_upscale_photo.png" width="160">           | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/mitchell_upscale_sprite.png" width="192" style="image-rendering: pixelated;">           |
| `catmull_rom`<br>[Catmull-Rom spline](https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/catmull_rom_downscale_photo.png" width="160">        | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/catmull_rom_downscale_sprite.png" width="192" style="image-rendering: pixelated;">        | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/catmull_rom_upscale_photo.png" width="160">        | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/catmull_rom_upscale_sprite.png" width="192" style="image-rendering: pixelated;">        |
| `lanczos`<br>[Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling)                         | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/lanczos_downscale_photo.png" width="160">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/lanczos_downscale_sprite.png" width="192" style="image-rendering: pixelated;">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/lanczos_upscale_photo.png" width="160">            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/lanczos_upscale_sprite.png" width="192" style="image-rendering: pixelated;">            |
| `magic_kernel_sharp`<br>[Magic Kernel Sharp](https://johncostella.com/magic/)                               | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/magic_kernel_sharp_downscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/magic_kernel_sharp_downscale_sprite.png" width="192" style="image-rendering: pixelated;"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/magic_kernel_sharp_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/magic_kernel_sharp_upscale_sprite.png" width="192" style="image-rendering: pixelated;"> |

### Downscalers

Downscaling algorithm usually focus on preserving detail.

| Algorithm                                                                                                                                                           | Photo downscale                                                                                                                                    | Sprite downscale                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `content_adaptive_downscale`<br>[Content-Adaptive Image Downscaling](https://johanneskopf.de/publications/downscaling/)<br>(Does not fully reproduce paper results) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/content_adaptive_downscale_downscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/content_adaptive_downscale_downscale_sprite.png" width="192" style="image-rendering: pixelated;"> |

### Pixel-art scalers

Some algorithms may output different results depending on the fixed scale factor.

| Algorithm                                                                                                                                  | Photo reconstruction                                                                                                             | Sprite upscale                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original                                                                                                                                   | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/photo_upscale.jpg" width="160">             | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/sprite_upscale.png" width="48" style="image-rendering: pixelated;">             |
| `scale2x`<br>[Scale2x / EPX](https://www.scale2x.it/algorithm)                                                                             | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale2x_upscale_photo.png" width="160">    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale2x_upscale_sprite.png" width="96" style="image-rendering: pixelated;">    |
| `scale3x`<br>[Scale3x / EPX](https://www.scale2x.it/algorithm)                                                                             | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale3x_upscale_photo.png" width="160">    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale3x_upscale_sprite.png" width="144" style="image-rendering: pixelated;">   |
| `scale4x`<br>[Scale4x / EPX](https://www.scale2x.it/algorithm)                                                                             | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale4x_upscale_photo.png" width="160">    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale4x_upscale_sprite.png" width="192" style="image-rendering: pixelated;">   |
| `eagle2x`<br>[Eagle](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#Eagle)                                                     | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/eagle2x_upscale_photo.png" width="160">    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/eagle2x_upscale_sprite.png" width="96" style="image-rendering: pixelated;">    |
| `eagle3x`<br>[Eagle 3x](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#Eagle)<br>(community 3x extension of Eagle)             | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/eagle3x_upscale_photo.png" width="160">    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/eagle3x_upscale_sprite.png" width="144" style="image-rendering: pixelated;">   |
| `sai2x`<br>[2xSaI](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#2%C3%97SaI)<br>(Derek "Kreed" Liauw Kie Fa, 1999)            | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/sai2x_upscale_photo.png" width="160">      | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/sai2x_upscale_sprite.png" width="96" style="image-rendering: pixelated;">      |
| `supereagle`<br>[SuperEagle](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#2%C3%97SaI)<br>(Derek "Kreed" Liauw Kie Fa, 1999)  | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/supereagle_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/supereagle_upscale_sprite.png" width="96" style="image-rendering: pixelated;"> |
| `super2xsai`<br>[Super 2xSaI](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#2%C3%97SaI)<br>(Derek "Kreed" Liauw Kie Fa, 1999) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/super2xsai_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/super2xsai_upscale_sprite.png" width="96" style="image-rendering: pixelated;"> |
| `hq2x`<br>[hq2x](https://en.wikipedia.org/wiki/Hqx_(algorithm))<br>(Maxim Stepin, 2003)                                                    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/hq2x_upscale_photo.png" width="160">       | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/hq2x_upscale_sprite.png" width="96" style="image-rendering: pixelated;">       |
| `hq3x`<br>[hq3x](https://en.wikipedia.org/wiki/Hqx_(algorithm))<br>(Maxim Stepin, 2003)                                                    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/hq3x_upscale_photo.png" width="160">       | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/hq3x_upscale_sprite.png" width="144" style="image-rendering: pixelated;">      |
| `hq4x`<br>[hq4x](https://en.wikipedia.org/wiki/Hqx_(algorithm))<br>(Maxim Stepin, 2003)                                                    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/hq4x_upscale_photo.png" width="160">       | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/hq4x_upscale_sprite.png" width="192" style="image-rendering: pixelated;">      |
| `xbrz` (2x-6x)<br>[xBRZ](https://sourceforge.net/projects/xbrz/)<br>(Zenju)                                                                | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/xbrz_upscale_photo.png" width="160">       | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/xbrz_upscale_sprite.png" width="192" style="image-rendering: pixelated;">      |
| `superxbr`<br>[Super xBR](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#xBR_family)<br>(Hyllian, 2015; MIT reference port)    | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/superxbr_upscale_photo.png" width="160">   | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/superxbr_upscale_sprite.png" width="96" style="image-rendering: pixelated;">   |

### Vectorizing scalers

Vectorizing scalers reconstruct smooth resolution-independent region outlines before
rasterizing at the target size, so they support arbitrary (non-integer) factors.

| Algorithm                                                                                | Photo reconstruction                                                                                                             | Sprite upscale                                                                                                                                                        |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original                                                                                 | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/photo_upscale.jpg" width="160">             | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/sprite_upscale.png" width="48" style="image-rendering: pixelated;">              |
| `depixelize`<br>[Depixelizing Pixel Art](https://johanneskopf.de/publications/pixelart/) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/depixelize_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/depixelize_upscale_sprite.png" width="192" style="image-rendering: pixelated;"> |

Gallery sources: Dietmar
Rabich's [Münster market](https://commons.wikimedia.org/wiki/File:M%C3%BCnster,_Wochenmarkt_--_2017_--_2333.jpg)
([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)), Andrew Pons's
[macaw](https://commons.wikimedia.org/wiki/File%3AMacaw_parrot_%28Unsplash%29.jpg)
([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)), and LucasVB's
[Pixelart TV](https://commons.wikimedia.org/wiki/File:Pixelart-tv-iso.png)
([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)), and the
[pixel-art scaling comparison](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#/media/File:Test_nn.png),
via Wikimedia Commons.

## Roadmap

| Algorithm                                | Family   |
|------------------------------------------|----------|
| EASU / RCAS (FSR 1 core, CPU)            | extended |
| EWA / Jinc (elliptical weighted average) | extended |
| Gamma-correct (linear-light) resampling  | extended |
| Perceptual / spectral downscaling        | research |
| SABR (legacy retro)                      | contrib  |

Machine-learning upscalers are out of scope.

## Development

```bash
uv sync --group dev
uv run pre-commit install                   # ruff check + format on commit
uv run pytest                               # smoke suite
uv run mypy src                             # type check
uv run python scripts/generate_previews.py  # regenerate the gallery above
uv run python scripts/generate_docs.py      # regenerate docs/index.md

cd docs
bundle config set path vendor/bundle && bundle install
bundle exec jekyll serve --source . --destination _site # --host 127.0.0.1 --port 4000
```

## License

Scalerack's current source tree is licensed under the [Mozilla Public License 2.0](LICENSE).

The implementations are best-effort expressions based on the cited public descriptions,
specifications, and papers, with credit to original authors and work wherever possible.
Common technical elements such as kernels, coefficients, thresholds, and lookup tables may
also appear in other implementations, including implementations available under different
licenses; MPL-2.0 applies to the contributions and rights Scalerack's contributors can grant.
