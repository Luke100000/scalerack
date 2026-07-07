# Scalerack

[![PyPI version](https://img.shields.io/pypi/v/scalerack.svg)](https://pypi.org/project/scalerack/)

Many image up- and downscaling algorithms behind one unified interface: classical kernel resampling, edge-aware
reconstruction for crisp low-resolution art, vectorizing and perceptual downscaling approaches. Whatever goes in
comes back out in the same format:

- NumPy arrays: `(H, W)`, `(H, W, 3)`, `(H, W, 4)` in `uint8`, `float32`, `float64`
- Pillow images: `L`, `RGB`, `RGBA`

## Install

[PyPI package](https://pypi.org/project/scalerack/)

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

| Algorithm                                                      | Photo reconstruction                                                                                                          | Sprite upscale                                                                                                                                                     |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original                                                       | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/photo_upscale.jpg" width="160">          | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/sprite_upscale.png" width="48" style="image-rendering: pixelated;">           |
| `scale2x`<br>[Scale2x / EPX](https://www.scale2x.it/algorithm) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale2x_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale2x_upscale_sprite.png" width="96" style="image-rendering: pixelated;">  |
| `scale3x`<br>[Scale3x / EPX](https://www.scale2x.it/algorithm) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale3x_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale3x_upscale_sprite.png" width="144" style="image-rendering: pixelated;"> |
| `scale4x`<br>[Scale4x / EPX](https://www.scale2x.it/algorithm) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale4x_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/scale4x_upscale_sprite.png" width="192" style="image-rendering: pixelated;"> |

### Vectorizing scalers

Vectorizing scalers reconstruct smooth resolution-independent region outlines before
rasterizing at the target size, so they support arbitrary (non-integer) factors.

| Algorithm                                                                                | Photo reconstruction                                                                                                             | Sprite upscale                                                                                                                                                        |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original                                                                                 | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/photo_upscale.jpg" width="160">             | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/samples/sprite_upscale.png" width="48" style="image-rendering: pixelated;">              |
| `depixelize`<br>[Depixelizing Pixel Art](https://johanneskopf.de/publications/pixelart/) | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/depixelize_upscale_photo.png" width="160"> | <img src="https://raw.githubusercontent.com/Luke100000/scalerack/master/docs/previews/depixelize_upscale_sprite.png" width="192" style="image-rendering: pixelated;"> |

Sources: [Münster market](https://commons.wikimedia.org/wiki/File:M%C3%BCnster,_Wochenmarkt_--_2017_--_2333.jpg),
[macaw](https://commons.wikimedia.org/wiki/File%3AMacaw_parrot_%28Unsplash%29.jpg),
and [Pixelart TV](https://commons.wikimedia.org/wiki/File:Pixelart-tv-iso.png), via Wikimedia Commons.

## Roadmap

| Algorithm                                                                           | Family    | Status              |
|-------------------------------------------------------------------------------------|-----------|---------------------|
| nearest, box, bilinear, bicubic, mitchell, catmull_rom, lanczos, magic_kernel_sharp | classical | ✅ implemented       |
| scale2x / scale3x / scale4x (EPX)                                                   | pixel-art | ✅ implemented       |
| hq2x / hq3x / hq4x                                                                  | pixel-art | ⬜ to be implemented |
| xBRZ (2x-6x)                                                                        | pixel-art | ⬜ to be implemented |
| super-xBR                                                                           | pixel-art | ⬜ to be implemented |
| EASU / RCAS (FSR 1 core, CPU)                                                       | extended  | ⬜ to be implemented |
| EWA / Jinc (elliptical weighted average)                                            | extended  | ⬜ to be implemented |
| Gamma-correct (linear-light) resampling                                             | extended  | ⬜ to be implemented |
| Depixelizing Pixel Art (vectorizing)                                                | research  | ✅ implemented       |
| Content-adaptive downscaling                                                        | research  | ✅ implemented       |
| Perceptual / spectral downscaling                                                   | research  | ⬜ to be implemented |
| Eagle, 2xSaI, SuperEagle, SABR (legacy retro)                                       | contrib   | ⬜ to be implemented |

Machine-learning upscalers are out of scope.

## Development

```bash
uv sync --group dev
uv run pre-commit install                   # ruff check + format on commit
uv run pytest                               # smoke suite
uv run mypy src                             # type check
uv run python scripts/generate_previews.py  # regenerate the gallery above
```

## License

[MIT](LICENSE). All bundled algorithms are implemented from scratch from their specifications or papers.
