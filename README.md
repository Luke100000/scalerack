# Scalerack

Many image up- and downscaling algorithms behind one unified interface: classical kernel resampling, edge-aware
reconstruction for crisp low-resolution art, vectorizing and perceptual downscaling approaches. Whatever goes in
comes back out in the same format:

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

# generic dispatch + discovery
out = scalerack.resize("catmull_rom", image, 2)
print(*scalerack.ALGORITHMS)  # all algorithm names
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

| Algorithm            | Photo                                                              | Sprite                                                              |
|----------------------|--------------------------------------------------------------------|---------------------------------------------------------------------|
| `nearest`            | <img src="docs/previews/nearest_photo.png" width="320">            | <img src="docs/previews/nearest_sprite.png" width="200">            |
| `box`                | <img src="docs/previews/box_photo.png" width="320">                | <img src="docs/previews/box_sprite.png" width="200">                |
| `bilinear`           | <img src="docs/previews/bilinear_photo.png" width="320">           | <img src="docs/previews/bilinear_sprite.png" width="200">           |
| `bicubic`            | <img src="docs/previews/bicubic_photo.png" width="320">            | <img src="docs/previews/bicubic_sprite.png" width="200">            |
| `mitchell`           | <img src="docs/previews/mitchell_photo.png" width="320">           | <img src="docs/previews/mitchell_sprite.png" width="200">           |
| `catmull_rom`        | <img src="docs/previews/catmull_rom_photo.png" width="320">        | <img src="docs/previews/catmull_rom_sprite.png" width="200">        |
| `lanczos`            | <img src="docs/previews/lanczos_photo.png" width="320">            | <img src="docs/previews/lanczos_sprite.png" width="200">            |
| `magic_kernel_sharp` | <img src="docs/previews/magic_kernel_sharp_photo.png" width="320"> | <img src="docs/previews/magic_kernel_sharp_sprite.png" width="200"> |

### Pixel-art scalers

| Algorithm | Photo                                                   | Sprite                                                   |
|-----------|---------------------------------------------------------|----------------------------------------------------------|
| `scale2x` | <img src="docs/previews/scale2x_photo.png" width="320"> | <img src="docs/previews/scale2x_sprite.png" width="200"> |
| `scale3x` | <img src="docs/previews/scale3x_photo.png" width="320"> | <img src="docs/previews/scale3x_sprite.png" width="200"> |
| `scale4x` | <img src="docs/previews/scale4x_photo.png" width="320"> | <img src="docs/previews/scale4x_sprite.png" width="200"> |

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
| Depixelizing Pixel Art (vectorizing)                                                | research  | ⬜ to be implemented |
| Content-adaptive / perceptual / spectral downscaling                                | research  | ⬜ to be implemented |
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

[MIT](LICENSE). All bundled algorithm implementations are original (clean-room from published algorithm
descriptions).
