import importlib.util
from pathlib import Path

import numpy as np
import pytest

import scalerack

CLASSICAL_SMOKE_FACTOR = 1.7
SMOKE_FACTORS = {
    "content_adaptive_downscale": 0.5,
    "scale2x": 2.0,
    "scale3x": 3.0,
    "scale4x": 4.0,
}
PILLOW_INSTALLED = importlib.util.find_spec("PIL") is not None


class TestScalerSmoke:
    """Every algorithm completes one complex resize with the expected shape."""

    @pytest.mark.parametrize("name", sorted(scalerack.ALGORITHMS))
    def test_algorithm_scales_complex_rgba_image(
        self, name: str, complex_image: np.ndarray
    ) -> None:
        """Each algorithm returns a same-dtype ndarray of the expected size."""
        factor = SMOKE_FACTORS.get(name, CLASSICAL_SMOKE_FACTOR)
        result = scalerack.resize(name, complex_image, factor)
        assert isinstance(result, np.ndarray)
        assert result.dtype == complex_image.dtype
        expected_height = round(complex_image.shape[0] * factor)
        expected_width = round(complex_image.shape[1] * factor)
        expected_channels = 3 if name == "content_adaptive_downscale" else complex_image.shape[2]
        assert result.shape == (expected_height, expected_width, expected_channels)

    @pytest.mark.skipif(not PILLOW_INSTALLED, reason="Pillow not installed")
    def test_pil_image_round_trips(self, complex_image: np.ndarray) -> None:
        """A PIL image in yields a PIL image of the same mode out."""
        from PIL import Image

        source = Image.fromarray(complex_image)
        result = scalerack.lanczos(source, factor=2)
        assert isinstance(result, Image.Image)
        assert result.mode == source.mode
        assert result.size == (source.width * 2, source.height * 2)


@pytest.mark.skipif(not PILLOW_INSTALLED, reason="Pillow not installed")
class TestCliSmoke:
    """The CLI scales one file end to end."""

    def test_scale_command_writes_scaled_file(
        self, complex_image: np.ndarray, tmp_path: Path
    ) -> None:
        """One scale invocation reads a PNG and writes a 3x-scaled PNG."""
        from PIL import Image

        from scalerack.cli import main

        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        Image.fromarray(complex_image).save(input_path)
        exit_code = main(
            ["scale", str(input_path), str(output_path), "--method", "scale3x", "--factor", "3"]
        )
        assert exit_code == 0
        with Image.open(output_path) as result:
            assert result.size == (complex_image.shape[1] * 3, complex_image.shape[0] * 3)
