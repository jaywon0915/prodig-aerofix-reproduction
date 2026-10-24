import sys
from pathlib import Path
import torch

from prodig.aerofix import AeroFixInput, IdentityAeroFix, load_aerofix_factory


def test_identity_aerofix_returns_noisy_image_with_unit_quality():
    noisy = torch.rand(3, 8, 8)
    ref = torch.rand(3, 8, 8)
    out = IdentityAeroFix().fix(AeroFixInput(noisy_rgb=noisy, reference_rgb=ref))
    assert torch.allclose(out.fixed_rgb, noisy)
    assert out.quality == 1.0


def test_factory_loader(tmp_path: Path):
    module = tmp_path / "custom_af.py"
    module.write_text("from prodig.aerofix import IdentityAeroFix\ndef create():\n    return IdentityAeroFix()\n")
    sys.path.insert(0, str(tmp_path))
    try:
        af = load_aerofix_factory("custom_af:create")
        assert isinstance(af, IdentityAeroFix)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("custom_af", None)


def test_refined_view_with_zero_quality_is_rejected_even_at_default_threshold():
    from prodig.trainer import accept_refined_view

    assert not accept_refined_view(0.0, 0.0)  # refiner's explicit rejection
    assert accept_refined_view(1.0, 0.0)
    assert not accept_refined_view(0.005, 0.01)
    assert accept_refined_view(0.01, 0.01)
