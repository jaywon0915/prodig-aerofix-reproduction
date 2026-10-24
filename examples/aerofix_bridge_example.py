"""Example bridge for plugging an existing AeroFix model into prodig.train.

Run with:
    python -m prodig.train ... \
      --aerofix-factory examples.aerofix_bridge_example:create_aerofix

Replace `YourAeroFixModel` and the call inside `fix()` with your real inference API.
"""

from prodig.aerofix import AeroFixInput, AeroFixResult


class AeroFixBridge:
    def __init__(self):
        # Example only:
        # self.model = YourAeroFixModel.from_pretrained(...).cuda().eval()
        self.model = None

    def fix(self, item: AeroFixInput) -> AeroFixResult:
        # Inputs are CPU float tensors in [0,1], CHW:
        #   item.noisy_rgb
        #   item.reference_rgb
        # Camera objects contain c2w and K:
        #   item.novel_camera.c2w / .K
        #   item.reference_camera.c2w / .K
        # Optional depth, if --reference-depth-dir was supplied:
        #   item.reference_depth
        #
        # Replace the following line with real AeroFix inference.
        fixed = item.noisy_rgb

        # If your AeroFix model has a confidence/consistency score, return it here.
        # ProDiGTrainer discards views below --min-aerofix-quality and multiplies
        # the synthetic-view RGB loss by `quality * synthetic_loss_weight`.
        return AeroFixResult(fixed_rgb=fixed, quality=1.0)


def create_aerofix():
    return AeroFixBridge()
