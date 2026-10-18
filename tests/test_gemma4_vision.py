"""Regression checks for Gemma4 ONNX export; run in the SDK environment."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
    import transformers.masking_utils as masks
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
except ImportError:
    torch = None

spec = importlib.util.spec_from_file_location('vision_launcher', ROOT / 'tools/run_vision.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@unittest.skipIf(torch is None, 'Requires the SDK torch/transformers environment')
class Gemma4TracingTests(unittest.TestCase):
    def test_rms_norm_keeps_large_fp16_residuals_finite(self):
        from types import SimpleNamespace
        torch.manual_seed(0)
        norm = SimpleNamespace(eps=1e-6)
        for magnitude in (1e-5, 1e-3, 1, 300, 1000):
            with self.subTest(magnitude=magnitude):
                values = (torch.randn(2, 32, 768) * magnitude).half()
                reference = values.float() * (values.float().square().mean(-1, keepdim=True) + norm.eps).pow(-.5)
                actual = launcher.stable_rms_norm(norm, values)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual.float(), reference, rtol=.005, atol=.002)
        zeros = launcher.stable_rms_norm(norm, torch.zeros(1, 768, dtype=torch.float16))
        self.assertTrue(torch.equal(zeros, torch.zeros_like(zeros)))

    def test_traced_mask_matches_eager_with_padding(self):
        config = Gemma4VisionConfig()
        config._attn_implementation = 'eager'

        class Mask(torch.nn.Module):
            def forward(self, embeds, valid):
                return masks.create_bidirectional_mask(config, embeds, valid)

        embeds = torch.zeros(2, 7, 8)
        valid = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
        expected = Mask()(embeds, valid)
        original = masks.sdpa_mask
        self.addCleanup(setattr, masks, 'sdpa_mask', original)
        launcher.patch_gemma4_export()
        actual = torch.jit.trace(Mask(), (embeds, valid))(embeds, valid)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
