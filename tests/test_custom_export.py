"""Run with python -m unittest discover -s tests; no ML toolkits required."""
from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import custom_export as converter


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'data.json'
        self.dataset.write_text('[{"input":"question", "target":"answer"}]')

    def args(self, *extra):
        return converter.build_parser().parse_args(['-m', 'org/model', *extra])

    def main(self, *extra):
        with patch.object(sys, 'argv', ['custom_export.py', '-m', 'org/model',
                                        '--output-dir', str(self.root / 'output'), *extra]):
            return converter.main()

    def test_sdk_matrix(self):
        matrix = converter.conversion_matrix('ALL', 'all')
        self.assertEqual(len(matrix), 19)
        self.assertEqual(len(set(matrix)), 19)
        for platform, dtype in matrix:
            self.assertIn(dtype, converter.PLATFORM_DTYPES[platform])
        self.assertEqual(converter.conversion_matrix('CUDA', None), [('CUDA', 'w4a16_g32')])
        self.assertEqual(len(converter.conversion_matrix('ALL', None)), 7)
        self.assertEqual(len(converter.conversion_matrix('ALL', 'fp')), 4)
        with self.assertRaises(SystemExit):
            converter.conversion_matrix('RK3588', 'w4a16')
        with self.assertRaises(SystemExit):
            converter.conversion_matrix('RV1126B', 'w8a8_g128')

    def test_validation_before_loading(self):
        for options in (['--num-npu-core', '3', '-p', 'RK3576'],
                        ['-q', 'w4a16_g32', '-p', 'RK3576', '--device', 'cpu'],
                        ['-q', 'w8a8', '--quantized-algorithm', 'grq'],
                        ['--kind', 'vlm', '--vision-only', '--skip-vision'],
                        ['--kind', 'vlm', '--model-name', 'gemma-4', '--batch-size', '2'],
                        ['--audio-only']):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                args = self.args(*options)
                converter.validate_options(args, Path(args.model), converter.conversion_matrix(args.platform, args.dtype))
        for rank in (1, 8, 64):
            self.assertEqual(converter.quantization_algorithm(f'grq_r{rank}'), f'grq_r{rank}')
        for algorithm in ('grq_r0', 'normal_r65', 'grq_r8junk'):
            with self.assertRaises(Exception):
                converter.quantization_algorithm(algorithm)
        for value in ('1', '-0.1', 'nan'):
            with self.assertRaises(Exception):
                converter.unit_interval(value)

    def test_full_matrix_api_and_resume(self):
        calls = []
        class FakeRKLLM:
            def load_huggingface(self, **kwargs):
                calls.append(('load', kwargs)); return 0
            def build(self, **kwargs):
                calls.append(('build', kwargs)); return 0
            def export_rkllm(self, path, **kwargs):
                calls.append(('export', kwargs)); Path(path).write_text('artifact'); return 0
        modules = {'rkllm': types.ModuleType('rkllm'), 'rkllm.api': types.SimpleNamespace(RKLLM=FakeRKLLM)}
        with patch.dict(sys.modules, modules):
            self.main('-p', 'ALL', '-q', 'all', '--dataset', str(self.dataset))
            builds = [args for method, args in calls if method == 'build']
            self.assertEqual(len(builds), 19)
            self.assertEqual({(x['target_platform'], x['quantized_dtype']) for x in builds},
                             set(converter.conversion_matrix('ALL', 'all')))
            self.assertEqual({x['num_npu_core'] for x in builds}, {1, 2, 3})
            self.main('-p', 'ALL', '-q', 'all', '--dataset', str(self.dataset))
            self.assertEqual(len(calls), 19 * 3)

    def test_wrapper_keywords_match_bundled_sdk(self):
        wheel = next((TOOLS.parent / 'rkllm-toolkit/packages').glob('*1.3.1-cp311*.whl'))
        with zipfile.ZipFile(wheel) as archive:
            sdk = ast.parse(archive.read('rkllm/api/rkllm.py'))
        methods = {node.name: {arg.arg for arg in node.args.args}
                   for node in ast.walk(sdk) if isinstance(node, ast.FunctionDef)}
        wrapper = ast.parse((TOOLS / 'custom_export.py').read_text())
        checked = set()
        for node in ast.walk(wrapper):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'llm':
                    self.assertLessEqual({keyword.arg for keyword in node.keywords}, methods[node.func.attr])
                    checked.add(node.func.attr)
        self.assertEqual(checked, {'load_huggingface', 'load_gguf', 'build', 'export_rkllm'})

    def test_language_build_failure_stops_export(self):
        calls = []
        class FakeRKLLM:
            def load_huggingface(self, **kwargs): return 0
            def build(self, **kwargs): return -1
            def export_rkllm(self, *a, **k): calls.append('export'); return 0
        with patch.dict(sys.modules, {'rkllm.api': types.SimpleNamespace(RKLLM=FakeRKLLM)}):
            with self.assertRaisesRegex(RuntimeError, 'build failed'):
                converter.export_rkllm(self.args('-q', 'w8a8'), Path('org/model'), self.dataset,
                                       self.root / 'failure.rkllm')
        self.assertEqual(calls, [])

    def test_gguf_loader_fp_and_export_options(self):
        model = self.root / 'model.gguf'
        model.touch()
        calls = []
        class FakeRKLLM:
            def load_gguf(self, **kwargs):
                calls.append(('gguf', kwargs)); return 0
            def build(self, **kwargs):
                calls.append(('build', kwargs)); return 0
            def export_rkllm(self, path, **kwargs):
                calls.append(('export', kwargs)); return 0
        args = self.args('--format', 'gguf', '-q', 'fp', '--no-export-tokenizer', '--no-export-embedding')
        with patch.dict(sys.modules, {'rkllm.api': types.SimpleNamespace(RKLLM=FakeRKLLM)}):
            converter.export_rkllm(args, model, None, self.root / 'fp.rkllm')
        self.assertEqual(calls[0], ('gguf', {'model': str(model)}))
        self.assertFalse(calls[1][1]['do_quantization'])
        self.assertIsNone(calls[1][1]['dataset'])
        self.assertEqual(calls[2][1], {'export_tokenizer': False, 'export_embedding': False})

    def test_user_manifest_never_regenerated(self):
        missing = self.root / 'missing.json'
        with patch.object(converter, 'run') as run, self.assertRaises(SystemExit):
            self.main('--dataset', str(missing))
        run.assert_not_called()
        self.dataset.write_text('[]')
        with self.assertRaises(SystemExit):
            converter.validate_calibration_dataset(self.dataset)

    def test_relative_multimodal_manifest_paths(self):
        samples = self.root / 'samples'
        samples.mkdir()
        (samples / 'entry').write_bytes(b'calibration')
        self.dataset.write_text(json.dumps([{'sample': 'samples/entry', 'token_nums': 32}]))
        destination = self.root / 'output' / 'resolved.json'
        converter.validate_calibration_dataset(self.dataset)
        converter.make_vlm_dataset_paths_absolute(self.dataset, destination)
        self.assertEqual(json.loads(destination.read_text())[0]['sample'], str((samples / 'entry').resolve()))
        (samples / 'entry').unlink()
        with self.assertRaises(SystemExit):
            converter.validate_calibration_dataset(self.dataset)

    def test_qparams_cache(self):
        cache = self.root / 'grq.qparams'
        cache.write_bytes(b'binary cache')
        self.assertEqual(converter.qparams_value(str(cache)), str(cache.resolve()))
        self.assertEqual(converter.qparams_value('{"key": 1}'), {'key': 1})

    def test_model_detection(self):
        cases = {'google/gemma-4-E2B-it': 'gemma-4', 'Qwen/Qwen2-VL-2B-Instruct': 'qwen2-vl',
                 'Qwen/Qwen2.5-VL-3B-Instruct': 'qwen2_5-vl-3b', 'Qwen/Qwen3.5-2B': 'qwen3.5'}
        for model, expected in cases.items():
            self.assertEqual(converter.infer_vlm_model_name(Path(model), None), expected)
        with self.assertRaises(SystemExit):
            converter.infer_vlm_model_name(Path('LiquidAI/LFM2.5-VL-450M'), None)

    def test_gemma_vision_audio_platform_fanout(self):
        commands = []
        def run(command, cwd):
            commands.append(command)
            for flag in ('--export_vision_path', '--export_audio_path', '--output'):
                if flag in command:
                    target = Path(command[command.index(flag) + 1])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
        with patch.object(converter, 'run', side_effect=run):
            self.main('-m', 'google/gemma-4-E2B-it', '--kind', 'vlm', '-p', 'ALL',
                      '--vision-only', '--audio')
        self.assertEqual(sum('export_vision_gemma4.py' in str(c) for c in commands), 1)
        self.assertEqual(sum('export_audio_gemma4.py' in str(c) for c in commands), 1)
        self.assertEqual(sum('--target-platform' in c for c in commands), 8)
        onnx = self.root / 'output' / 'gemma-4-E2B-it_vision.onnx'
        self.assertTrue(onnx.is_file())

    def test_legacy_qwen2_runs_both_stages(self):
        args = self.args('--kind', 'vlm', '--model-name', 'qwen2-vl', '--vision-stage', 'onnx')
        commands = []
        def run(command, cwd):
            commands.append(command)
            Path(command[command.index('--savepath') + 1]).touch()
        with patch.object(converter, 'run', side_effect=run):
            converter.export_vlm_vision(args, Path('Qwen2-VL'), self.root / 'RK3588', 'RK3588')
        self.assertEqual([c[c.index('--step') + 1] for c in commands], ['1', '2'])

    def test_fp_skips_calibration(self):
        with patch.object(converter, 'run') as run, patch.object(converter, 'export_rkllm') as export:
            self.main('-q', 'fp')
        run.assert_not_called()
        self.assertIsNone(export.call_args.args[2])

    def test_onnx_external_data_copy(self):
        source = self.root / 'source' / 'vision.onnx'
        source.parent.mkdir()
        source.write_bytes(b'graph')
        (source.parent / 'weights.bin').write_bytes(b'weights')
        tensor = types.SimpleNamespace(DESCRIPTOR=types.SimpleNamespace(full_name='onnx.TensorProto'),
                                       external_data=[types.SimpleNamespace(key='location', value='weights.bin')],
                                       ListFields=lambda: [])
        with patch.dict(sys.modules, {'onnx': types.SimpleNamespace(load=lambda *a, **k: tensor)}):
            destination = self.root / 'dest' / 'renamed.onnx'
            converter.copy_onnx(source, destination)
        self.assertEqual(destination.read_bytes(), b'graph')
        self.assertEqual((destination.parent / 'weights.bin').read_bytes(), b'weights')


class EncoderTests(unittest.TestCase):
    def test_rknn_failure_is_propagated_and_released(self):
        import export_encoder_rknn as encoder
        calls = []
        class FakeRKNN:
            def __init__(self, **kwargs): pass
            def config(self, **kwargs): calls.append('config'); return 0
            def load_onnx(self, **kwargs): calls.append('load'); return -1
            def release(self): calls.append('release')
        modules = {'numpy': types.ModuleType('numpy'), 'rknn.api': types.SimpleNamespace(RKNN=FakeRKNN)}
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), patch.object(sys, 'argv', [
            'encoder', '--path', 'audio.onnx', '--model_name', 'gemma-4-audio',
            '--target-platform', 'rk3588', '--output', f'{temp}/audio.rknn']):
            with self.assertRaisesRegex(RuntimeError, 'load_onnx'):
                encoder.main()
        self.assertEqual(calls, ['config', 'load', 'release'])


if __name__ == '__main__':
    unittest.main()
