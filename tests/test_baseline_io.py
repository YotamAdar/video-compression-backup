import importlib.util
from pathlib import Path
import tempfile
import unittest

from video_compression.bitstreams.baseline import pack, unpack
from video_compression.codecs.turbo_baseline import preflight


BASELINE = Path.home() / "Turbo-DDCM"


class BaselineIOTests(unittest.TestCase):
    @unittest.skipUnless((BASELINE / "turbo_ddcm/bit_stream_lex_enc.py").is_file(), "External baseline not available")
    def test_native_combination_roundtrip_and_byte_padding(self):
        # Load only the standalone CPU encoder; no torch or model imports.
        spec = importlib.util.spec_from_file_location("native_encoder", BASELINE / "turbo_ddcm/bit_stream_lex_enc.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoder = module.BitStreamEncoder(32, 3, 1, use_precomputed_combs=False)
        cases = [([31, 0, 7], [1, 0, 1]), ([0, 1, 2], [0, 0, 0]), ([29, 30, 31], [1, 0, 1])]
        for indices, signs in cases:
            encoder.add(indices, signs)
        expected = [tuple(map(list, zip(*sorted(zip(indices, signs))))) for indices, signs in cases]
        self.assertEqual(encoder.decode(unpack(pack(encoder.get_encoding()))), expected)

    def test_missing_table_fails_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "turbo_ddcm").mkdir()
            (root / "turbo_ddcm/turbo_ddcm.py").touch()
            (root / "turbo_ddcm/utils.py").write_text("SCHEDULER = list(range(30))\n".replace("list(range(30))", repr(list(range(30)))))
            before = sorted(root.rglob("*"))
            with self.assertRaises(FileNotFoundError):
                preflight(root, {"K": 32, "M": 3, "T": 30})
            self.assertEqual(sorted(root.rglob("*")), before)

    def test_invalid_payload_rejected(self):
        for value in ("", "012", "hello"):
            with self.assertRaises(ValueError):
                pack(value)
        with self.assertRaises(ValueError):
            unpack(b"")
