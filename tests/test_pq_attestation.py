# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Quantova Inc

import base64
import os
import tempfile
import unittest

from fdbja import pq_attestation as pqa

RESULTS = {"system": "duplex-a", "tor": 0.81, "turnnat": 0.62}


class PqAttestationTest(unittest.TestCase):
    def setUp(self):
        try:
            import dilithium_py  # noqa: F401
        except ImportError:
            self.skipTest("dilithium-py not installed")
        self.payload = pqa.canonicalize(RESULTS)
        self.public_key, self.secret_key = pqa.generate_keypair()

    def test_build_and_verify_roundtrip(self):
        att = pqa.build_attestation(self.payload, self.public_key, self.secret_key)
        self.assertEqual(att["scheme"], "ML-DSA-65")
        self.assertEqual(att["issuer"], "Quantova")
        self.assertTrue(pqa.verify_attestation(self.payload, att))
        self.assertTrue(pqa.verify_attestation(self.payload, att, self.public_key))

    def test_tampered_results_rejected(self):
        att = pqa.build_attestation(self.payload, self.public_key, self.secret_key)
        altered = pqa.canonicalize({**RESULTS, "tor": 0.0})
        self.assertFalse(pqa.verify_attestation(altered, att))

    def test_wrong_signer_rejected_when_pinned(self):
        other_public_key, _ = pqa.generate_keypair()
        att = pqa.build_attestation(self.payload, self.public_key, self.secret_key)
        self.assertFalse(pqa.verify_attestation(self.payload, att, other_public_key))

    def test_forged_signature_rejected(self):
        att = pqa.build_attestation(self.payload, self.public_key, self.secret_key)
        att["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
        self.assertFalse(pqa.verify_attestation(self.payload, att))

    def test_malformed_field_rejected(self):
        att = pqa.build_attestation(self.payload, self.public_key, self.secret_key)
        att["public_key"] = "!!!not base64!!!"
        self.assertFalse(pqa.verify_attestation(self.payload, att))

    def test_signer_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "signer.json")
            pqa.save_signer(path, self.public_key, self.secret_key)
            loaded_public, loaded_secret = pqa.load_signer(path)
        self.assertEqual(loaded_public, self.public_key)
        self.assertEqual(loaded_secret, self.secret_key)


if __name__ == "__main__":
    unittest.main()
