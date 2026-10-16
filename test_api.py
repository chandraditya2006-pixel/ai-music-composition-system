"""
test_api.py
===========
Automated, non-destructive test suite for the AI Music Composition Flask API.
Tests request validation, malformed payloads, boundaries, downloads, and security.
"""

import unittest
import json
from app import app


class APIReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    # -------------------------------------------------------------
    # 1. Health Endpoint
    # -------------------------------------------------------------
    def test_health_check(self):
        """GET /api/health should return 200 with healthy status and metadata."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))
        self.assertEqual(data.get('sequence_length'), 20)
        self.assertEqual(data.get('unique_pitches_count'), 71)

    # -------------------------------------------------------------
    # 2. Malformed, Non-JSON, and Empty Payloads
    # -------------------------------------------------------------
    def test_empty_body_rejected(self):
        """POST /api/generate with an empty body should return 400 Bad Request."""
        # Empty body with JSON content type
        response = self.client.post(
            '/api/generate',
            data='',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')
        self.assertIn('empty', data.get('message', '').lower())

    def test_empty_body_without_content_type_rejected(self):
        """POST /api/generate with no content-type and empty body returns 400."""
        response = self.client.post('/api/generate', data='')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')
        self.assertIn('empty', data.get('message', '').lower())

    def test_malformed_json_rejected(self):
        """POST /api/generate with malformed JSON syntax should return 400 Bad Request."""
        response = self.client.post(
            '/api/generate',
            data='{num_notes: 40, bad_syntax:',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')
        self.assertIn('malformed', data.get('message', '').lower())

    def test_non_json_content_type_rejected(self):
        """POST /api/generate with non-JSON content-type should return 400 Bad Request."""
        response = self.client.post(
            '/api/generate',
            data='num_notes=40&temperature=0.8',
            content_type='application/x-www-form-urlencoded'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')
        self.assertIn('content-type', data.get('message', '').lower())

    def test_plain_text_rejected(self):
        """POST /api/generate with text/plain body should return 400 Bad Request."""
        response = self.client.post(
            '/api/generate',
            data='generate 50 notes please',
            content_type='text/plain'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')
        self.assertIn('content-type', data.get('message', '').lower())

    def test_non_object_json_rejected(self):
        """POST /api/generate with JSON array or primitive should return 400."""
        response = self.client.post(
            '/api/generate',
            data=json.dumps([1, 2, 3]),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')
        self.assertIn('object', data.get('message', '').lower())

    # -------------------------------------------------------------
    # 3. Parameter Boundary Validation
    # -------------------------------------------------------------
    def test_num_notes_below_min_rejected(self):
        """num_notes < 10 must return 400."""
        response = self.client.post(
            '/api/generate',
            json={'num_notes': 9, 'temperature': 0.8}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('between 10 and 500', response.get_json().get('message', ''))

    def test_num_notes_above_max_rejected(self):
        """num_notes > 500 must return 400."""
        response = self.client.post(
            '/api/generate',
            json={'num_notes': 501, 'temperature': 0.8}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('between 10 and 500', response.get_json().get('message', ''))

    def test_num_notes_invalid_type_rejected(self):
        """Non-numeric num_notes must return 400."""
        response = self.client.post(
            '/api/generate',
            json={'num_notes': 'thirty', 'temperature': 0.8}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('integer', response.get_json().get('message', '').lower())

    def test_temperature_below_min_rejected(self):
        """temperature < 0.1 must return 400."""
        response = self.client.post(
            '/api/generate',
            json={'num_notes': 20, 'temperature': 0.05}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('between 0.1 and 1.5', response.get_json().get('message', ''))

    def test_temperature_above_max_rejected(self):
        """temperature > 1.5 must return 400."""
        response = self.client.post(
            '/api/generate',
            json={'num_notes': 20, 'temperature': 1.55}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('between 0.1 and 1.5', response.get_json().get('message', ''))

    def test_temperature_invalid_type_rejected(self):
        """Non-numeric temperature must return 400."""
        response = self.client.post(
            '/api/generate',
            json={'num_notes': 20, 'temperature': 'warm'}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('number', response.get_json().get('message', '').lower())

    # -------------------------------------------------------------
    # 4. Valid Music Generation Requests
    # -------------------------------------------------------------
    def test_valid_generation_custom_params(self):
        """Valid JSON request should return 201 Created and valid metadata."""
        response = self.client.post(
            '/api/generate',
            json={'num_notes': 12, 'temperature': 0.75}
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('num_notes'), 12)
        self.assertEqual(data.get('temperature'), 0.75)
        self.assertTrue(data.get('filename', '').endswith('.mid'))
        self.assertTrue(data.get('download_url', '').startswith('/api/download/'))

        # Also test downloading the freshly generated file
        dl_response = self.client.get(data.get('download_url'))
        self.assertEqual(dl_response.status_code, 200)
        self.assertEqual(dl_response.mimetype, 'audio/midi')
        self.assertTrue(dl_response.data.startswith(b'MThd'))
        dl_response.close()

    def test_valid_generation_empty_json_object_uses_defaults(self):
        """Valid empty JSON object '{}' should succeed and apply default parameters (50 notes, 0.8 temp)."""
        response = self.client.post(
            '/api/generate',
            json={}
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('num_notes'), 50)
        self.assertEqual(data.get('temperature'), 0.8)

    # -------------------------------------------------------------
    # 5. Download Security and Safety
    # -------------------------------------------------------------
    def test_download_nonexistent_file(self):
        """Requesting a missing file should return 404."""
        response = self.client.get('/api/download/missing_file_00000.mid')
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')

    def test_download_path_traversal_rejected(self):
        """Path traversal attempt should return 400."""
        response = self.client.get('/api/download/../models/music_preprocessing.json')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')

    def test_download_non_midi_rejected(self):
        """Requesting a non-.mid file should return 400."""
        response = self.client.get('/api/download/app.py')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'error')


if __name__ == '__main__':
    unittest.main(verbosity=2)
