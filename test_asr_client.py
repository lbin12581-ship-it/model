import unittest

from summary_system.asr_client import ASRClient, ASRConfig, _build_multipart_form


class ASRClientTests(unittest.TestCase):
    def test_extract_text_from_transcription_json(self):
        data = {
            "transcripts": [
                {
                    "text": "全文",
                    "sentences": [
                        {"text": "第一句。", "speaker_id": 0},
                        {"text": "第二句。", "speaker_id": 1},
                    ],
                }
            ]
        }
        text = ASRClient(ASRConfig(api_key="test"))._extract_text(data)
        self.assertIn("说话人0：第一句。", text)
        self.assertIn("说话人1：第二句。", text)

    def test_successful_transcription_urls(self):
        output = {
            "results": [
                {"subtask_status": "SUCCEEDED", "transcription_url": "https://example.com/a.json"},
                {"subtask_status": "FAILED", "transcription_url": "https://example.com/b.json"},
            ]
        }
        urls = ASRClient._successful_transcription_urls(output)
        self.assertEqual(urls, ["https://example.com/a.json"])

    def test_multipart_form_contains_file(self):
        body, boundary = _build_multipart_form({"key": "tmp/a.wav"}, "file", "a.wav", b"abc", "audio/wav")
        self.assertIn(boundary.encode("utf-8"), body)
        self.assertIn(b'name="key"', body)
        self.assertIn(b'filename="a.wav"', body)
        self.assertIn(b"abc", body)


if __name__ == "__main__":
    unittest.main()
