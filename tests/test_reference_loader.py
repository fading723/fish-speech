import unittest
from unittest.mock import Mock

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.utils.schema import ServeReferenceAudio


class ReferenceLoaderCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        # Bypass backend discovery; load_by_hash only needs the in-memory cache.
        self.loader = ReferenceLoader.__new__(ReferenceLoader)
        self.loader.ref_by_hash = {}
        self.loader.encode_reference = Mock(
            side_effect=lambda **_: f"encoded-{self.loader.encode_reference.call_count}"
        )

    def test_cached_audio_uses_text_from_current_request(self) -> None:
        audio = b"same-audio"

        first = self.loader.load_by_hash(
            [ServeReferenceAudio(audio=audio, text="first transcript")],
            "on",
        )
        second = self.loader.load_by_hash(
            [ServeReferenceAudio(audio=audio, text="second transcript")],
            "on",
        )

        self.assertEqual(first, (["encoded-1"], ["first transcript"]))
        self.assertEqual(second, (["encoded-1"], ["second transcript"]))
        self.loader.encode_reference.assert_called_once_with(
            reference_audio=audio,
            enable_reference_audio=True,
        )

    def test_duplicate_audio_keeps_each_reference_text(self) -> None:
        audio = b"same-audio"

        result = self.loader.load_by_hash(
            [
                ServeReferenceAudio(audio=audio, text="speaker one"),
                ServeReferenceAudio(audio=audio, text="speaker two"),
            ],
            "on",
        )

        self.assertEqual(
            result,
            (["encoded-1", "encoded-1"], ["speaker one", "speaker two"]),
        )
        self.loader.encode_reference.assert_called_once()


if __name__ == "__main__":
    unittest.main()
