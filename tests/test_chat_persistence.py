import tempfile
import unittest
from pathlib import Path

import backend.database_service as database_service


class ChatPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_path = database_service.DATABASE_PATH
        database_service.DATABASE_PATH = Path(self.temp_dir.name) / "youtube_analyzer_test.db"
        database_service.initialize_database()

    def tearDown(self):
        database_service.DATABASE_PATH = self.original_database_path
        self.temp_dir.cleanup()

    def test_chat_session_survives_database_round_trip(self):
        messages = [
            {"role": "user", "content": "What is the newest video?"},
            {
                "role": "assistant",
                "content": "Video A",
                "sources": [
                    {
                        "video_id": "video-a",
                        "title": "Video A",
                        "video_url": "https://www.youtube.com/watch?v=video-a",
                    }
                ],
            },
        ]

        database_service.save_chat_session(
            session_id="session-1",
            channel_id="UC_TEST",
            messages=messages,
            target_video_ids=["video-a"],
        )

        restored = database_service.get_saved_chat_session("session-1")

        self.assertIsNotNone(restored)
        self.assertEqual(restored["channel_id"], "UC_TEST")
        self.assertEqual(restored["messages"], messages)
        self.assertEqual(restored["target_video_ids"], ["video-a"])

    def test_delete_chat_session_removes_persisted_history(self):
        database_service.save_chat_session(
            session_id="session-delete",
            channel_id="UC_TEST",
            messages=[{"role": "user", "content": "Hello"}],
            target_video_ids=[],
        )

        database_service.delete_saved_chat_session("session-delete")

        self.assertIsNone(database_service.get_saved_chat_session("session-delete"))


if __name__ == "__main__":
    unittest.main()
