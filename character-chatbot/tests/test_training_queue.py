"""SDK registration and small, versioned queue envelopes."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import training_subscribers
from app.services.training_queue import VercelTrainingQueue


class TrainingQueueTests(unittest.TestCase):
    def test_subscriber_registration_and_message_validation(self):
        self.assertIsNotNone(training_subscribers.handle_plan)
        self.assertIsNotNone(training_subscribers.handle_chunk)
        self.assertIsNotNone(training_subscribers.handle_finalize)
        self.assertEqual(training_subscribers._ids({"v": 1, "job_id": "j"}), ("j", None))
        self.assertEqual(training_subscribers._ids({"v": 1, "job_id": "j", "chunk_id": "c"}, chunk=True), ("j", "c"))
        self.assertIsNone(training_subscribers._ids({"v": 2, "job_id": "j"}))
        self.assertIsNone(training_subscribers._ids({"v": 1, "job_id": "j", "chunk_id": 4}, chunk=True))

    def test_adapter_sends_only_ids_with_idempotency_keys(self):
        queue = VercelTrainingQueue()
        with patch("vercel.queue.send", new_callable=AsyncMock) as send:
            asyncio.run(queue.publish_plan("job"))
            asyncio.run(queue.publish_chunk("job", "chunk"))
            asyncio.run(queue.publish_finalize("job"))
        self.assertEqual(send.call_count, 3)
        self.assertEqual(send.call_args_list[1].args[1], {"v": 1, "job_id": "job", "chunk_id": "chunk"})
        self.assertEqual(send.call_args_list[1].kwargs["idempotency_key"], "chunk-chunk")
