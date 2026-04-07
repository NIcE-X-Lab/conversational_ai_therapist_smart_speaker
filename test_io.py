from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.abspath("."))

import json
from src.utils import io_record

# Mock the queue
io_record.INPUT_QUEUE = Mock()
io_record.INPUT_QUEUE.get.return_value = '{"transcript": "Hello, and I am good. but I am tired", "detected_emotion": "Neutral"}'

DLA, segments = io_record.get_answer()
print("Segments:", segments)

io_record.INPUT_QUEUE.get.return_value = '{"transcript": "Yes I agree.", "detected_emotion": "Happy"}'
resp = io_record.get_resp_log()
print("Resp Log:", resp)
