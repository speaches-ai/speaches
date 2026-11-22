from unittest.mock import Mock

import numpy as np
from pytest_mock import MockerFixture

from speaches.executors.silero_vad_v5 import SpeechTimestamp
from speaches.realtime.context import SessionContext
from speaches.realtime.input_audio_buffer import MS_SAMPLE_RATE, InputAudioBuffer
from speaches.realtime.input_audio_buffer_event_router import vad_detection_flow
from speaches.realtime.pubsub import EventPubSub
from speaches.types.realtime import InputAudioBufferSpeechStartedEvent, TurnDetection


def test_vad_starts_at_first_speech_segment_in_window(mocker: MockerFixture) -> None:
    buffer = InputAudioBuffer(Mock(spec=EventPubSub))
    buffer.append(np.zeros(3200 * MS_SAMPLE_RATE, dtype=np.float32))
    turn_detection = TurnDetection(create_response=False, threshold=0.9, silence_duration_ms=550, prefix_padding_ms=0)
    mock_vad = mocker.patch(
        "speaches.realtime.input_audio_buffer_event_router.get_speech_timestamps",
        return_value=[
            SpeechTimestamp(start=100 * MS_SAMPLE_RATE, end=300 * MS_SAMPLE_RATE),
            SpeechTimestamp(start=1100 * MS_SAMPLE_RATE, end=1600 * MS_SAMPLE_RATE),
        ],
    )
    context = Mock(spec=SessionContext, vad_model_manager=Mock())

    event = vad_detection_flow(buffer, turn_detection, context)

    assert isinstance(event, InputAudioBufferSpeechStartedEvent)
    assert event.audio_start_ms == 300
    assert buffer.vad_state.audio_start_ms == 300
    assert len(mock_vad.call_args.args[0]) == 3000 * MS_SAMPLE_RATE
