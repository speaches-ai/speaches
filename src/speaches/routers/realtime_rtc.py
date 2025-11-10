import asyncio
import base64
import logging
import time
from typing import Annotated

from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCPeerConnection,
    RTCRtpCodecParameters,
    RTCSessionDescription,
)
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from aiortc.sdp import SessionDescription
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from fastapi import (
    APIRouter,
    Query,
    Request,
    Response,
)
import numpy as np
from openai import AsyncOpenAI
from openai.types.beta.realtime.error_event import Error
from pydantic import ValidationError

from speaches.dependencies import (
    ConfigDependency,
    ExecutorRegistryDependency,
    TranscriptionClientDependency,
)
from speaches.realtime.context import SessionContext
from speaches.realtime.conversation_event_router import event_router as conversation_event_router
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer_event_router import (
    event_router as input_audio_buffer_event_router,
)
from speaches.realtime.response_event_router import event_router as response_event_router
from speaches.realtime.rtc.audio_stream_track import AudioStreamTrack
from speaches.realtime.session import create_session_object_configuration
from speaches.realtime.session_event_router import event_router as session_event_router
from speaches.realtime.utils import generate_event_id
from speaches.routers.realtime_ws import event_listener
from speaches.types.realtime import (
    SERVER_EVENT_TYPES,
    ErrorEvent,
    FullMessageEvent,
    InputAudioBufferAppendEvent,
    PartialMessageEvent,
    SessionCreatedEvent,
    client_event_type_adapter,
    server_event_type_adapter,
)

# NOTE: IMPORTANT! 24Khz because that's what the `input_audio_buffer.append` handler expects
SAMPLE_RATE = 24000
MIN_BUFFER_DURATION_MS = 200
MIN_BUFFER_SIZE = int(SAMPLE_RATE * MIN_BUFFER_DURATION_MS / 1000)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

event_router = EventRouter()
event_router.include_router(conversation_event_router)
event_router.include_router(input_audio_buffer_event_router)
event_router.include_router(response_event_router)
event_router.include_router(session_event_router)

# TODO: limit session duration

# https://stackoverflow.com/questions/77560930/cant-create-audio-frame-with-from-nd-array

rtc_session_tasks: dict[str, set[asyncio.Task[None]]] = {}

# Maximum size in bytes for each message fragment (just under 1 KiB)
MAX_FRAGMENT_SIZE = 900


def send_fragmented_message(channel: RTCDataChannel, message: str, event_id: str) -> None:
    """Send a message over the data channel, fragmenting it if necessary.

    Args:
        channel: The RTCDataChannel to send the message on
        message: The string message to send
        event_id: A unique ID to identify this message and its fragments

    """
    message_size = len(message)
    logger.debug(f"Processing message ({message_size} bytes)")

    if message_size <= MAX_FRAGMENT_SIZE:
        # Send as a full message
        full_message = FullMessageEvent(
            id=event_id, data=base64.b64encode(message.encode("utf-8")).decode("utf-8")
        ).model_dump_json()
        channel.send(full_message)
        logger.info(f"Sent full message ({len(full_message)} bytes)")
    else:
        # Encode the original message
        encoded_message = base64.b64encode(message.encode("utf-8")).decode("utf-8")
        # Calculate how many fragments we need
        fragment_size = MAX_FRAGMENT_SIZE - 100  # Account for the fragment metadata
        total_fragments = (len(encoded_message) + fragment_size - 1) // fragment_size

        logger.info(f"Fragmenting message into {total_fragments} fragments")

        # Split and send as fragments
        for i in range(total_fragments):
            start_pos = i * fragment_size
            end_pos = min((i + 1) * fragment_size, len(encoded_message))
            fragment_data = encoded_message[start_pos:end_pos]

            fragment = PartialMessageEvent(
                id=event_id, data=fragment_data, fragment_index=i, total_fragments=total_fragments
            ).model_dump_json()

            channel.send(fragment)
            logger.debug(f"Sent fragment {i + 1}/{total_fragments} ({len(fragment)} bytes)")

        logger.info(f"Sent all {total_fragments} fragments")


async def rtc_datachannel_sender(ctx: SessionContext, channel: RTCDataChannel) -> None:
    logger.info("Sender task started")
    q = ctx.pubsub.subscribe()

    try:
        while True:
            event = await q.get()
            if event.type not in SERVER_EVENT_TYPES:
                continue
            server_event = server_event_type_adapter.validate_python(event)
            if server_event.type == "response.audio.delta":
                logger.debug("Skipping response.audio.delta event")
                continue

            # Get JSON representation of the event
            message = server_event.model_dump_json()

            # Generate a unique ID for this message (for tracking fragments)
            event_id = str(event.event_id) if hasattr(event, "event_id") else generate_event_id()

            # Send the message, fragmenting if necessary
            logger.debug(f"Processing {event.type} event message ({len(message)} bytes)")
            send_fragmented_message(channel, message, event_id)
            logger.info(f"Sent {event.type} event message")

    except BaseException:
        logger.exception("Sender task failed")
        ctx.pubsub.subscribers.remove(q)
        raise


def message_handler(ctx: SessionContext, message: str) -> None:
    logger.info(f"Message received: {message}")
    try:
        event = client_event_type_adapter.validate_json(message)
    except ValidationError as e:
        ctx.pubsub.publish_nowait(ErrorEvent(error=Error(type="invalid_request_error", message=str(e))))
        logger.exception(f"Received an invalid client event: {message}")
        return

    logger.debug(f"Received {event.type} event")
    ctx.pubsub.publish_nowait(event)
    # asyncio.create_task(event_router.dispatch(ctx, event))


async def audio_receiver(ctx: SessionContext, track: RemoteStreamTrack) -> None:
    # Initialize buffer to store audio data
    buffer = np.array([], dtype=np.int16)

    while True:
        frames = await track.recv()
        # ensure that the received frames are of expected format
        assert isinstance(frames, AudioFrame)
        assert frames.sample_rate == 48000
        assert frames.layout.name == "stereo"
        assert frames.format.name == "s16"

        resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        frames = resampler.resample(frames)

        # Accumulate audio data
        for frame in frames:
            arr = frame.to_ndarray()
            buffer = np.append(buffer, arr.flatten())  # Flatten and append to buffer

            # When buffer reaches or exceeds target size, emit event
            if len(buffer) >= MIN_BUFFER_SIZE:
                # Convert to bytes and emit event
                audio_bytes = buffer.tobytes()
                assert len(audio_bytes) == len(buffer) * 2, "Audio sample width is not 2 bytes"
                ctx.pubsub.publish_nowait(
                    InputAudioBufferAppendEvent(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(audio_bytes).decode(),
                    )
                )

                buffer = np.array([], dtype=np.int16)


def datachannel_handler(ctx: SessionContext, channel: RTCDataChannel) -> None:
    logger.info(f"Data channel created: {channel}")

    # Send the session created event - use the fragmentation logic for consistency
    session_created_event = SessionCreatedEvent(session=ctx.session)
    session_message = session_created_event.model_dump_json()

    # Send the session created event using our helper function
    logger.debug(f"Sending session.created event message ({len(session_message)} bytes)")
    send_fragmented_message(channel, session_message, str(session_created_event.event_id))
    logger.info("Sent session.created event message")

    # Start the data channel sender task
    rtc_session_tasks[ctx.session.id].add(asyncio.create_task(rtc_datachannel_sender(ctx, channel)))

    # Set up the message handler
    channel.on("message")(lambda message: message_handler(ctx, message))

    @channel.on("open")
    def _handle_datachannel_open(*args, **kwargs) -> None:
        logger.info(f"Data channel opened: {channel.id} (args={args}, kwargs={kwargs})")

    @channel.on("close")
    def _handle_datachannel_close(*args, **kwargs) -> None:
        logger.info(f"Data channel closed: {channel.id} (args={args}, kwargs={kwargs})")

    @channel.on("closing")
    def _handle_datachannel_closing(*args, **kwargs) -> None:
        logger.info(f"Data channel closing: {channel.id} (args={args}, kwargs={kwargs})")

    @channel.on("error")
    def _handle_datachannel_error(*args, **kwargs) -> None:
        logger.error(f"Data channel error: {channel.id} (args={args}, kwargs={kwargs})")

    @channel.on("bufferedamountlow")
    def _handle_datachannel_bufferedamountlow(*args, **kwargs) -> None:
        logger.info(f"Data channel buffered amount low: {channel.id} (args={args}, kwargs={kwargs})")


def iceconnectionstatechange_handler(_ctx: SessionContext, pc: RTCPeerConnection) -> None:
    logger.info(f"ICE connection state changed to {pc.iceConnectionState}")
    if pc.iceConnectionState in ["failed", "closed"]:
        logger.info("Peer connection closed")


def track_handler(ctx: SessionContext, track: RemoteStreamTrack) -> None:
    logger.info(f"Track received: kind={track.kind}")
    if track.kind == "audio":
        # Start a task to log audio data
        rtc_session_tasks[ctx.session.id].add(asyncio.create_task(audio_receiver(ctx, track)))
    track.on("ended")(lambda: logger.info(f"Track ended: kind={track.kind}"))


@router.post("/v1/realtime")
async def realtime_webrtc(
    request: Request,
    model: Annotated[str, Query(...)],
    config: ConfigDependency,
    transcription_client: TranscriptionClientDependency,
    executor_registry: ExecutorRegistryDependency,
) -> Response:
    completion_client = AsyncOpenAI(
        base_url=f"http://{config.host}:{config.port}/v1",
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=0,
    ).chat.completions
    ctx = SessionContext(
        transcription_client=transcription_client,
        completion_client=completion_client,
        vad_model_manager=executor_registry.vad.model_manager,
        session=create_session_object_configuration(model, "conversation", None, None),
    )
    rtc_session_tasks[ctx.session.id] = set()

    # TODO: handle both application/sdp and application/json
    sdp = (await request.body()).decode("utf-8")
    # session_description = SessionDescription.parse(sdp)
    # for media in session_description.media:
    #     logger.info(f"offer media: {media}")
    offer = RTCSessionDescription(sdp=sdp, type="offer")
    logger.info(f"Received offer: {offer.sdp[:5]}")

    # Create a new RTCPeerConnection
    rtc_configuration = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(rtc_configuration)

    pc.on("datachannel", lambda channel: datachannel_handler(ctx, channel))
    pc.on("iceconnectionstatechange", lambda: iceconnectionstatechange_handler(ctx, pc))
    pc.on("track", lambda track: track_handler(ctx, track))
    pc.on(
        "icegatheringstatechange",
        lambda: logger.info(f"ICE gathering state changed to {pc.iceGatheringState}"),
    )
    # NOTE: will never be called according to https://github.com/aiortc/aiortc/issues/1344
    pc.on(
        "icecandidate",
        lambda *args, **kwargs: logger.info(f"ICE candidate: {args}, {kwargs}. {pc.iceGatheringState}"),
    )

    logger.info("Created peer connection")

    # NOTE: is relay needed?
    audio_track = AudioStreamTrack(ctx)
    pc.addTrack(audio_track)

    # Set the remote description and create an answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    assert answer is not None
    answer_session_description = SessionDescription.parse(answer.sdp)

    # Remove all codecs except opus. This **should** ensure that we only receive opus audio. This is done because no other codecs supported.
    for media in answer_session_description.media:
        if media.kind != "audio":
            continue
        filtered_codecs: list[RTCRtpCodecParameters] = []
        for codec in media.rtp.codecs:
            if codec.name != "opus":
                logger.info(f"Removing codec: {codec}")
                continue
            filtered_codecs.append(codec)
        if len(filtered_codecs) == 0:
            logger.error("No appropriate codecs found")
        media.rtp.codecs = filtered_codecs
        logger.info(f"Filtered codecs: {media.rtp.codecs}")

    start = time.perf_counter()
    # NOTE: when connected to Tailscale, this step takes ~5 seconds (unless list of iceServers is empty). Somewhat relevant https://groups.google.com/g/discuss-webrtc/c/MYTwERXGrM8
    await pc.setLocalDescription(RTCSessionDescription(sdp=str(answer_session_description), type="answer"))
    logger.info(f"Setting local description took {time.perf_counter() - start:.3f} seconds")

    rtc_session_tasks[ctx.session.id].add(asyncio.create_task(event_listener(ctx)))

    return Response(content=pc.localDescription.sdp, media_type="text/plain charset=utf-8")
