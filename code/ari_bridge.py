import asyncio
import audioop
import base64
import logging
import os
import random
import struct
import time
import uuid
from queue import Queue
import threading
import contextlib

import ari

logger = logging.getLogger(__name__)

ARI_MODE = os.getenv("ARI_MODE", "1") == "1"
ARI_URL = os.getenv("ARI_URL", "http://127.0.0.1:8088")
ARI_USERNAME = os.getenv("ARI_USERNAME", "ari-user")
ARI_PASSWORD = os.getenv("ARI_PASSWORD", "ari-password")
ARI_APP_NAME = os.getenv("ARI_APP_NAME", "ai-bot")
ARI_MEDIA_IP = os.getenv("ARI_MEDIA_IP", "127.0.0.1")
ARI_MEDIA_PORT = int(os.getenv("ARI_MEDIA_PORT", "10000"))


class RtpUdpProtocol(asyncio.DatagramProtocol):
    """Simple UDP protocol to receive and send RTP packets from Asterisk."""

    def __init__(self, audio_queue: asyncio.Queue) -> None:
        self.audio_queue = audio_queue
        self.transport: asyncio.DatagramTransport | None = None
        self.remote_addr: tuple | None = None
        self.sequence: int = random.randint(0, 65535)
        self.timestamp: int = random.randint(0, 4294967295)
        self.ssrc: int = random.getrandbits(32)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        logger.info(
            f"ARI UDP server listening on {ARI_MEDIA_IP}:{ARI_MEDIA_PORT}"
        )

    def datagram_received(self, data: bytes, addr) -> None:
        if not self.remote_addr:
            self.remote_addr = addr
            logger.info(f"First RTP packet from Asterisk at {addr}")
        if len(data) > 12:
            self.audio_queue.put_nowait(data[12:])

    def send_audio(self, payload: bytes) -> None:
        if self.transport and self.remote_addr:
            samples = len(payload)
            header = struct.pack(
                "!BBHII",
                0x80,
                0x00,
                self.sequence,
                self.timestamp,
                self.ssrc,
            )
            self.sequence = (self.sequence + 1) & 0xFFFF
            self.timestamp = (self.timestamp + samples) & 0xFFFFFFFF
            self.transport.sendto(header + payload, self.remote_addr)


def _ari_forward_audio(asterisk_q: asyncio.Queue, target_q: asyncio.Queue) -> asyncio.Task:
    async def forward():
        while True:
            ulaw = await asterisk_q.get()
            pcm_8k = audioop.ulaw2lin(ulaw, 2)
            pcm_48k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 48000, None)
            await target_q.put({"pcm": pcm_48k})
    return asyncio.create_task(forward())


def _ari_send_tts(message_q: asyncio.Queue, udp: RtpUdpProtocol) -> asyncio.Task:
    async def sender():
        buffer = b""
        frame_size = 160 * 2  # 20ms of 16-bit audio at 8kHz
        next_send = time.monotonic()
        while True:
            msg = await message_q.get()
            msg_type = msg.get("type")
            if msg_type == "tts_chunk":
                pcm_48k = base64.b64decode(msg["content"])
                pcm_8k, _ = audioop.ratecv(pcm_48k, 2, 1, 48000, 8000, None)
                buffer += pcm_8k
                while len(buffer) >= frame_size:
                    frame = buffer[:frame_size]
                    buffer = buffer[frame_size:]
                    ulaw = audioop.lin2ulaw(frame, 2)
                    now = time.monotonic()
                    if now < next_send:
                        await asyncio.sleep(next_send - now)
                    udp.send_audio(ulaw)
                    next_send += 0.02
            elif msg_type == "tts_end":
                if buffer:
                    pad_len = frame_size - len(buffer)
                    buffer += b"\x00" * pad_len
                    ulaw = audioop.lin2ulaw(buffer, 2)
                    now = time.monotonic()
                    if now < next_send:
                        await asyncio.sleep(next_send - now)
                    udp.send_audio(ulaw)
                buffer = b""
                next_send = time.monotonic() + 0.02
    return asyncio.create_task(sender())


def _ari_play_welcome(app, message_q: asyncio.Queue) -> asyncio.Task:
    async def worker():
        text = "Welcome to the realtime voice chat"
        temp_q: Queue[bytes] = Queue()
        stop_event = threading.Event()
        await asyncio.to_thread(
            app.state.SpeechPipelineManager.audio.synthesize,
            text,
            temp_q,
            stop_event,
            "[ARI Welcome]",
        )
        while not temp_q.empty():
            chunk = temp_q.get()
            b64 = app.state.Upsampler.get_base64_chunk(chunk)
            if b64:
                message_q.put_nowait({"type": "tts_chunk", "content": b64})
        final = app.state.Upsampler.flush_base64_chunk()
        if final:
            message_q.put_nowait({"type": "tts_chunk", "content": final})
        message_q.put_nowait({"type": "tts_end"})
    return asyncio.create_task(worker())


def _ari_send_tts_and_forward_audio(app, message_q: asyncio.Queue, asterisk_q: asyncio.Queue,
                                    udp: RtpUdpProtocol, callbacks, send_tts_chunks_func) -> list[asyncio.Task]:
    audio_q = asyncio.Queue()
    tasks = [
        _ari_forward_audio(asterisk_q, audio_q),
        asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_q)),
        asyncio.create_task(send_tts_chunks_func(app, message_q, callbacks)),
        _ari_send_tts(message_q, udp),
    ]
    return tasks


async def _ari_handle_call(client, channel_id: str, app, asterisk_q: asyncio.Queue,
                           udp: RtpUdpProtocol, loop: asyncio.AbstractEventLoop,
                           callbacks_class, send_tts_chunks_func) -> None:
    live_channel = client.channels.get(channelId=channel_id)
    caller_id = live_channel.json.get("caller", {}).get("number")
    logger.info(f"Handling new ARI call: {channel_id} from {caller_id}")
    message_q = asyncio.Queue()
    session_id = str(uuid.uuid4())
    callbacks = callbacks_class(app, message_q, session_id=session_id, caller_id=caller_id)
    if hasattr(app.state.SpeechPipelineManager.llm, "open_persistent_ws") and callable(app.state.SpeechPipelineManager.llm.open_persistent_ws):
        app.state.SpeechPipelineManager.llm.open_persistent_ws()

    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial
    app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    app.state.AudioInputProcessor.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
    app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start
    app.state.AudioInputProcessor.silence_active_callback = callbacks.on_silence_active
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text

    tasks = _ari_send_tts_and_forward_audio(app, message_q, asterisk_q, udp, callbacks, send_tts_chunks_func)
    call_end = asyncio.Event()
    bridge = None
    try:
        live_channel.answer()
        await asyncio.sleep(0.05)

        bridge = client.bridges.create(type="mixing")
        external_host = f"{ARI_MEDIA_IP}:{ARI_MEDIA_PORT}"
        external_channel = client.channels.externalMedia(
            app=ARI_APP_NAME,
            external_host=external_host,
            format="ulaw",
            encapsulation="rtp",
        )
        bridge.addChannel(channel=external_channel.id)
        bridge.addChannel(channel=live_channel.id)

        def on_end(channel_obj, ev):
            if channel_obj.id == live_channel.id:
                call_end.set()

        live_channel.on_event("StasisEnd", on_end)

        callbacks.tts_to_client = True
        app.state.SpeechPipelineManager.prepare_generation(
            "hello",
            session_id=session_id,
            caller_id=caller_id,
        )
        await call_end.wait()
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if live_channel:
            with contextlib.suppress(Exception):
                live_channel.hangup()
        if bridge:
            with contextlib.suppress(Exception):
                bridge.destroy()
        udp.remote_addr = None
        app.state.SpeechPipelineManager.reset()
        app.state.AudioInputProcessor.interrupted = False
        if hasattr(app.state.SpeechPipelineManager.llm, "close_persistent_ws") and callable(app.state.SpeechPipelineManager.llm.close_persistent_ws):
            app.state.SpeechPipelineManager.llm.close_persistent_ws()


async def start_ari_bridge(app, callbacks_class, send_tts_chunks_func) -> None:
    if not ARI_MODE:
        return

    loop = asyncio.get_running_loop()
    asterisk_q: asyncio.Queue = asyncio.Queue()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: RtpUdpProtocol(asterisk_q),
        local_addr=(ARI_MEDIA_IP, ARI_MEDIA_PORT),
    )
    logger.info("ARI UDP transport created")

    call_q: asyncio.Queue = asyncio.Queue()
    client = ari.connect(ARI_URL, ARI_USERNAME, ARI_PASSWORD)

    def on_stasis_start_sync(objs, event):
        channel = objs["channel"]
        channel_name = channel.json.get("name")
        if not channel_name or not channel_name.startswith("SIP/"):
            return
        loop.call_soon_threadsafe(call_q.put_nowait, channel.id)

    async def call_worker():
        while True:
            chan_id = await call_q.get()
            logger.info(f"New channel queued for ARI handling: {chan_id}")
            asyncio.create_task(
                _ari_handle_call(client, chan_id, app, asterisk_q, protocol, loop, callbacks_class, send_tts_chunks_func)
            )

    asyncio.create_task(call_worker())
    client.on_channel_event("StasisStart", on_stasis_start_sync)

    def run_client():
        logger.info("ARI client thread started")
        client.run(apps=ARI_APP_NAME)

    threading.Thread(target=run_client, daemon=True).start()
    logger.info("ARI bridge started.")
    await asyncio.Event().wait()
