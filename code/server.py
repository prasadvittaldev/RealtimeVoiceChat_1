import asyncio
import audioop
import base64
import logging
import os
import random
import struct
import time
import uuid
from queue import Queue, Empty
from types import SimpleNamespace
import threading

from logsetup import setup_logging
from upsample_overlap import UpsampleOverlap
from audio_in import AudioInputProcessor
from speech_pipeline_manager import SpeechPipelineManager
from colors import Colors
import ari

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🖥️👋 Starting ARI voice pipeline server")

# Configuration
TTS_START_ENGINE = os.getenv("TTS_ENGINE", "kokoro")
TTS_ORPHEUS_MODEL = os.getenv("TTS_ORPHEUS_MODEL", "lex-au/Orpheus-3b-FT-Q8_0.gguf")
LLM_START_PROVIDER = os.getenv("LLM_PROVIDER", "websocket")
LLM_START_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
NO_THINK = os.getenv("NO_THINK", "0") == "1"
DIRECT_STREAM = TTS_START_ENGINE == "kokoro"

ARI_URL = os.getenv("ARI_URL", "http://127.0.0.1:8088")
ARI_USERNAME = os.getenv("ARI_USERNAME", "ari-user")
ARI_PASSWORD = os.getenv("ARI_PASSWORD", "ari-password")
ARI_APP_NAME = os.getenv("ARI_APP_NAME", "ai-bot")
ARI_MEDIA_IP = os.getenv("ARI_MEDIA_IP", "127.0.0.1")
ARI_MEDIA_PORT = int(os.getenv("ARI_MEDIA_PORT", "10000"))

BARGE_ENABLED = os.getenv("BARGE_ENABLED", "1") == "1"

try:
    MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 50))
    logger.info(f"🖥️⚙️ Audio queue size limit set to: {MAX_AUDIO_QUEUE_SIZE}")
except ValueError:
    logger.warning("🖥️⚠️ Invalid MAX_AUDIO_QUEUE_SIZE env var. Using default: 50")
    MAX_AUDIO_QUEUE_SIZE = 50


class App:
    """Lightweight container for shared state."""
    def __init__(self):
        self.state = SimpleNamespace()


class RtpUdpProtocol(asyncio.DatagramProtocol):
    """UDP protocol for receiving and sending RTP packets from Asterisk."""

    def __init__(self, audio_queue: asyncio.Queue) -> None:
        self.audio_queue = audio_queue
        self.transport = None
        self.remote_addr = None
        self.sequence = random.randint(0, 65535)
        self.timestamp = random.randint(0, 4294967295)
        self.ssrc = random.getrandbits(32)

    def connection_made(self, transport):
        self.transport = transport
        logger.info(f"ARI UDP server listening on {ARI_MEDIA_IP}:{ARI_MEDIA_PORT}")

    def datagram_received(self, data: bytes, addr):
        if not self.remote_addr:
            self.remote_addr = addr
            logger.info(f"First RTP packet from Asterisk at {addr}")
        if len(data) > 12:
            self.audio_queue.put_nowait(data[12:])

    def send_audio(self, payload: bytes) -> None:
        if self.transport and self.remote_addr:
            header = struct.pack("!BBHII", 0x80, 0x00, self.sequence, self.timestamp, self.ssrc)
            self.sequence = (self.sequence + 1) & 0xFFFF
            self.timestamp = (self.timestamp + len(payload)) & 0xFFFFFFFF
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


async def _reset_interrupt_flag_async(app: App, callbacks: "TranscriptionCallbacks"):
    await asyncio.sleep(1)
    if app.state.AudioInputProcessor.interrupted:
        app.state.AudioInputProcessor.interrupted = False
        callbacks.interruption_time = 0


def send_tts_chunks(app: App, message_queue: asyncio.Queue, callbacks: "TranscriptionCallbacks") -> asyncio.Task:
    async def runner():
        while True:
            await asyncio.sleep(0.001)
            if not callbacks.tts_to_client:
                continue
            gen = app.state.SpeechPipelineManager.running_generation
            if not gen or gen.abortion_started:
                continue
            if not gen.audio_quick_finished:
                gen.tts_quick_allowed_event.set()
            try:
                chunk = gen.audio_chunks.get_nowait()
            except Empty:
                final_expected = gen.quick_answer_provided
                if not final_expected or gen.audio_final_finished:
                    callbacks.send_final_assistant_answer()
                    app.state.SpeechPipelineManager.running_generation = None
                    callbacks.tts_chunk_sent = False
                    callbacks.tts_client_playing = False
                    callbacks.tts_to_client = False
                    message_queue.put_nowait({"type": "tts_end"})
                continue
            b64 = app.state.Upsampler.get_base64_chunk(chunk)
            message_queue.put_nowait({"type": "tts_chunk", "content": b64})
            if not callbacks.tts_chunk_sent:
                asyncio.create_task(_reset_interrupt_flag_async(app, callbacks))
            callbacks.tts_chunk_sent = True
            callbacks.tts_client_playing = True
    return asyncio.create_task(runner())


class TranscriptionCallbacks:
    def __init__(self, app: App, message_queue: asyncio.Queue, session_id: str | None = None, caller_id: str | None = None):
        self.app = app
        self.message_queue = message_queue
        self.session_id = session_id
        self.caller_id = caller_id
        self.tts_to_client = False
        self.tts_chunk_sent = False
        self.tts_client_playing = False
        self.user_interrupted = False
        self.interruption_time = 0.0
        self.assistant_answer = ""
        self.final_transcription = ""
        self.partial_transcription = ""

    def on_partial(self, txt: str):
        self.partial_transcription = txt

    def on_tts_allowed_to_synthesize(self):
        gen = self.app.state.SpeechPipelineManager.running_generation
        if gen and not gen.abortion_started:
            gen.tts_quick_allowed_event.set()

    def on_potential_sentence(self, txt: str):
        pass

    def on_potential_final(self, txt: str):
        pass

    def on_potential_abort(self):
        pass

    def on_before_final(self, audio: bytes, txt: str):
        if not self.app.state.AudioInputProcessor.interrupted:
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time()
        self.tts_to_client = True
        self.final_transcription = txt
        self.message_queue.put_nowait({"type": "final_user_request", "content": txt})
        self.app.state.SpeechPipelineManager.history.append({"role": "user", "content": txt})

    def on_final(self, txt: str):
        self.final_transcription = txt
        self.app.state.SpeechPipelineManager.prepare_generation(
            txt,
            session_id=self.session_id,
            caller_id=self.caller_id,
        )

    def on_silence_active(self, silence_active: bool):
        pass

    def on_partial_assistant_text(self, txt: str):
        self.assistant_answer = txt

    def on_recording_start(self):
        if self.tts_client_playing and BARGE_ENABLED:
            self.tts_to_client = False
            self.tts_client_playing = False
            self.user_interrupted = True
            self.message_queue.put_nowait({"type": "tts_end"})
            self.app.state.SpeechPipelineManager.abort_generation("on_recording_start, user interrupt")

    def send_final_assistant_answer(self, forced: bool = False):
        gen = self.app.state.SpeechPipelineManager.running_generation
        final_answer = ""
        if gen:
            final_answer = (gen.quick_answer + gen.final_answer).strip()
        if not final_answer and forced:
            final_answer = self.assistant_answer.strip()
        if final_answer:
            self.message_queue.put_nowait({"type": "final_assistant_answer", "content": final_answer})
            self.app.state.SpeechPipelineManager.history.append({"role": "assistant", "content": final_answer})


async def _ari_handle_call(client, channel_id: str, app: App, asterisk_q: asyncio.Queue, udp: RtpUdpProtocol):
    live_channel = client.channels.get(channelId=channel_id)
    caller_id = live_channel.json.get("caller", {}).get("number")
    logger.info(f"Handling new ARI call: {channel_id} from {caller_id}")
    message_q = asyncio.Queue()
    audio_q = asyncio.Queue()
    session_id = str(uuid.uuid4())
    callbacks = TranscriptionCallbacks(app, message_q, session_id=session_id, caller_id=caller_id)
    llm = app.state.SpeechPipelineManager.llm
    open_ws = getattr(llm, "open_persistent_ws", None)
    if callable(open_ws):
        open_ws()

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

    tasks = [
        _ari_forward_audio(asterisk_q, audio_q),
        asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_q)),
        send_tts_chunks(app, message_q, callbacks),
        _ari_send_tts(message_q, udp),
    ]
    call_end = asyncio.Event()
    bridge = None
    try:
        live_channel.answer()
        await asyncio.sleep(0.05)
        bridge = client.bridges.create(type="mixing")
        external_channel = client.channels.externalMedia(
            app=ARI_APP_NAME,
            external_host=f"{ARI_MEDIA_IP}:{ARI_MEDIA_PORT}",
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
            try:
                live_channel.hangup()
            except Exception:
                pass
        if bridge:
            try:
                bridge.destroy()
            except Exception:
                pass
        udp.remote_addr = None
        app.state.SpeechPipelineManager.reset()
        app.state.AudioInputProcessor.interrupted = False
        close_ws = getattr(llm, "close_persistent_ws", None)
        if callable(close_ws):
            close_ws()


async def start_ari_bridge(app: App):
    loop = asyncio.get_running_loop()
    asterisk_q: asyncio.Queue = asyncio.Queue()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: RtpUdpProtocol(asterisk_q),
        local_addr=(ARI_MEDIA_IP, ARI_MEDIA_PORT),
    )
    call_q: asyncio.Queue = asyncio.Queue()
    client = ari.connect(ARI_URL, ARI_USERNAME, ARI_PASSWORD)

    def on_stasis_start_sync(objs, event):
        channel = objs["channel"]
        channel_name = channel.json.get("name")
        if channel_name and channel_name.startswith("SIP/"):
            loop.call_soon_threadsafe(call_q.put_nowait, channel.id)

    async def call_worker():
        while True:
            chan_id = await call_q.get()
            asyncio.create_task(_ari_handle_call(client, chan_id, app, asterisk_q, protocol))

    asyncio.create_task(call_worker())
    client.on_channel_event("StasisStart", on_stasis_start_sync)

    def run_client():
        client.run(apps=ARI_APP_NAME)

    threading.Thread(target=run_client, daemon=True).start()
    logger.info("ARI bridge started.")


async def main():
    app = App()
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_engine=TTS_START_ENGINE,
        llm_provider=LLM_START_PROVIDER,
        llm_model=LLM_START_MODEL,
        no_think=NO_THINK,
        orpheus_model=TTS_ORPHEUS_MODEL,
    )
    app.state.Upsampler = UpsampleOverlap()
    app.state.AudioInputProcessor = AudioInputProcessor(
        "en",
        is_orpheus=TTS_START_ENGINE == "orpheus",
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000,
    )
    await start_ari_bridge(app)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
