import falcon.asgi
import falcon.uri

import tts_worker_process
import asr_worker_process
from communication import AsrRequest, AsrResponse, TtsRequest, TtsResponse
from config import (
    AsrConfig,
    PhoneticsConfig,
    load_config_from_file,
    save_config_to_file,
    TtsConfig,
)

from asyncio import Lock
from aioprocessing import AioPipe, AioProcess
from typing import Dict, AsyncGenerator
import fonetika.soundex
import json
import dataclasses


CONFIG_FILENAME = "config.json"
RESPONSE_CONTENT_TYPE = "audio/ogg"
RESPONSE_FILENAME = "result.ogg"


class TokenCheckMiddleware:
    def __init__(self, token: str):
        self.token = token

    async def process_request(self, req: falcon.asgi.Request, resp: falcon.asgi.Response):
        token = req.get_header("Authorization", required=True)

        if token != self.token:
            raise falcon.HTTPUnauthorized()


class TtsProcessor:
    def __init__(self, language: str, model_dir: str, venv: str):
        self.language = language

        (conn1, conn2) = AioPipe()

        self.process = AioProcess(
            target=tts_worker_process.process, args=(conn2, language, model_dir, venv)
        )
        self.connection = conn1

        self.lock = Lock()

        self.process.start()

    def __del__(self):
        self.connection.send(None)
        self.process.join()

    async def process_request(self, request: TtsRequest) -> AsyncGenerator[bytes, None]:
        async with self.lock:
            yield b""
            await self.connection.coro_send(request)
            response: TtsResponse = await self.connection.coro_recv()
            yield response.audio_data


class TtsResource:
    languages: Dict[str, TtsProcessor] = {}

    def __init__(self, config: TtsConfig):
        for language, language_config in config.languages.items():
            self.languages[language] = TtsProcessor(
                language, language_config.model_path, config.venv
            )

    async def on_get(
        self, req: falcon.asgi.Request, resp: falcon.asgi.Response, language: str
    ):
        text: str = req.get_param("text", required=True)

        if language not in self.languages:
            raise falcon.HTTPInvalidParam("language not found", "language")

        processor = self.languages[language]

        request = TtsRequest(text)

        resp.content_type = RESPONSE_CONTENT_TYPE
        resp.downloadable_as = RESPONSE_FILENAME
        resp.stream = processor.process_request(request)


class AsrProcessor:
    def __init__(self, language: str, venv: str):
        self.language = language

        (conn1, conn2) = AioPipe()

        self.process = AioProcess(
            target=asr_worker_process.process, args=(conn2, language, venv)
        )
        self.connection = conn1

        self.lock = Lock()

        self.process.start()

    def __del__(self):
        self.connection.send(None)
        self.process.join()

    async def process_request(self, request: AsrRequest) -> str:
        async with self.lock:
            await self.connection.coro_send(request)
            response: AsrResponse = await self.connection.coro_recv()
            return json.dumps(dataclasses.asdict(response))


class AsrResource:
    languages: Dict[str, AsrProcessor]
    max_audio_size = (
        asr_worker_process.MAX_PACKET_COUNT * asr_worker_process.MAX_PACKET_SIZE
    )

    def __init__(self, config: AsrConfig):
        self.languages = {}

        for language in config.languages:
            self.languages[language] = AsrProcessor(language, config.venv)

    async def on_post(
        self, req: falcon.asgi.Request, resp: falcon.asgi.Response, language: str
    ):
        if language not in self.languages:
            raise falcon.HTTPInvalidParam("language not found", "language")

        processor = self.languages[language]

        if req.content_length >= self.max_audio_size:
            raise falcon.HTTPPayloadTooLarge()

        if req.content_type != "application/extcord-opus-packets":
            raise falcon.HTTPUnsupportedMediaType()

        audio_data = await req.stream.read(self.max_audio_size)

        if len(audio_data) == self.max_audio_size:
            raise falcon.HTTPPayloadTooLarge()

        request = AsrRequest(audio_data)

        resp.content_type = "application/json"
        resp.text = await processor.process_request(request)


class PhoneticsResource:
    languages: Dict[str, fonetika.soundex.Soundex]

    def __init__(self, config: PhoneticsConfig):
        self.languages = {}

        for language in config.languages:
            soundex: fonetika.soundex.Soundex

            if language == "fin":
                soundex = fonetika.soundex.FinnishSoundex()
            elif language == "eng":
                soundex = fonetika.soundex.EnglishSoundex()
            else:
                raise Exception("Unsupported phonetics language")

            self.languages[language] = soundex

    async def on_get(
        self, req: falcon.asgi.Request, resp: falcon.asgi.Response, language: str
    ):
        if language not in self.languages:
            raise falcon.HTTPInvalidParam("language not found", "language")

        soundex = self.languages[language]

        text: str = req.get_param("text", required=True)
        text = text.strip()

        resp.content_type = "text/plain"
        resp.text = "" if text == "" else soundex.transform(text)[1:]


cfg = load_config_from_file(CONFIG_FILENAME)
save_config_to_file(CONFIG_FILENAME, cfg)

app = falcon.asgi.App()
app.add_middleware(TokenCheckMiddleware(cfg.token))
app.add_route("/tts/{language}", TtsResource(cfg.tts))
app.add_route("/asr/{language}", AsrResource(cfg.asr))
app.add_route("/phonetics/{language}", PhoneticsResource(cfg.phonetics))
