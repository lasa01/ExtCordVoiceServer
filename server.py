import falcon.asgi
import falcon.uri
import worker_process
from communication import TtsRequest, TtsResponse

from asyncio import Lock
from aioprocessing import AioPipe, AioProcess


MODEL_DIR = r"D:\lasa\Code\vits\fin"


class TtsResource:
    def __init__(self):
        (conn1, conn2) = AioPipe()

        self.process = AioProcess(
            target=worker_process.process, args=(conn2, MODEL_DIR)
        )
        self.connection = conn1

        self.lock = Lock()

        self.process.start()

    def __del__(self):
        self.connection.send(None)
        self.process.join()

    async def on_get(
        self, req: falcon.asgi.Request, resp: falcon.asgi.Response, language: str
    ):
        params = falcon.uri.parse_query_string(req.query_string)

        if "text" not in params:
            raise falcon.HTTPMissingParam("text")

        text = params["text"]

        request = TtsRequest(language, text)

        async with self.lock:
            await self.connection.coro_send(request)
            response: TtsResponse = await self.connection.coro_recv()

        resp.content_type = "audio/wav"
        resp.downloadable_as = "result.wav"
        resp.data = response.audio_data


app = falcon.asgi.App()
app.add_route("/tts/{language}", TtsResource())
