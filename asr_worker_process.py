from multiprocessing.connection import Connection
from communication import AsrRequest, AsrResponse

from typing import List, Optional
import sys
import os
import traceback
import re


MAX_PACKET_COUNT = 1024
MAX_PACKET_SIZE = 7664


def decode_extcord_packets(data: bytes) -> Optional[List[bytes]]:
    import struct
    import io

    if len(data) < 5:
        print(f"too small packets, length is {len(data)}")

    (packet_count,) = struct.unpack("<H", data[:2])

    if packet_count > MAX_PACKET_COUNT:
        print("too many packets, cannot decode")
        return None

    stream = io.BytesIO(data)
    stream.seek(2)

    packets = []

    for _ in range(packet_count):
        packet_size_bytes = stream.read(2)
        if len(packet_size_bytes) != 2:
            print("unexpected end of data, cannot decode")
            return None

        (packet_size,) = struct.unpack("<H", packet_size_bytes)

        if packet_size > MAX_PACKET_SIZE:
            print("too large packet, cannot decode")
            return None

        packet = stream.read(packet_size)

        if len(packet) != packet_size:
            print("unexpected end of data, cannot decode")
            return None

        packets.append(packet)

    return packets


def process(
    pipe: Connection,
    language: str,
    model_dir_or_id: str,
    accurate_model_dir_or_id: str,
    venv: str,
):
    print(f"{language} Starting ASR worker process...")

    sys.path.append(venv)
    sys.path.append(os.path.join(venv, "Lib", "site-packages"))
    sys.path.append(os.path.join(venv, "lib", "python3.8", "site-packages"))

    from faster_whisper import WhisperModel
    import pyogg
    import numpy as np
    from fonetika.soundex import FinnishSoundex, EnglishSoundex

    # from fonetika.distance import PhoneticsInnerLanguageDistance

    model = WhisperModel(model_dir_or_id, device="cpu", compute_type="int8")
    accurate_model = WhisperModel(
        accurate_model_dir_or_id, device="cpu", compute_type="int8"
    )

    print(f"{language} Waiting for ASR requests")

    soundex = None
    # phon_distance = None

    if language == "fin":
        soundex = FinnishSoundex()
        # phon_distance = PhoneticsInnerLanguageDistance(soundex)
    elif language == "eng":
        soundex = EnglishSoundex()
        # phon_distance = PhoneticsInnerLanguageDistance(soundex)
    else:
        raise Exception("Unsupported language")

    while True:
        req: AsrRequest = pipe.recv()

        if req is None:
            break

        try:
            packets = decode_extcord_packets(req.opus_packets)

            if packets is None:
                print(f"{language} ASR could not decode packets")
                resp = AsrResponse("ERROR")
                pipe.send(resp)
                continue

            decoder = pyogg.OpusDecoder()
            decoder.set_channels(1)
            decoder.set_sampling_frequency(16000)

            pcm_output = bytearray()

            for packet in packets:
                pcm_output += decoder.decode(memoryview(bytearray(packet)))
        except Exception as e:
            print(f"{language} ASR could not decode packets")
            traceback.print_exc(e)
            resp = AsrResponse("ERROR")
            pipe.send(resp)
            continue

        i = np.iinfo(np.int16)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max

        pcm_float = (
            np.frombuffer(pcm_output, dtype=np.int16).astype(np.float32) - offset
        ) / abs_max

        if req.accurate:
            segments, info = accurate_model.transcribe(
                pcm_float, beam_size=1, language=language[:2]
            )
        else:
            segments, info = model.transcribe(
                pcm_float, beam_size=1, language=language[:2]
            )

        transcription = " ".join(segment.text for segment in segments)
        transcription = re.sub("[^\\w ]", "", transcription)
        transcription = transcription.strip()

        # print(f"{language} Transcription: {transcription}")

        transcription_phonetic = (
            "" if transcription == "" else soundex.transform(transcription)[1:]
        )

        # words = transcription.split(" ")
        # keyword_length = len(req.keyword)

        # keyword_candidate = ""
        # keyword_candidate_finished = False

        # rest = ""

        # for word in words:
        #     if keyword_candidate_finished:
        #         if rest != "":
        #             rest += " "
        #         rest += word

        #         continue

        #     if len(keyword_candidate) + len(word) > keyword_length:
        #         length_diff = keyword_length - len(keyword_candidate)
        #         length_diff_next = len(keyword_candidate) + len(word) - keyword_length

        #         if length_diff_next <= length_diff:
        #             keyword_candidate += word
        #         else:
        #             rest += word

        #         keyword_candidate_finished = True
        #     else:
        #         keyword_candidate += word

        # print(f"{language} Keyword candidate: {keyword_candidate}")
        # print(f"{language} Rest: {rest}")

        # keyword_candidate_phonetic = (
        #     "" if keyword_candidate == "" else soundex.transform(keyword_candidate)
        # )
        # rest_phonetic = "" if rest == "" else soundex.transform(rest)

        # keyword_distance = phon_distance.distance(keyword_candidate, req.keyword)

        # print(f"{language} Keyword distance: {keyword_distance}")

        # keyword_confidence = 1.0 - keyword_distance / keyword_length

        # print(f"{language} Keyword confidence: {keyword_confidence}")

        resp = AsrResponse(
            transcription,
            transcription_phonetic,
            # keyword_candidate,
            # keyword_candidate_phonetic,
            # keyword_confidence,
            # rest,
            # rest_phonetic,
        )

        pipe.send(resp)

        # print(f"{language} ASR response sent")
