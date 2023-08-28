from multiprocessing.connection import Connection
from communication import TtsRequest, TtsResponse

import sys
import os


def process(pipe: Connection, language: str, model_dir: str, venv: str):
    print(f"{language} Starting TTS worker process...")

    sys.path.append(venv)
    sys.path.append(os.path.join(venv, "Lib", "site-packages"))
    sys.path.append(os.path.join(venv, "lib", "python3.8", "site-packages"))
    sys.path.append(os.path.join(os.path.dirname(__file__), "vits"))

    import torch
    from vits import commons
    from vits import utils
    from vits.models import SynthesizerTrn
    from io import BytesIO
    import numpy as np
    import pyogg
    from num2words import num2words

    class TextMapper(object):
        def __init__(self, vocab_file):
            self.symbols = [
                x.replace("\n", "")
                for x in open(vocab_file, encoding="utf-8").readlines()
            ]
            self.SPACE_ID = self.symbols.index(" ")
            self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
            self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        def text_to_sequence(self, text, cleaner_names):
            """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
            Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through
            Returns:
            List of integers corresponding to the symbols in the text
            """
            sequence = []
            clean_text = text.strip()
            for symbol in clean_text:
                symbol_id = self._symbol_to_id[symbol]
                sequence += [symbol_id]
            return sequence

        # def uromanize(self, text, uroman_pl):
        #     iso = "xxx"
        #     with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
        #         with open(tf.name, "w") as f:
        #             f.write("\n".join([text]))
        #         cmd = f"perl " + uroman_pl
        #         cmd += f" -l {iso} "
        #         cmd += f" < {tf.name} > {tf2.name}"
        #         os.system(cmd)
        #         outtexts = []
        #         with open(tf2.name) as f:
        #             for line in f:
        #                 line = re.sub(r"\s+", " ", line).strip()
        #                 outtexts.append(line)
        #         outtext = outtexts[0]
        #     return outtext

        def get_text(self, text, hps):
            text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
            if hps.data.add_blank:
                text_norm = commons.intersperse(text_norm, 0)
            text_norm = torch.LongTensor(text_norm)
            return text_norm

        def filter_oov(self, text, lang=None):
            text = self.preprocess_char(text, lang=lang)
            val_chars = self._symbol_to_id
            txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
            # print(f"{language} Text after filtering OOV: {txt_filt}")
            return txt_filt

        def process_numbers(self, text: str, lang=None):
            if lang == None:
                return text

            words = text.split()

            return " ".join(
                num2words(word, lang=lang) if word.isdecimal() else word
                for word in words
            )

        def preprocess_char(self, text, lang=None):
            """
            Special treatement of characters in certain languages
            """
            if lang == "ron":
                text = text.replace("ț", "ţ")
                # print(f"{lang} (ț -> ţ): {text}")
            return text

    ckpt_dir = model_dir

    device = torch.device("cpu")

    print(f"{language} Run inference with {device}")
    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.to(device)
    _ = net_g.eval()

    g_pth = f"{ckpt_dir}/G_100000.pth"
    print(f"{language} Load {g_pth}")

    _ = utils.load_checkpoint(g_pth, net_g, None)

    print(f"{language} Waiting for requests")

    while True:
        req: TtsRequest = pipe.recv()

        if req is None:
            break

        txt = req.text

        # print(f"{language} Text: {txt}")

        # is_uroman = hps.data.training_files.split(".")[-1] == "uroman"
        # if is_uroman:
        #     with tempfile.TemporaryDirectory() as tmp_dir:
        #         if args.uroman_dir is None:
        #             cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
        #             print(cmd)
        #             subprocess.check_output(cmd, shell=True)
        #             args.uroman_dir = tmp_dir
        #         uroman_pl = os.path.join(args.uroman_dir, "bin", "uroman.pl")
        #         print(f"uromanize")
        #         txt = text_mapper.uromanize(txt, uroman_pl)
        #         print(f"uroman text: {txt}")

        txt = txt.lower()
        txt = text_mapper.process_numbers(txt, lang=language)
        txt = text_mapper.filter_oov(txt, lang=language)
        stn_tst = text_mapper.get_text(txt, hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            hyp = (
                net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                )[0][0, 0]
                .cpu()
                .float()
                .numpy()
            )

        buffer = BytesIO()

        encoder = pyogg.OpusBufferedEncoder()
        encoder.set_application("audio")
        encoder.set_sampling_frequency(hps.data.sampling_rate)
        encoder.set_channels(1)
        encoder.set_frame_size(20)  # milliseconds

        writer = pyogg.OggOpusWriter(buffer, encoder)

        i = np.iinfo(np.int16)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        pcm16 = (hyp.ravel() * abs_max + offset).clip(i.min, i.max).astype(np.int16)

        writer.write(pcm16.view("b").data)
        writer.close()

        resp = TtsResponse(txt, buffer.getvalue())

        pipe.send(resp)

        # print(f"{language} Response sent")
