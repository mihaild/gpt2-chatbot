import importlib
import os
from typing import Optional
import json
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)


class GPT2Model:
    def __init__(self, model_dir: str, seed: Optional[int] = None, nsamples: int = 1, batch_size: int = 1, top_k: int = 40, temperature: float = 10.0):
        gpt2_modules = {}
        modules_needed = ['model', 'sample', 'encoder']
        try:
            for module in modules_needed:
                gpt2_modules[module] = importlib.import_module(f'{module}')
        except ImportError:
            logger.warning('GPT2 modules not in path, trying gpt-2/src')
            if 'gpt-2/src' not in sys.path:
                sys.path.append('gpt-2/src')
            for module in modules_needed:
                gpt2_modules[module] = importlib.import_module(f'{module}')
            sys.path.remove('gpt-2/src')

        self.tf = importlib.import_module('tensorflow')

        # get_encoder from gpt-2 assumes model is located in models dir
        # so we need to copy paste it here
        with open(os.path.join(model_dir, 'encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(os.path.join(model_dir, 'vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self.encoder = gpt2_modules['encoder'].Encoder(
            encoder=encoder,
            bpe_merges=bpe_merges,
        )

        if batch_size is None:
            batch_size = 1
        self.batch_size = batch_size
        assert nsamples % batch_size == 0

        hparams = gpt2_modules['model'].default_hparams()
        with open(os.path.join(model_dir, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        length = 64
        top_k = 2

        config = self.tf.ConfigProto()

        self.sess = self.tf.Session(config=config)  # graph=self.graph)

        self.context = self.tf.placeholder(self.tf.int32, [batch_size, None])
        np.random.seed(seed)
        self.tf.set_random_seed(seed)
        self.output = gpt2_modules['sample'].sample_sequence(
            hparams=hparams, length=length,
            context=self.context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = self.tf.train.Saver()
        ckpt = self.tf.train.latest_checkpoint(model_dir)
        saver.restore(self.sess, ckpt)

    def generate(self, text):
        context_tokens = self.encoder.encode(text)
        logger.debug(f"Input: {len(text)} symbols, {len(context_tokens)} tokens")
        out = self.sess.run(self.output, feed_dict={
            self.context: [context_tokens for _ in range(self.batch_size)]
        })[:, len(context_tokens):]
        result = self.encoder.decode(out[0])
        logger.debug("Output: {len(result)} symbols, {len(out)} tokens")
        return result


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, text):
        logger.info(f"Input text: {text}")
        return f"Generated <>& {text}"
