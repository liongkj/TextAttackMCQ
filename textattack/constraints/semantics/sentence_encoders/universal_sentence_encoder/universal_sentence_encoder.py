"""
universal sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import random
from multiprocessing import Process, Queue

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader

hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")
import tensorflow as tf


class UniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/3"

        self._tfhub_url = tfhub_url
        # Lazily load the model
        self.model = None
        gpus = tf.config.experimental.list_physical_devices('GPU')
        ##### Hack for tensorflow gpu not releasing memory after inference
        if gpus:
        # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')
            except RuntimeError as e:
                # Visible devices must be set at program startup
                print(e)

    def encode(self, sentences):
        if not self.model:
            self.model = hub.load(self._tfhub_url)
        encoding = self.model(sentences)
        if isinstance(encoding, dict):
            encoding = encoding["outputs"]

        return encoding.numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None
