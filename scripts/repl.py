"""Start-up script for briefly exploring steps with IPython."""

import json

from transformers import AutoTokenizer

from pragma import InferenceConfig
from pragma.tokenize import decode_ngram
