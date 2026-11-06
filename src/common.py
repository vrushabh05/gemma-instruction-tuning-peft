"""Shared setup: seeds, 4-bit quantization config, Hugging Face login from an environment variable."""
import os
import random

import numpy as np
import torch
from transformers import BitsAndBytesConfig

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

HF_TOKEN = os.environ.get("HF_TOKEN")  # never hard-code tokens; set HF_TOKEN in your environment
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
