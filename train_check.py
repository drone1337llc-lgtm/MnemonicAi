import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

t0 = time.time()
from mnemonicai.appconfig import AppConfig
from mnemonicai.backend import TransformersPeftBackend

cfg = AppConfig.load(os.path.join(BASE_DIR, "config.json"))
cfg.backend = "transformers"
be = TransformersPeftBackend(cfg)
import torch

vram_str = "N/A"
if torch.cuda.is_available():
    vram_str = f"{torch.cuda.memory_allocated()/1e9:.1f}GB"

trainable_params = sum(p.numel() for p in be.model.parameters() if p.requires_grad) / 1e6

print(f"TRAIN_STACK_OK loaded in {time.time()-t0:.0f}s "
      f"vram={vram_str} "
      f"trainable={trainable_params:.1f}M params",
      flush=True)
