"""Frozen encoder controls using the existing ProtT5 training/evaluation loop."""
import sys
from pathlib import Path
import torch
from torch import nn
from transformers import T5EncoderModel

ROOT = Path('/sddn/yyf_work/chenzhenghang/mars-tool')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import upstream_training as upstream


class FrozenPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(
            config['model_path'], local_files_only=True, torch_dtype=torch.float16)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = nn.Sequential(nn.LayerNorm(self.encoder.config.d_model),
                                  nn.Linear(self.encoder.config.d_model, 256),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 6))
        assert not any(p.requires_grad for p in self.encoder.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            hidden = self.encoder(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask']).last_hidden_state
            mask = batch['residue_mask'].unsqueeze(-1)
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)


if __name__ == '__main__':
    upstream.Predictor = FrozenPredictor
    upstream.main()
