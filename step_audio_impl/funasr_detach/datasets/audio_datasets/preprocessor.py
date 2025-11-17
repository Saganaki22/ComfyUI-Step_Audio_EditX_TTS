import os
import json
import torch
import logging
import concurrent.futures
import librosa
import soundfile as sf
import torch.distributed as dist
from typing import Collection
import torch
from torch import nn
import random
import re
from funasr_detach.tokenizer.cleaner import TextCleaner
from funasr_detach.register import tables


@tables.register("preprocessor_classes", "SpeechPreprocessSpeedPerturb")
class SpeechPreprocessSpeedPerturb(nn.Module):
    def __init__(self, speed_perturb: list = None, **kwargs):
        super().__init__()
        self.speed_perturb = speed_perturb

    def forward(self, waveform, fs, **kwargs):
        if self.speed_perturb is None:
            return waveform
        speed = random.choice(self.speed_perturb)
        if speed != 1.0:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)

            # Convert to numpy for librosa processing
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.cpu().numpy()
            else:
                waveform_np = waveform

            # Resample using librosa to change speed (equivalent to speed perturbation)
            new_fs = int(fs * speed)
            waveform_np = librosa.resample(waveform_np, orig_sr=fs, target_sr=new_fs)

            # Resample back to original sample rate
            waveform_np = librosa.resample(waveform_np, orig_sr=new_fs, target_sr=fs)

            waveform = torch.from_numpy(waveform_np).float()

        return waveform


@tables.register("preprocessor_classes", "TextPreprocessSegDict")
class TextPreprocessSegDict(nn.Module):
    def __init__(
        self,
        seg_dict: str = None,
        text_cleaner: Collection[str] = None,
        split_with_space: bool = False,
        **kwargs
    ):
        super().__init__()

        self.text_cleaner = TextCleaner(text_cleaner)

    def forward(self, text, **kwargs):
        text = self.text_cleaner(text)

        return text
