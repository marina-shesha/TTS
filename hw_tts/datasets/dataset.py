from torch.utils.data import Dataset
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
import time
import os
from hw_tts.text import text_to_sequence
from torchaudio.transforms import Spectrogram
from torchaudio.transforms import MelScale
import pyworld as pw
import os.path


class MelSpectrogramConfig:
    sr = 22050
    num_mels = 80
    mel_fmin = 0
    mel_fmax = 8000
    num_fft = 1024
    hop_length = 256
    win_length = 1024
    data_path = "./data/train.txt"
    mel_path = "./mels"
    alignment_path = "./alignments"
    wav_path = './data/LJSpeech-1.1/wavs'
    text_cleaners = ['english_cleaners']


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(config=MelSpectrogramConfig):
    buffer = list()
    text = process_text(config.data_path)
    to_spec_trans = Spectrogram(n_fft=config.num_fft, win_length=config.win_length, hop_length=config.hop_length)
    to_mel_trans = MelScale(n_mels=config.num_mels, sample_rate=config.sr,
                            f_min=config.mel_fmin, f_max=config.mel_fmax, n_stft=config.num_fft // 2 + 1)
    wavs_name = sorted(list(map(lambda x: x[2:-4], os.listdir(config.wav_path))))
    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        duration = np.load(os.path.join(
            config.alignment_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        wav, sr = torchaudio.load(os.path.join(
            config.wav_path, f"LJ{wavs_name[i]}.wav"
        ))
        wav = wav.squeeze().double()
        pitch, t = pw.dio(wav.numpy(), config.sr)
        pitch = pw.stonemask(wav.numpy(), pitch, t, config.sr)
        spectrogram = to_spec_trans(wav)
        energy = torch.norm(spectrogram, p='fro', dim=1)
        mel_target = to_mel_trans(spectrogram.float())
        #mel_target = mel_target.transpose(-1, -2)
        pitch = torch.tensor(pitch)
        buffer.append({"text": character, "duration": duration, 'pitch': pitch,
                       "energy": energy, "mel_target": mel_target})

    end = time.perf_counter()
    np.save('buffer.npy', buffer)
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class LJSpeechDataset(Dataset):
    def __init__(self):
        if os.path.exists('buffer.npy'):
            self.buffer = list(np.load('buffer.npy',  allow_pickle=True))
        else:
            self.buffer = get_data_to_buffer()
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
