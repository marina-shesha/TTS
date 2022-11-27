import torch
from torch import nn
from dataclasses import dataclass
from .mask import get_mask_from_lengths
from .variance_adaptor import VarianceAdaptor
from .modules import Encoder, Decoder

@dataclass
class FastSpeech2Config:
    num_mels: int = 80

    vocab_size: int = 300
    max_seq_len: int = 3000

    encoder_dim: int = 256
    encoder_n_layer: int = 4
    encoder_head: int = 2
    encoder_conv1d_filter_size: int = 1024

    decoder_dim: int = 256
    decoder_n_layer: int = 4
    decoder_head: int = 2
    decoder_conv1d_filter_size: int = 1024

    fft_conv1d_kernel: list = (9, 1)
    fft_conv1d_padding: list = (4, 0)

    n_bins = 256
    pitch_min = -2.917079304729967
    pitch_max = 11.391254536985784
    energy_min = -1.431044578552246
    energy_max = 8.184337615966797

    variance_predictor_filter_size: int = 256
    variance_predictor_kernel_size: int = 3
    dropout: float = 0.1

    PAD: int = 0
    UNK: int = 1
    BOS: int = 2
    EOS: int = 3

    PAD_WORD: str = '<blank>'
    UNK_WORD: str = '<unk>'
    BOS_WORD: str = '<s>'
    EOS_WORD: str = '</s>'


class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config=FastSpeech2Config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(self.model_config)
        self.variance_adaptor = VarianceAdaptor(self.model_config)
        self.decoder = Decoder(self.model_config)

        self.mel_linear = nn.Linear(self.model_config.decoder_dim, self.model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None,
                length_target=None, pitch_target=None, energy_target=None,
                length_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            mask = (mel_pos == 0)
            output, log_duration_output, pitch_output, energy_output = self.variance_adaptor(
                x,  length_target, pitch_target, energy_target,
                length_alpha, pitch_alpha, energy_alpha, mel_max_length, mask)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, log_duration_output, pitch_output, energy_output
        else:
            output, mel_pos, pitch_output, energy_output = self.variance_adaptor(
                x,  length_alpha, pitch_alpha, energy_alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output


