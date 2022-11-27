import torch.nn.functional as F
from ..alignment.alignment import create_alignment
import torch
from torch import nn
import numpy as np


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.model_config = model_config
        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.variance_predictor_filter_size
        self.kernel = model_config.variance_predictor_kernel_size
        self.conv_output_size = model_config.variance_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulatorlog(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulatorlog, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        log_duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, log_duration_predictor_output
        else:
            duration_predictor_output = torch.clamp(
                (torch.round(torch.exp(log_duration_predictor_output) - 1) * alpha),
                min=0,
            )
            output = self.LR(x, duration_predictor_output)

            mel_pos = torch.stack(
                [torch.Tensor([i + 1 for i in range(output.size(1))])]).long().to(x.device)
        return output, mel_pos


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.length_regulator = LengthRegulatorlog(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.pitch_bins = nn.Parameter(
            torch.exp(
                torch.linspace(np.log(model_config.pitch_min), np.log(model_config.pitch_max), model_config.n_bins - 1)
            ),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(model_config.energy_min, model_config.energy_max, model_config.n_bins - 1),
            requires_grad=False,
        )
        self.pitch_embedding = nn.Embedding(
            model_config.n_bins, model_config.encoder_dim
        )
        self.energy_embedding = nn.Embedding(
            model_config.n_bins, model_config.encoder_dim
        )

    def forward(self, x,
                duration_target=None, pitch_target=None, energy_target=None,
                alpha_duration=1.0, alpha_pitch=1.0, alpha_energy=1.0,
                mel_max_length=None, mask=None):

        x, log_duration_prediction = self.length_regulator(x, alpha_duration, duration_target, mel_max_length)
        pitch_prediction = self.pitch_predictor(x)
        if pitch_target is not None:
            emb_pitch = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))
            pitch_prediction.masked_fill(mask, 0.0)
        else:
            pitch_prediction *= alpha_pitch
            emb_pitch = self.pitch_embedding(torch.bucketize(pitch_prediction, self.pitch_bins))
        x = x + emb_pitch

        energy_prediction = self.energy_predictor(x)
        if energy_target is not None:
            emb_energy = self.energy_embedding(torch.bucketize(energy_target, self.energy_bins))
            energy_prediction.masked_fill(mask, 0.0)
        else:
            energy_prediction *= alpha_energy
            emb_energy = self.energy_embedding(torch.bucketize(energy_prediction, self.energy_bins))
        x = x + emb_energy

        return x, log_duration_prediction, pitch_prediction, energy_prediction


