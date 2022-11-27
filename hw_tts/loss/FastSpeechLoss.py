import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, log_duration_predicted, pitch_predicted, energy_predicted,
                mel_target, duration_predictor_target, pitch_target, energy_target):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(log_duration_predicted,
                                               torch.log(duration_predictor_target.float() + 1))
        pitch_predicted_loss = self.mse_loss(pitch_predicted, pitch_target.float())
        energy_predicted_loss = self.mse_loss(energy_predicted, energy_target.float())

        return mel_loss, duration_predictor_loss, pitch_predicted_loss, energy_predicted_loss
