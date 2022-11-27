from torch import nn
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from hw_tts.base import BaseTrainer
from hw_tts.utils import inf_loop, MetricTracker
import random
import numpy as np
import os
import torchaudio
from text import text_to_sequence
import waveglow
import utils

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            data_loader,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader) * data_loader.batch_expand_size
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = 300

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss", "grad norm",  writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        batch_idx = 0
        for batch in tqdm(self.train_dataloader):
            for db in tqdm(batch):
                batch_idx += 1

                character = db["text"].long().to(self.device)
                mel_target = db["mel_target"].float().to(self.device)
                duration = db["duration"].int().to(self.device)
                pitch = db["pitch"].float().to(self.device)
                energy = db["energy"].float().to(self.device)
                mel_pos = db["mel_pos"].long().to(self.device)
                src_pos = db["src_pos"].long().to(self.device)
                max_mel_len = db["mel_max_len"]

                mel_output, duration_predictor_output, pitch_output, energy_output = self.model(
                    character,
                    src_pos,
                    mel_pos=mel_pos,
                    mel_max_length=max_mel_len,
                    length_target=duration,
                    pitch_target=pitch,
                    energy_target=energy
                )

                mel_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
                    mel_output,
                    duration_predictor_output,
                    pitch_output,
                    energy_output,
                    mel_target,
                    duration,
                    pitch,
                    energy
                )
                total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                self.train_metrics.update("grad norm", self.get_grad_norm())
                self.train_metrics.update("loss", total_loss.item())
                self.train_metrics.update("mel_loss", mel_loss.item())
                self.train_metrics.update("duration_loss", duration_loss.item())
                self.train_metrics.update("pitch_loss", pitch_loss.item())
                self.train_metrics.update("energy_loss", energy_loss.item())

                total_loss.backward()

                self._clip_grad_norm()

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f} mel loss: {:.6f} duration loss: {:.6f} pitch loss: {:.6f} energy loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), total_loss.item(),  mel_loss.item(), duration_loss.item(), pitch_loss.item(), energy_loss.item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()

            if batch_idx >= self.len_epoch:
                break

        log = last_train_metrics
        self.evaluation()

        return log

    def evaluation(self):
        self.model.eval()

        WaveGlow = utils.get_WaveGlow()
        WaveGlow = WaveGlow.to(self.device)

        def get_data():
            tests = [
                "I remove attention module in decoder and use average pooling to implement predicting r frames at once",
                "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted.",
                "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
            ]
            data_list = list(text_to_sequence(test, ['english_cleaners']) for test in tests)

            return data_list

        data_list = get_data()
        os.makedirs("results", exist_ok=True)

        def synthesis(model, text, alpha=1.0):
            text = np.array(text)
            text = np.stack([text])
            src_pos = np.array([i + 1 for i in range(text.shape[1])])
            src_pos = np.stack([src_pos])
            sequence = torch.from_numpy(text).long().to(self.device)
            src_pos = torch.from_numpy(src_pos).long().to(self.device)

            with torch.no_grad():
                mel = model.forward(sequence, src_pos, length_alpha=alpha)
            return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

        for speed in [0.8, 1., 1.3]:
            for i, phn in tqdm(enumerate(data_list)):
                mel, mel_cuda = synthesis(self.model, phn, speed)

                waveglow.inference.inference(
                    mel_cuda, WaveGlow,
                    f"results/s={speed}_{i}_waveglow.wav"
                )
        i = 0
        for f in os.listdir('results'):
            wav, sr = torchaudio.load(os.path.join('results', f))
            self._log_audio(i, torch.tensor(wav), sr)
            i += 1

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_audio(self, name, audio, sr):
        self.writer.add_audio(f"audio{name}", audio, sample_rate=sr)
