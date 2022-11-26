from torch.utils.data import DataLoader
from hw_tts.collate.collate_fn import Collator


class LJSpeechDataloader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        batch_expand_size,
        num_workers
    ):
        super().__init__(
            dataset,
            batch_size=batch_expand_size * batch_size,
            shuffle=True,
            collate_fn=Collator(batch_expand_size),
            drop_last=True,
            num_workers=num_workers,
        )
        self.batch_expand_size = batch_expand_size
