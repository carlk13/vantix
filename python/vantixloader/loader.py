import torch
import numpy as np
import threading
import queue
from torch.utils.data import IterableDataset
from .vantix_core import load_images_fast


class VantixLoader(IterableDataset):
    def __init__(
        self,
        image_paths,
        width=224,
        height=224,
        batch_size=64,
        augment=False,
        shuffle=True,
        transform=None,
        queue_size=3,
    ):
        self.image_paths = np.array(image_paths)
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.transform = transform
        self.queue_size = queue_size
        self.exit_event = threading.Event()

    def _worker(self, indices, batch_queue):
        """Background thread that fills the queue."""
        try:
            for start_idx in range(0, len(indices), self.batch_size):
                if self.exit_event.is_set():
                    break

                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                batch_paths = self.image_paths[batch_indices].tolist()

                np_batch = load_images_fast(
                    batch_paths, self.width, self.height, self.augment
                )

                tensor = torch.from_numpy(np_batch)

                # Block if queue is full
                batch_queue.put(tensor)

        except Exception as e:
            batch_queue.put(e)
        finally:
            batch_queue.put(None)

    def __iter__(self):
        indices = self._get_indices()

        if self.shuffle:
            np.random.shuffle(indices)

        # Create Queue and Thread
        batch_queue = queue.Queue(maxsize=self.queue_size)
        self.exit_event.clear()

        worker_thread = threading.Thread(
            target=self._worker, args=(indices, batch_queue), daemon=True
        )
        worker_thread.start()

        # Consumer Loop
        try:
            while True:
                batch = batch_queue.get()

                if batch is None:
                    break

                if isinstance(batch, Exception):
                    raise batch

                if self.transform:
                    batch = self.transform(batch)

                yield batch

        finally:
            self.exit_event.set()  # Ensure thread dies if loop breaks early

    def _get_indices(self):
        """Handles simple indexing or DDP sharding."""
        total_len = len(self.image_paths)
        indices = np.arange(total_len)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            indices = indices[rank::world_size]

        return indices

    def __len__(self):
        # Adjust length for DDP
        length = len(self.image_paths)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            length = length // world_size

        return (length + self.batch_size - 1) // self.batch_size
