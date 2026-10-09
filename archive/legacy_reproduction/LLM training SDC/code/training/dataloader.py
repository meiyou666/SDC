import gzip
import itertools
import json
import torch

from pathlib import Path
from torch.utils.data import IterableDataset, Dataset, get_worker_info


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.inputs = []
        self.labels = []

    def add_batch(self, batch):
        input_ids, labels = batch
        self.inputs.append(input_ids.cpu())
        self.labels.append(labels.cpu())

    def __len__(self):
        return sum(x.size(0) for x in self.inputs)

    def __getitem__(self, idx):
        flat_inputs = torch.cat(self.inputs, dim=0)
        flat_labels = torch.cat(self.labels, dim=0)
        return flat_inputs[idx], flat_labels[idx]

    def reset(self):
        self.inputs = []
        self.labels = []


class PreprocessedDataset(Dataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.data))
        batch = [self.data[i] for i in range(start, end)]

        tokenized_examples = [self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ) for example in batch]

        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in tokenized_examples])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in tokenized_examples])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_data = iter(self.data)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class LocalJsonlDataset(Dataset):
    """Offline dataset that reads a JSONL file (plain or gzip-compressed).

    Each line must be a JSON object containing at least a ``text`` field.
    The file is indexed once at construction time by byte offset, then
    individual lines are decoded on demand.  This avoids loading the entire
    dataset into RAM.
    """

    def __init__(self, path: str, rank: int = 0, world_size: int = 1):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.rank = rank
        self.world_size = world_size
        self._is_gz = self.path.suffix == ".gz"
        self._line_offsets = self._build_index()

        # Shard by rank/world_size.
        self._indices = list(range(rank, len(self._line_offsets), world_size))

        # Cached file handle for sequential reads (rebuilt after pickling).
        self._file = None
        self._last_idx = -1

    def _open(self):
        if self._file is not None:
            return self._file
        opener = gzip.open if self._is_gz else open
        self._file = opener(self.path, "rb")
        return self._file

    def _build_index(self):
        """Record byte offsets of every non-empty line."""
        offsets = []
        opener = gzip.open if self._is_gz else open
        with opener(self.path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def _read_line(self, idx: int) -> dict:
        absolute_idx = self._indices[idx]
        offset = self._line_offsets[absolute_idx]

        f = self._open()
        # Fast path: next requested line is the next line in the file.
        if idx != self._last_idx + 1:
            f.seek(offset)
        self._last_idx = idx

        line = f.readline()
        return json.loads(line.decode("utf-8"))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._read_line(idx)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_last_idx"] = -1
        return state


class OfflinePreprocessedDataset(Dataset):
    """Batches and tokenizes examples from a LocalJsonlDataset on the fly.

    ``repeat`` replicates the underlying data for multiple passes (index is
    taken modulo the number of batches in one pass), which allows training
    longer than one epoch when the local dataset is smaller than the target
    step count. Batch order is identical across passes.
    """

    def __init__(self, data: LocalJsonlDataset, tokenizer, batch_size: int, max_length: int, repeat: int = 1):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.repeat = max(1, int(repeat))

    def _batches_per_pass(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self._batches_per_pass() * self.repeat

    def __getitem__(self, index):
        index = index % self._batches_per_pass()
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.data))
        batch = [self.data[i] for i in range(start, end)]

        tokenized_examples = [self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ) for example in batch]

        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in tokenized_examples])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in tokenized_examples])

        return {"input_ids": input_ids, "attention_mask": attention_mask}
