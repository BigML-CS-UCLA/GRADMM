import random
from typing import Any, List
from collections import defaultdict

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split


class TextDataset:
    def __init__(
        self, device, dataset, split, n_inputs, batch_size, n_fewshot=1, seed=42
    ):
        seq_keys = {
            'sst2': 'sentence',
            'rotten_tomatoes': 'text',
            'TwitterEmotion': 'text',
            'imdb': 'text',
            'rtpolarity': 'inputs',
        }
        seq_key = seq_keys[dataset]

        if dataset in ['sst2']:
            full = load_dataset('glue', dataset)[split]
        elif dataset == 'TwitterEmotion':
            full = load_dataset('dair-ai/emotion', 'split')[split].filter(
                lambda x: x['label'] in [0, 1]
            )
        elif dataset == 'imdb':
            full = load_dataset(
                'json',
                data_files=f'../data/imdb/{split}_len256.jsonl',
            )['train']
        elif dataset == 'rtpolarity':
            full = load_dataset(
                'json',
                data_files=f'../data/rtpolarity/{split}.jsonl',
            )['train']
        else:
            full = load_dataset(dataset)[split]
        
        n_all_samples = len(full)
        list_labels = [full[idx]['label'] for idx in range(n_all_samples)]
        idxs = list(range(n_all_samples))

        n_samples = n_inputs * batch_size + n_fewshot
        _, idxs = train_test_split(
            idxs,
            stratify=list_labels,
            test_size=n_samples,
            random_state=seed,
        )

        # Slice
        self.seqs = []
        self.labels = []
        for i in range(n_inputs):
            seqs = []
            for j in range(batch_size):
                if isinstance(seq_key, list):
                    seqs.append([full[idxs[i * batch_size + j]][key] for key in seq_key])
                else:
                    seqs.append(full[idxs[i * batch_size + j]][seq_key])
            
            labels = torch.tensor(
                [full[idxs[i * batch_size : (i + 1) * batch_size]]['label']],
                device=device,
            )
            self.seqs.append(seqs)
            self.labels.append(labels)

        for i in range(n_fewshot):
            idx_fewshot = n_inputs * batch_size + i
            if isinstance(seq_key, list):
                self.seqs.append(
                    [full[idxs[idx_fewshot : idx_fewshot + 1]][key] for key in seq_key]
                )
            else:
                self.seqs.append(full[idxs[idx_fewshot : idx_fewshot + 1]][seq_key])
            self.labels.append(
                torch.tensor([full[idxs[idx_fewshot]]['label']], device=device)
            )

        assert len(self.seqs) == n_inputs + n_fewshot
        assert len(self.labels) == n_inputs + n_fewshot

    def __getitem__(self, idx):
        return (self.seqs[idx], self.labels[idx])


class BatchDatasetLoader:
    """Batch dataset loader with options for shuffling and drop_last."""

    def __init__(
        self,
        seqs: List[Any],
        labels: List[Any],
        batch_size: int,
        drop_last: bool = True,
    ):
        self.seqs = seqs
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = len(self.labels)
        self._shuffle_data()
        print(
            f'Num samples: {self.length}, batch size: {self.batch_size}, drop_last:'
            f' {self.drop_last}'
        )

    def _shuffle_data(self):
        combined = list(zip(self.seqs, self.labels))
        random.shuffle(combined)
        self.seqs, self.labels = zip(*combined)
        self.index = 0  # Reset index after shuffling

    def _sort_data(self):
        combined = list(zip(self.seqs, self.labels))
        combined.sort(key=lambda x: len(x[0]))
        self.seqs, self.labels = zip(*combined)
        self.index = 0  # Reset index after sorting

    def __len__(self):
        if self.drop_last:
            return self.length // self.batch_size
        else:
            return (self.length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self._shuffle_data()  # Shuffle at the beginning of each new epoch
        return self

    def __next__(self):
        if self.index >= self.length:
            self._shuffle_data()  # Reshuffle if all samples have been used
            self.index = 0

        start, end = self.index, self.index + self.batch_size
        if end > self.length:
            if self.drop_last:
                self._shuffle_data()  # Reshuffle and reset index if drop_last
                self.index = 0
            end = self.length

        inputs = self.seqs[start:end]
        outputs = self.labels[start:end]

        if len(inputs) < self.batch_size and self.drop_last:
            self._shuffle_data()
            self.index = 0

        self.index += self.batch_size
        return inputs, outputs


class ClusterDatasetLoader:
  """Cluster dataset loader with options for shuffling."""

  def __init__(
      self,
      seqs: List[Any],
      labels: List[Any],
      cluster_labels: List[int],
  ):
    self.seqs = seqs
    self.labels = labels
    self.length = len(self.labels)
    self.label_to_idx = defaultdict(list)
    for i, label in enumerate(cluster_labels):
      self.label_to_idx[label].append(i)
    self.list_labels = list(self.label_to_idx.keys())
    self.num_clusters = len(self.list_labels)
    self._shuffle_data()
    print(f'Num samples: {self.length}, num clusters: {self.num_clusters}')

  def __len__(self):
    return self.num_clusters

  def _shuffle_data(self):
    random.shuffle(self.list_labels)
    self.index = 0  # Reset index after shuffling

  def __iter__(self):
    self._shuffle_data()  # Shuffle at the beginning of each new epoch
    return self

  def __next__(self):
    if self.index >= self.length:
      self._shuffle_data()  # Reshuffle if all samples have been used
      self.index = 0

    list_idx = self.label_to_idx[self.list_labels[self.index]]
    print(f'Cluster {self.list_labels[self.index]} has {len(list_idx)} samples')

    inputs = [self.seqs[idx] for idx in list_idx]
    outputs = [self.labels[idx] for idx in list_idx]

    self.index += 1
    return inputs, outputs
