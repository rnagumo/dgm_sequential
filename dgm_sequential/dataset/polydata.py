
"""Polyphonic dataset

* Data
http://www-etud.iro.umontreal.ca/~boulanni/icml2012
"""

import pathlib
import pickle

import torch


class PolyphonicDataset(torch.utils.data.Dataset):
    def __init__(self, data, total_length, note_range=88, bias=20):
        super().__init__()

        # Data size of (seq_len, batch_size, input_size)
        data, seq_len = self._preprocess(data, total_length, note_range, bias)
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, idx):
        return self.data[:, idx], self.seq_len[idx]

    def __len__(self):
        return self.data.size(1)

    @staticmethod
    def _preprocess(data, total_length, note_range, bias):
        res = []

        # For each sequence
        for seq in range(len(data)):
            seq_len = len(data[seq])
            sequence = torch.zeros((seq_len, note_range))

            # For each time step
            for t in range(seq_len):
                note_slice = torch.tensor(data[seq][t]) - bias
                slice_len = note_slice.size(0)
                if slice_len > 0:
                    # Convert index list to one-hot vector
                    sequence[t, note_slice] = torch.ones(slice_len)

            # Append to list
            res.append(sequence)

        # Pack sequences
        pack = torch.nn.utils.rnn.pack_sequence(res, enforce_sorted=False)

        # Pad packed sequences with given total length
        data, seq_len = torch.nn.utils.rnn.pad_packed_sequence(
            pack, total_length=total_length)

        return data, seq_len


def init_poly_dataloader(path, cuda=False, batch_size=20):
    # Load data from pickle file
    with pathlib.Path(path).open("rb") as f:
        data = pickle.load(f)

    # Max length of all sequences
    max_len = max(len(l) for key in data for l in data[key])

    # Kwargs for data loader
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else{}

    # Instantiate data loader
    train_loader = torch.utils.data.DataLoader(
        PolyphonicDataset(data["train"], max_len),
        batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        PolyphonicDataset(data["valid"], max_len),
        batch_size=batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        PolyphonicDataset(data["test"], max_len),
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader
