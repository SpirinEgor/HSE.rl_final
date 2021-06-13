import json
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import state_dict_to_array


def count_lines_in_file(file_path: str, buffer_size: int = 1024 * 1024) -> int:
    """Count the number of lines in the given file.
    :param file_path: path to the file
    :param buffer_size: size of temporary buffer during reading the file
    :return: number of lines
    """
    n_lines = 0
    with open(file_path, "rb") as file:
        file_reader = file.read
        buffer = file_reader(buffer_size)
        while buffer:
            n_lines += buffer.count(b"\n")
            buffer = file_reader(buffer_size)
    return n_lines


def get_lines_offsets(file_path: str, show_progress_bar: bool = True) -> List[int]:
    """Calculate cumulative offsets for all lines in the given file.
    :param file_path: path to the file
    :param show_progress_bar: if True then tqdm progress bar will be display
    :return: list of ints with cumulative offsets
    """
    line_offsets: List[int] = []
    cumulative_offset = 0
    with open(file_path, "r") as file:
        file_iter = tqdm(file, total=count_lines_in_file(file_path)) if show_progress_bar else file
        for line in file_iter:
            line_offsets.append(cumulative_offset)
            cumulative_offset += len(line.encode(file.encoding))
    return line_offsets


def get_line_by_offset(file_path: str, offset: int) -> str:
    """Get line by byte offset from the given file.
    :param file_path: path to the file
    :param offset: byte offset
    :return: read line
    """
    with open(file_path, "r") as data_file:
        data_file.seek(offset)
        line = data_file.readline().strip()
    return line


class StateDataset(Dataset):
    def __init__(self, data_path: str, mode: str):
        self._data_path = data_path
        self._mode = mode
        self._samples_offsets = get_lines_offsets(data_path)
        self._n_samples = len(self._samples_offsets)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_sample = json.loads(get_line_by_offset(self._data_path, self._samples_offsets[item]))
        state_dict = raw_sample["state"]
        rand_target = random.randint(0, len(state_dict[self._mode]) - 1)
        if self._mode == "predators":
            target_action = raw_sample["pred_act"][rand_target]
            target_state = state_dict_to_array(
                [state_dict["predators"][rand_target]], state_dict["preys"], state_dict["obstacles"]
            )
        elif self._mode == "preys":
            target_action = raw_sample["prey_act"][rand_target]
            target_state = state_dict_to_array(
                state_dict["predators"], [state_dict["preys"][rand_target]], state_dict["obstacles"]
            )
        else:
            raise ValueError()
        return torch.tensor(target_state), torch.tensor(target_action)

    def __len__(self):
        return self._n_samples
