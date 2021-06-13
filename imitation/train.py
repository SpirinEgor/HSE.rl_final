from argparse import ArgumentParser
from os import cpu_count

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from imitation.dataset import StateDataset
from imitation.model import ImitationModel
from predators_and_preys_env.env import PredatorsAndPreysEnv
from utils import seed_everything

SEED = 7
DATA_PATH = ""
BATCH_SIZE = 512
LR = 0.01
N_EPOCHS = 10


def train(mode: str):
    env = PredatorsAndPreysEnv()
    seed_everything(env, SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = StateDataset(DATA_PATH, mode, device)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=cpu_count())

    state, action = dataset[0]
    model = ImitationModel(state.shape[0], action.shape[0])

    optimizer = AdamW(model.parameters(), lr=LR)

    mse_loss = torch.nn.MSELoss().to(device)

    for _ in tqdm(range(N_EPOCHS), desc="Epoch: ", total=N_EPOCHS):
        batch_bar = tqdm(dataloader, desc="Batch: ")
        for batch_x, batch_y in batch_bar:
            pred_y = model(batch_x)
            loss = mse_loss(pred_y, batch_y)
            batch_bar.set_description(f"Batch (loss: f{loss.item()}):")

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()

    model.save(f"{mode}.pkl")


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-m", "--mode", help="who to train", choices=["predators", "preys"])
    __args = __arg_parser.parse_args()
    train(__args.mode)
