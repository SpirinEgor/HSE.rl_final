from argparse import ArgumentParser
from os import cpu_count

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from imitation.dataset import StateDataset
from imitation.model import ImitationModel
from utils import seed_everything

SEED = 7
BATCH_SIZE = 2048
LR = 0.001
LR_GAMMA = 0.95
N_EPOCHS = 20

N_LAYERS = 3
HIDDEN_DIM = 512


def train(mode: str, data_path: str):
    seed_everything(None, SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = StateDataset(data_path, mode)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=cpu_count())

    state, action = dataset[0]
    model = ImitationModel(state.shape[0], 1, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(device)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data)

    optimizer = AdamW(model.parameters(), lr=LR)

    mse_loss = torch.nn.MSELoss().to(device)

    for _ in tqdm(range(N_EPOCHS), desc="Epoch: ", total=N_EPOCHS):
        batch_bar = tqdm(dataloader, desc="Batch: ")
        for batch_x, batch_y in batch_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_y = model(batch_x).squeeze(1)
            loss = mse_loss(pred_y, batch_y)
            batch_bar.set_description(f"Batch (loss: {loss.item()}):")

            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()
            model.zero_grad()
        for g in optimizer.param_groups:
            g['lr'] *= LR_GAMMA
        model.save(f"{mode}.pkl")


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-m", "--mode", help="who to train", choices=["predators", "preys"], required=True)
    __arg_parser.add_argument("-d", "--dataset", help="path to dataset", required=True)
    __args = __arg_parser.parse_args()
    train(__args.mode, __args.dataset)
