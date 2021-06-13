from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from imitation.dataset import StateDataset
from imitation.model import ImitationModel
from utils import seed_everything

SEED = 7
BATCH_SIZE = 2048
LR = 0.001
N_EPOCHS = 100


def train(mode: str, data_path: str):
    seed_everything(None, SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = StateDataset(data_path, mode, device)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=0)

    state, action = dataset[0]
    model = ImitationModel(state.shape[0], 1)

    optimizer = AdamW(model.parameters(), lr=LR)

    mse_loss = torch.nn.MSELoss().to(device)

    for _ in tqdm(range(N_EPOCHS), desc="Epoch: ", total=N_EPOCHS):
        batch_bar = tqdm(dataloader, desc="Batch: ")
        for batch_x, batch_y in batch_bar:
            pred_y = model(batch_x).squeeze(1)
            loss = mse_loss(pred_y, batch_y)
            batch_bar.set_description(f"Batch (loss: {loss.item()}):")

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()

    model.save(f"{mode}.pkl")


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-m", "--mode", help="who to train", choices=["predators", "preys"], required=True)
    __arg_parser.add_argument("-d", "--dataset", help="path to dataset", required=True)
    __args = __arg_parser.parse_args()
    train(__args.mode, __args.dataset)
