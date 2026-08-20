import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd

from gamblingfinder.model import FeatureConfig, get_column_transformer

SEED = 739841


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


class BinaryNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        activation: str,
        dropout: float,
        num_hidden_layers: int,
    ):
        super().__init__()

        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(input_size, input_size))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.fc_out = nn.Linear(input_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = self.activation_fn(fc(x))
            x = dropout(x)

        x = self.fc_out(x)
        return x

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        self.eval()

        x = torch.FloatTensor(x)

        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)

        self.train()

        return probabilities.numpy()


TQDM_ENABLED = False


def maybe_tqdm(iterable):
    if TQDM_ENABLED:
        return tqdm(iterable)
    else:
        return iterable


def train_nn(
    X_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    activation: str,
    dropout: float,
    num_hidden_layers: int,
) -> tuple[BinaryNNClassifier, int]:
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.LongTensor(y_train)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    model = BinaryNNClassifier(
        input_size=input_size,
        activation=activation,
        dropout=dropout,
        num_hidden_layers=num_hidden_layers,
    )

    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for _ in maybe_tqdm(range(epochs)):
        for X_batch, y_batch in maybe_tqdm(dataloader):
            outputs = model(X_batch)
            loss = cross_entropy(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, input_size


class GamblingNN:
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        activation: str,
        dropout: float,
        num_hidden_layers: int,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers

        self.column_transformer = get_column_transformer(
            FeatureConfig(use_bag_of_words=0, only_use_embeddings=True)
        )
        self.model = None

    def train(self, X_train: pd.DataFrame, y_train):
        X_train_transformed = self.column_transformer.fit_transform(X_train)
        self.model, self.input_size = train_nn(
            X_train_transformed,
            y_train,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            activation=self.activation,
            dropout=self.dropout,
            num_hidden_layers=self.num_hidden_layers,
        )

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        X_transformed = self.column_transformer.transform(X)
        return self.model.predict(X_transformed)

    def save(self, name: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "column_transformer": self.column_transformer,
                "input_size": self.input_size,
                "activation": self.activation,
                "num_hidden_layers": self.num_hidden_layers,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
            },
            f"data/{name}.pt",
        )

    @staticmethod
    def load(name: str) -> "GamblingNN":
        checkpoint = torch.load(f"data/{name}.pt", weights_only=False)
        column_transformer = checkpoint["column_transformer"]
        input_size = checkpoint["input_size"]
        activation = checkpoint["activation"]
        num_hidden_layers = checkpoint["num_hidden_layers"]
        dropout = checkpoint["dropout"]
        batch_size = checkpoint["batch_size"]
        learning_rate = checkpoint["learning_rate"]
        epochs = checkpoint["epochs"]

        model = BinaryNNClassifier(
            input_size=input_size,
            activation=activation,
            dropout=dropout,
            num_hidden_layers=num_hidden_layers,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        gambling_nn = GamblingNN(
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            activation=activation,
            dropout=dropout,
            num_hidden_layers=num_hidden_layers,
        )
        gambling_nn.model = model
        gambling_nn.column_transformer = column_transformer
        gambling_nn.input_size = input_size
        return gambling_nn
