import torch; import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import optuna
import pandas as pd; import numpy as np

from Mneme import BlockReader
from Mneme.preprocessing import (ParMinMaxScaler, ParMaxAbsScaler, ParStandardScaler,
                                 ParallelPipeline, ParLabelEncoder)

def train_loop(model, optimizer, criterion, train_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            if X_batch.dim() == 1:
                X_batch = X_batch.unsqueeze(0)
                y_batch = y_batch.unsqueeze(0)
                
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


def eval_loop(model, optimizer, criterion, eval_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in eval_loader:
            if X_batch.dim() == 1:
                X_batch = X_batch.unsqueeze(0)
                y_batch = y_batch.unsqueeze(0)

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return correct / total if total > 0 else 0


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                            nn.Dropout(0.2), nn.Linear(hidden_dim, num_classes))
    
    def forward(self, x):
        return self.model(x)


class CSVDataset(IterableDataset):
    def __init__(self, path, pipeline, chunksize = 5000):
        self.path, self.pipeline, self.chunksize = path, pipeline, chunksize
    def __iter__(self):
        for chunk in pd.read_csv(self.path, chunksize = self.chunksize):
            data = self.pipeline.transform(chunk)
            X, y = data[:, :-1], data[:, -1]
            for xi, yi in zip(X, y):
                yield torch.tensor(xi, dtype = torch.float32), torch.tensor(yi, dtype = torch.long)


def build_pipelines(filepath, num_idxs, cat_idxs, block_offset_cache):
    block_reader = BlockReader(filepath, block_offset_cache = block_offset_cache)
    pipelines = [ParallelPipeline({"InputFeatures": [sc(num_idxs = num_idxs)], 
                                   "TargetVar": [ParLabelEncoder(cat_idxs = cat_idxs)]},
            filepath)
        for sc in [ParMinMaxScaler, ParMaxAbsScaler]]
    
    for p in pipelines:
        p.fit(block_reader = block_reader, num_workers = 64, IO_workers = 2)
    
    return pipelines


def train_loop(model, optimizer, criterion, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for X_batch, y_batch in train_loader:
        if X_batch.dim() == 1:
            X_batch = X_batch.unsqueeze(0)
            y_batch = y_batch.unsqueeze(0)
        
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()


def eval_loop(model, eval_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in eval_loader:
            if X_batch.dim() == 1:
                X_batch = X_batch.unsqueeze(0)
                y_batch = y_batch.unsqueeze(0)

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total if total > 0 else 0


def train_eval_loop(model, optimizer, criterion, train_loader, eval_loader, epochs):
    best_acc = 0
    for epoch in range(epochs):
        train_loop(model, optimizer, criterion, train_loader)
        acc = eval_loop(model, eval_loader)
        if acc > best_acc:
            best_acc = acc
    return best_acc


def objective(trial, train_path, eval_path, input_dim, num_classes, pipelines, epochs):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    pipeline = pipelines[trial.suggest_categorical("pipeline_idx", [0, 1])]

    model = MLP(input_dim, hidden_dim, num_classes).to(torch.device("cuda"))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_dataset = CSVDataset(train_path, pipeline)
    eval_dataset = CSVDataset(eval_path, pipeline)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    accuracy = train_eval_loop(model, optimizer, criterion, train_loader, eval_loader, epochs)
    return accuracy


if __name__ == "__main__":
    datapath_train = "/path/to/train.csv"; datapath_eval = "/path/to/eval.csv"; block_offset_cache = "/path/to/block_offset_cache.dat"
    num_idxs = [f"x{i}" for i in range(700)]; cat_idxs = ["y0"] 
    input_dim = 700; num_classes = 4; epochs = 5; n_trials = 200

    pipelines = build_pipelines(datapath_train, num_idxs, cat_idxs, block_offset_cache)
    study = optuna.create_study(direction = "maximize")
 
    func = lambda trial: objective(trial, datapath_train, datapath_eval, 
                                   input_dim, num_classes, pipelines, epochs)
    study.optimize(func, n_trials = n_trials)