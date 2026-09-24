"""Tiny optional Torch adapter; all tensors and fits use synthetic inputs."""
import numpy as np
import pandas as pd


class TorchLinear:
    def __init__(self):
        import torch
        torch.manual_seed(37)
        self.model = torch.nn.Linear(3, 1)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)

    def fit(self, context):
        import torch
        self.targets = list(context.y)
        # The adapter owns BOTH parameter placement and dataframe-to-tensor transfer.
        X = torch.as_tensor(context.X.to_numpy(dtype=np.float32), device=self.device)
        y = torch.as_tensor(context.y.to_numpy(dtype=np.float32), device=self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.0001)
        for _ in range(5):
            optimizer.zero_grad()
            loss = ((self.model(X) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
        self.fit_tensor_device = str(X.device)
        self.parameter_device = str(next(self.model.parameters()).device)

    def predict(self, context):
        import torch
        X = torch.as_tensor(context.X.to_numpy(dtype=np.float32), device=self.device)
        with torch.no_grad():
            values = self.model(X).cpu().numpy()
        self.prediction_tensor_device = str(X.device)
        return {'predict': pd.DataFrame(values, index=context.X.index, columns=self.targets)}


def move_torch_model(adapter, device):
    adapter.to(device)
