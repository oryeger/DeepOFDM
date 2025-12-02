import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .tdcnn_detector import TDCNNDetector
from python_code import DEVICE, conf

class TDCNNTrainer:
    def __init__(self, lr=1e-3):
        self.model = TDCNNDetector().to(DEVICE)
        self.criterion = torch.nn.MSELoss()  # Change to suitable loss for your task
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, x_train, y_train, batch_size=32, epochs=10):
        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = self.criterion(output, yb)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, x_test, y_test, batch_size=32):
        dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = self.model(xb)
                loss = self.criterion(output, yb)
                total_loss += loss.item() * xb.size(0)
        return total_loss / len(dataset)
