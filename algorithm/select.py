import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from algorithm.NeuralNetworks import MLPDropoutReg

class WeightedDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = X
        self.y = y
        self.weights = weights

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

def select_pseudo_labels(trained_model, input_size, X_u, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model.eval()
    with torch.no_grad():
        X_u_tensor = torch.Tensor(X_u).to(device)
        pseudo_labels = trained_model(X_u_tensor).cpu().numpy().squeeze()


    default_model = MLPDropoutReg.getDefaultModel(n_input=input_size, n_output=1)
    default_model.to(device)
    default_model.eval()
    with torch.no_grad():
        default_model_predictions = default_model(X_u_tensor).cpu().numpy().squeeze()

    u_i = np.abs(pseudo_labels - default_model_predictions)

    # parameter analysis
    top_indices = np.argsort(u_i)[:256]
    selected_u_i = u_i[top_indices]
    selected_X_u = X_u[top_indices]
    selected_pseudo_labels = pseudo_labels[top_indices]

    weights = selected_u_i / np.sum(selected_u_i)

    weighted_dataset = WeightedDataset(selected_X_u, selected_pseudo_labels, weights)
    weighted_dataLoader = DataLoader(weighted_dataset, batch_size=batch_size, shuffle=True)

    return weighted_dataLoader
