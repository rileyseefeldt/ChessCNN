#loads model and evaluates accuracy

import torch
from chess_dataset import ChessDataset
from torch.utils.data import DataLoader
from model import ChessNet

def evaluate_model(model, dataloader, device):
    model.eval()
    total_value_loss = 0.0
    total_policy_loss = 0.0
    num_batches = 0

    criterion_value = torch.nn.MSELoss()
    criterion_policy = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_batch, target in dataloader:
            x_batch = x_batch.to(device)
            policy_target = target[:, 0].long().to(device)
            value_target = target[:, 1].to(device)

            policy_pred, value_pred = model(x_batch)
            
            loss_value = criterion_value(value_pred.squeeze(), value_target)
            loss_policy = criterion_policy(policy_pred, policy_target)

            total_value_loss += loss_value.item()
            total_policy_loss += loss_policy.item()
            num_batches += 1

    avg_value_loss = total_value_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    print(f"Average Value Loss: {avg_value_loss:.4f}, Average Policy Loss: {avg_policy_loss:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChessDataset("data/chess_data.h5")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    model = ChessNet().to(device)

    model.load_state_dict(torch.load("models/chess_model_best.pth", map_location=device))

    evaluate_model(model, dataloader, device)
