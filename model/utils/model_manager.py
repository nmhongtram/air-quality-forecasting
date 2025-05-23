import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


class ModelManager:
    def __init__(self, model, train_loader, val_loader=None, lr=0.001, patience=100):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    def train(self, num_epochs, save_dir='.'):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'best-{self.model.__class__.__name__}.pth')

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()
            total_train_loss = 0

            for inputs, targets in self.train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_train_loss / len(self.train_loader)
            val_loss = self.evaluate(self.val_loader) if self.val_loader is not None else 0.0

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            if self.early_stopping(val_loss, save_path):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'time: {int(time.time() - start_time)}s, '
                  f'loss: {avg_train_loss:.4f}, '
                  f'val_loss: {val_loss:.4f}')

        # Load lại best model
        self.load_model(save_path)

        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        plt.grid(True)
        # plot_path = os.path.join(save_dir, f'{self.model.__class__.__name__}_loss_plot.png')
        # plt.savefig(plot_path)
        plt.show()
        plt.close()


    def evaluate(self, loader):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        return avg_loss


    def calc_metrics(self, loader):
        self.model.eval()
        total_loss = 0

        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                all_targets.append(targets.cpu())
                all_outputs.append(outputs.cpu())

        # Tính MAE trung bình (criterion)
        avg_mae_criterion = total_loss / len(loader)

        # Gộp các batch lại
        all_targets = torch.cat(all_targets, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)

        # Metrics trên dữ liệu log1p
        actual_mae = torch.mean(torch.abs(all_outputs - all_targets)).item()
        rmse = torch.sqrt(torch.mean((all_outputs - all_targets) ** 2)).item()

        ss_res = torch.sum((all_targets - all_outputs) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = (1 - ss_res / ss_tot).item() if ss_tot != 0 else float('nan')

        # -----------------------------------------------
        # Inverse log1p
        all_targets_inv = torch.expm1(all_targets)
        all_outputs_inv = torch.expm1(all_outputs)

        # Metrics sau inverse log1p (trong đơn vị gốc)
        mae_inv = torch.mean(torch.abs(all_outputs_inv - all_targets_inv)).item()
        rmse_inv = torch.sqrt(torch.mean((all_outputs_inv - all_targets_inv) ** 2)).item()


        ss_res_inv = torch.sum((all_targets_inv - all_outputs_inv) ** 2)
        ss_tot_inv = torch.sum((all_targets_inv - torch.mean(all_targets_inv)) ** 2)
        r2_inv = (1 - ss_res_inv / ss_tot_inv).item() if ss_tot_inv != 0 else float('nan')

        # -----------------------------------------------

        return {
            # Metric trong không gian log1p (dành cho so sánh huấn luyện)
            'MAE (log1p)': actual_mae,
            'RMSE (log1p)': rmse,
            'R2 (log1p)': r2,

            # Metric sau inverse log1p
            'MAE (original)': mae_inv,
            'RMSE (original)': rmse_inv,
            'R2 (original)': r2_inv
        }


    def early_stopping(self, val_loss, save_path):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model(save_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Model loaded from {load_path}')

    def predict(self, input_data):
        self.model.eval()   # Set the model to evaluation mode

        if isinstance(input_data, DataLoader):
            # If input_data is a DataLoader, iterate through batches and concatenate predictions
            predictions = []
            with torch.no_grad():
                for inputs, _ in input_data:
                    outputs = self.model(inputs)
                    predictions.append(outputs)
            predictions = torch.cat(predictions, dim=0)
        else:
            # Assume input_data is a single input tensor
            with torch.no_grad():
                predictions = self.model(input_data.unsqueeze(0))

        return predictions

    def plot(self, y, yhat, feature_names=None, save_dir='.', save_plots=True, num_elements=None):
        if feature_names is None:
            feature_names = [f'Feature {i + 1}' for i in range(y.shape[2])]

        if num_elements is not None:
            y = y[:num_elements]
            yhat = yhat[:num_elements]

        for feature_index, feature_name in enumerate(feature_names):
            plt.figure(figsize=(10, 5))
            plt.plot(y[:, :, feature_index].flatten(), label='y', linestyle='-', linewidth=0.7)
            plt.plot(yhat[:, :, feature_index].flatten(), label='y_hat', linestyle='--', linewidth=1)
            plt.title(feature_name)
            plt.xlabel('Time Step')
            plt.ylabel('Values')
            plt.legend()

            if save_plots:
                # Create the save directory if it doesn’t exist
                os.makedirs(os.path.join(save_dir, self.model.__class__.__name__), exist_ok=True)
                # Save the plot
                save_path = os.path.join(save_dir, self.model.__class__.__name__, f'{feature_name}.png')
                plt.savefig(save_path)

            plt.show()
            plt.close()  # Close the plot to avoid overlapping in saved images
