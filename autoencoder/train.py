from torch import nn, device, save
from torch.optim import Adam, Adamax
from autoencoder import ConvEncoder, ConvDecoder
from torch.utils.data import DataLoader
import torch
import os

# from torch.utils.tensorboard import SummaryWriter
from dataset import CifarDataset
import mlflow


class AutoEncoderTrainer:
    def __init__(
        self,
        epochs: int = 75,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        output_name: str = "model",
    ):
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.device = device("cuda:0")
        self.encoder = ConvEncoder().to(self.device)
        self.decoder = ConvDecoder().to(self.device)
        self.output_path = f"{output_name}.pth"
        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_loader = DataLoader(
            CifarDataset("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

        self.test_loader = DataLoader(
            CifarDataset("test"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def train(self):
        with mlflow.start_run():
            mlflow.log_param("Learning Rate", self.learning_rate)
            mlflow.log_param("Optimizer", "ADAM")
            mlflow.log_param("Batch Size", self.batch_size)
            mlflow.log_param("NumEpochs", self.epochs)
            for epoch in range(self.epochs):
                running_loss = 0.0
                self.encoder.train()
                self.decoder.train()
                for i, (inputs, targets, paths) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    targets = targets.float().to(self.device)
                    inputs = inputs.float().to(self.device)
                    encoder_output = self.encoder(inputs)
                    decoder_output = self.decoder(encoder_output)
                    loss = self.criterion(decoder_output, targets)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(self.train_loader)
                mlflow.log_metric("TrainingLossMSE", epoch_loss)
                mlflow.log_metric("TrainingEpoch", epoch)
                print(f"Training - Epoch: {epoch}\tLoss: {epoch_loss}")
                if epoch % 5 == 0:
                    self.run_validation(epoch)
                    print(f"saving model epoch: {epoch}")
                    save(
                        trainer.encoder.state_dict(),
                        f"savestate/encoder_epoch{epoch}_{self.output_path}",
                    )
                    save(
                        trainer.decoder.state_dict(),
                        f"savestate/decoder_epoch{epoch}_{self.output_path}",
                    )

    def run_validation(self, epoch: int):
        self.encoder.eval()
        self.decoder.eval()
        running_loss = 0.0
        for i, (inputs, targets, paths) in enumerate(self.test_loader):
            with torch.no_grad():
                targets = targets.float().to(self.device)
                inputs = inputs.float().to(self.device)
                encoder_output = self.encoder(inputs)
                decoder_output = self.decoder(encoder_output)
                loss = self.criterion(decoder_output, targets)
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(self.train_loader)
        mlflow.log_metric("TestLossMSE", epoch_loss)
        mlflow.log_metric("TestEpoch", epoch)
        print(f"Validation - Epoch: {epoch}\tLoss: {epoch_loss}")


if __name__ == "__main__":
    os.makedirs("savestate", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    trainer = AutoEncoderTrainer(
        epochs=100, batch_size=128, learning_rate=1e-4, output_name="model"
    )
    trainer.train()
    save(trainer.encoder.state_dict(), f"models/encoderfinal_{trainer.output_path}")
    save(trainer.decoder.state_dict(), f"models/decoderfinal_{trainer.output_path}")
