import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import hparams
from torch.utils.tensorboard import SummaryWriter
from data import TrainData
from net import CNN


class TrainNetwork():
    def __init__(self):
        self.tdata = TrainData(
            directory='data', transforms=self.transforms(), batch_size=hparams.BATCH_SIZE)
        self.net = CNN().to(device=self.device())
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=hparams.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, factor=0.1, patience=3, verbose=True)

        # Tensorboard setup
        self.writer = SummaryWriter()

        self.train_network()

        # Finish writing operations
        self.writer.flush()

        # Check net accuracy
        self.check_net_accuracy('train')
        self.check_net_accuracy('test')

        # Save model
        self.save_model()

    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transforms(self):
        return transforms.Compose([
            transforms.ToTensor()
        ])

    def train_network(self):
        # Train mode
        self.net.train()

        global_step = 0

        for epoch in range(hparams.NUM_EPOCHS):
            losses = []

            for image, label in self.tdata.train_data():
                image = image.to(device=self.device())
                label = label.to(device=self.device())

                output = self.net(image)
                loss = self.loss(output, label)

                losses.append(loss.item())

                # Record loss to tensorboard
                self.writer.add_scalar("Loss/train", loss, global_step)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                global_step += 1

            mean_loss = sum(losses)/len(losses)
            self.scheduler.step(mean_loss)

            print(
                f'Cost at epoch {epoch} (/{hparams.NUM_EPOCHS}) is {mean_loss}')

        # Switch out of train mode
        self.net.eval()

    def check_net_accuracy(self, type):
        datatc = None
        if type == 'train':
            datatc = self.tdata.train_data()
        elif type == 'test':
            datatc = self.tdata.test_data()
        else:
            print('Invalid network data label')
            return

        num_correct = 0
        num_samples = 0

        self.net.eval()

        with torch.no_grad():
            for image, label in datatc:
                image = image.to(device=self.device())
                label = label.to(device=self.device())

                output = self.net(image)

                # Get position of most likely label (indices gets the position)
                _, predc = output.max(1)
                num_correct += (predc == label).sum()
                num_samples += predc.size(0)

            print(
                f'Got {num_correct} / {num_samples} with an accuracy of {float(num_correct) / float(num_samples)*100:.2f}')
            self.net.train()

    def save_model(self):
        FILE = "model.pth"
        torch.save(self.net.state_dict(), FILE)


if __name__ == "__main__":
    t = TrainNetwork()
