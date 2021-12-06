import torch
import torch.nn.functional as F
from net import CNN


class RunModel():
    def __init__(self):
        FILE = 'model.pth'
        self.net = CNN().to(self.device())
        self.net.load_state_dict(torch.load(FILE))

    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_model(self, img):
        self.net.eval()

        output = self.net(img)

        pred = output.max(1).indices
        probs = F.softmax(output, dim=1)
        conf, _ = torch.max(probs, 1)

        return (pred[0], conf[0])
