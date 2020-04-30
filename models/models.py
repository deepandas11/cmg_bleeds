import torch
import torch.nn as nn
import torchvision.models as imagemodels


class EncoderCNN(nn.Module):
    def __init__(self, pretrained=False, remove_layers=1, base_model='resnet50', downscale=False):
        """
        Args:
            pretrained: Decide if pretrained on image-net needed
            remove_layers: Number of layers to be removed from the end
            base_model: VGG or alexnet
        """
        super(EncoderCNN, self).__init__()
        self.base_model = base_model
        self.downscale = downscale
        self.pretrained_model = getattr(
            imagemodels, base_model)(pretrained=pretrained)
        # Classifier layer includes FC layers

        if 'resnet' not in self.base_model:
            layers = list(self.pretrained_model.classifier.children())
            modules = layers[:-remove_layers]
            modules.append(nn.Linear(4096, 1024))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(1024, 32))
            modules.append(nn.ReLU())
            self.pretrained_model.classifier = nn.Sequential(*modules)
        else:
            layers = list(self.pretrained_model.children())
            modules = layers[:-remove_layers]
            final_module = [
                nn.Linear(512, 32),
                nn.ReLU()]
            if self.downscale:
                final_module.append(nn.Linear(32, 1))
                final_module.append(nn.ReLU())

            self.pretrained_model = nn.Sequential(*modules)
            self.final_module = nn.Sequential(*final_module)

    def forward(self, x):
        orig_shape = x.shape[:2]
        img_shape = x.shape[-3:]
        x = x.view(-1, *img_shape)
        x = self.pretrained_model(x)
        if 'resnet' in self.base_model:
            x = x.view(x.size(0), -1)
            x = self.final_module(x)

        x = x.view(*orig_shape, -1)
        return x


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim=32,
        hidden_dim=256,
        num_layers=1,
        num_classes=2,
        states=None
    ):

        super(DecoderLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = 256
        self.states = states
        self.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.final_module = [
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ]
        self.final_module = nn.Sequential(*self.final_module)

    def forward(self, x):
        x, (h_n, c_n) = self.LSTM(x)
        x = h_n[-1]
        print(x.shape)
        x = self.final_module(x)
        return x


class Aggregator(nn.Module):
    def __init__(
        self,
        embedding_dim=32,
        seq_len=38,
        max_aggregate=True
    ):

        super(Aggregator, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        if max_aggregate:
            self.layer1 = nn.MaxPool2d(kernel_size=(
                self.seq_len, self.embedding_dim))
        else:
            self.layer1 = nn.AvgPool2d(kernel_size=(
                self.seq_len, self.embedding_dim))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = x.squeeze()
        x = self.sig(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self, pretrained=False, remove_layers=1, base_model='resnet50', downscale=False):
        """
        Args:
            pretrained: Decide if pretrained on image-net needed
            remove_layers: Number of layers to be removed from the end
            base_model: VGG or alexnet
        """
        super(CNNClassifier, self).__init__()
        self.base_model = base_model
        self.downscale = downscale
        self.pretrained_model = getattr(
            imagemodels, base_model)(pretrained=pretrained)
        # Classifier layer includes FC layers

        if 'resnet' not in self.base_model:
            layers = list(self.pretrained_model.classifier.children())
            modules = layers[:-remove_layers]
            modules.append(nn.Linear(4096, 1024))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(1024, 32))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(32, 1))
            modules.append(nn.Sigmoid())
            self.pretrained_model.classifier = nn.Sequential(*modules)
        else:
            layers = list(self.pretrained_model.children())
            modules = layers[:-remove_layers]
            final_module = [
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ]
            
            self.pretrained_model = nn.Sequential(*modules)
            self.final_module = nn.Sequential(*final_module)

    def forward(self, x):

        x = self.pretrained_model(x)
        if 'resnet' in self.base_model:
            x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.final_module(x)
        # print(x.shape)

        # x = x.view(*orig_shape, -1)
        return x
