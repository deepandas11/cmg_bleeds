import torch
import torch.nn as nn
import torchvision.models as imagemodels


class EncoderCNN(nn.Module):
    def __init__(self, pretrained=False, remove_layers=1, base_model='alexnet'):
        """        
        Args:
            pretrained: Decide if pretrained on image-net needed
            remove_layers: Number of layers to be removed from the end
            base_model: VGG or alexnet
        """
        super(EncoderCNN, self).__init__()
        self.pretrained_model = getattr(
            imagemodels, base_model)(pretrained=pretrained)
        # Classifier layer includes FC layers
        layers = list(self.pretrained_model.classifier.children())
        modules = layers[:-remove_layers]
        modules.append(nn.Linear(4096, 2048))
        modules.append(nn.ReLU())
        self.pretrained_model.classifier = nn.Sequential(*modules)

    def forward(self, x):
        orig_shape = x.shape[:2]
        img_shape = x.shape[-3:]

        x = x.view(-1, *img_shape)
        x = self.pretrained_model(x)
        x = x.view(*orig_shape, -1)
        return x


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        encoding_dim=38,
        hidden_features=1,
        intermediate_features=128,
        feature_size=2048,
    ):
        super(DecoderLSTM, self).__init__()
        self.LSTM_input_size = encoding_dim
        self.hidden_features = hidden_features

        self.LSTM = nn.LSTM(
            input_size=self.LSTM_input_size,
            hidden_size=self.hidden_features
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, intermediate_features),
            nn.ReLU(),
            nn.Linear(intermediate_features, 1),
            nn.Sigmoid())

    def forward(self, x_seq):
        self.LSTM.flatten_parameters()
        x_seq = x_seq.permute(0, 2, 1)
        x, (h_n, h_c) = self.LSTM(x_seq, None)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
