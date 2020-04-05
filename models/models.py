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

    def forward(self, x_3d):
        cnn_embed_seq = []
        for frame_num in range(x_3d.size(0)):
            img_consider = x_3d[frame_num, :, :, :].unsqueeze(0)
            x = self.pretrained_model(img_consider)
            x = x.view(x.shape[1])
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(
        self,
        encoding_dim=2048,
        hidden_features=1024,
        intermediate_features=128
    ):
        super(DecoderRNN, self).__init__()
        self.LSTM_input_size = encoding_dim
        self.hidden_features = hidden_features

        self.LSTM = nn.LSTM(
            input_size=self.LSTM_input_size,
            hidden_size=self.hidden_features
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_features, intermediate_features),
            nn.ReLU(),
            nn.Linear(intermediate_features, 1),
            nn.Sigmoid())

    def forward(self, x_seq):
        print(x_seq.shape)
        x_seq = x_seq.unsqueeze(1)
        self.LSTM.flatten_parameters()
        x, (h_n, h_c) = self.LSTM(x_seq, None)
        print(x.shape)
        x = self.classifier(x[-1, :, :])

        # x = self.fc1(LSTM_out[-1, :, :])
        # x = F.relu(x)
        # x = F.dropout(x)
        # x = self.fc2(x)
        # x = F.sigmoid(x)
        return x
