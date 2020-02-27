import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels


class EncoderCNN(nn.Module):
    def __init__(self, pretrained=True, remove_layers=1):
        super(EncoderCNN, self).__init__()
        seed_model = getattr(imagemodels, 'vgg16')(pretrained=pretrained)
        # Classifier layer includes FC layers
        layers = list(seed_model.classifier.children())
        modules = layers[:-remove_layers]
        # Defining the model and changing just the classifier section
        self.pretrained_model = seed_model
        self.pretrained_model.classifier = nn.Sequential(*modules)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for frame_num in range(x_3d.size(1)):
            x = self.pretrained_model(x_3d[:, frame_num, :, :, :])
            x = x.view(x.size(0), -1)

            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(
        self,
        encoding_dim=4096,
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
        self.fc1 = nn.Linear(self.hidden_features, intermediate_features)
        self.fc2 = nn.Linear(intermediate_features, 1)

    def forward(self, x_seq):

        self.LSTM.flatten_parameters()
        LSTM_out, (h_n, h_c) = self.LSTM(x_seq, None)

        x = self.fc1(LSTM_out[-1, :, :])
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x
