from torch import nn
import numpy as np
import torch

class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.device = device
        
        # input shape: [b, 3, 256, 256]
        self.encoder = nn.ModuleList([
            # layer 1 - output shape: [b, 32, 128, 128]
            nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            # layer 2 - output shape: [b, 64, 64, 64]
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            # layer 3 - output shape: [b, 128, 32, 32]
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            # layer 4 - output shape: [b, 256, 16, 16]
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            # layer 5 - output shape: [b, 512, 8, 8]
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        ])

        # layer 6 - output shape: [b, 1024, 4, 4]
        self.last_layer = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=1024),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
    def forward(self, x):
        x = x.to(self.device)
        skip_connections = []

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
            
        x = self.last_layer(x)
        return x, skip_connections

class Decoder(nn.Module):
    def __init__(self, hidden_size, device):
        super(Decoder, self).__init__()
        self.device = device
        
        # layer 1 - output shape: [b, 512, 8, 8]
        self.bottleneck_layer = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
        )

        # input shape: [b, 1024, 4, 4]
        self.decoder = nn.ModuleList([
            # layer 2 - output shape: [b, 256, 16, 16]
            nn.Sequential(
                nn.Conv2d(in_channels=(512 + 512), out_channels=512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            ),
            # layer 3 - output shape: [b, 128, 32, 32]
            nn.Sequential(
                nn.Conv2d(in_channels=(256 + 256), out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            ),
            # layer 4 - output shape: [b, 64, 64, 64]
            nn.Sequential(
                nn.Conv2d(in_channels=(128 + 128), out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            ),
            # layer 5 - output shape: [b, 32, 128, 128]
            nn.Sequential(
                nn.Conv2d(in_channels=(64 + 64), out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            ),
            # layer 6 - output shape: [b, 1, 256, 256]
            nn.Sequential(
                nn.Conv2d(in_channels=(32 + 32), out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
            ),
        ])
    
    def forward(self, x, skip_connections):
        x = x.to(self.device)
        x = self.bottleneck_layer(x)

        for i, decoder_layer in enumerate(self.decoder):
            skip_connection = skip_connections[-(i + 1)]
            x = torch.cat((x, skip_connection), dim=1)
            x = decoder_layer(x)
        
        return x
    
class LSTMAutoEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, batch_size, device):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = Encoder(device)

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        
        self.hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        self.cn = torch.randn(self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        self.lstm = nn.LSTM(input_size=feature_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)

        self.decoder = Decoder(hidden_size=self.hidden_size, device=device)

    def forward(self, images, masks):
        # running the input through the output to get latent vector sequences
        x = torch.cat([images, masks], dim=1)
        batch_size, c, h, w = x.size()
        x, skip_connections = self.encoder.forward(x)   # shape: [16, 1024, 4, 4]

        # initializing input for RNN
        x = x.view(batch_size, self.feature_size, -1)   # shape: [16, 1024, 16]
        x = x.permute(0, 2, 1)                          # shape: [16, 16, 1024]
        x = x.to(self.device)
        
        
        h0, c0 = self.hn, self.cn
        if batch_size < self.batch_size:
            h0 = self.hn[:, :batch_size, :]
            c0 = self.cn[:, :batch_size, :]

        outputs = []
        h0, c0 = h0.to(self.device), c0.to(self.device)
        for t in range(x.size(1)):
            output, (h0, c0) = self.lstm(x[:, t, :].unsqueeze(1), (h0.detach(), c0.detach()))
            outputs.append(output)
        
        x = torch.cat(outputs, dim=1)
        # x shape: [16, 512]
        h = w = int(np.sqrt(x.size(1)))
        x = x.view(batch_size, self.hidden_size, h, w)
        x = self.decoder.forward(x, skip_connections)
        return x
