import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from utils import EdgeDecoder


class FIG(nn.Module):

    def __init__(self, num_classes=10, num_frcs=45*3, channels=1):
        super(FIG, self).__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(num_classes, channels)
        self.decoder = Decoder(num_classes, num_frcs, channels)

    
    def model(self, data, label=None):

        pyro.module("decoder", self)

        batch_size = data.size(0)

        with pyro.plate("data"):

            deformation_loc = torch.zeros([batch_size, 1])
            deformation_scale = torch.ones([batch_size, 1])

            label_prior = data.new_ones([batch_size, self.num_classes]) / (1.0 * self.num_classes)

            deformation = pyro.sample("deformation", dist.Normal(deformation_loc, deformation_scale).to_event(1))
            label = pyro.sample("label", dist.OneHotCategorical(label_prior), obs=label)

            final_image = self.decoder(deformation, label, data)

            pyro.sample("image", dist.Bernoulli(final_image).to_event(1), obs=data)  


    def guide(self, data, label=None):

        pyro.module("encoder", self)

        with pyro.plate("data"):

            label_prior, deformation_loc, deformation_scale = self.encoder(data)

            pyro.sample("label", dist.OneHotCategorical(label_prior))
            pyro.sample("deformation", dist.Normal(deformation_loc, deformation_scale))


    def infer(self, data):
        label_prior, d_loc, d_scale = self.encoder(data)
        deformation = dist.Normal(d_loc, d_scale).sample()
        return label_prior, deformation

    
    def reconstruct(self, data):
        label_prior, deformation = self.infer(data)
        return self.decoder(deformation, label_prior, data)


class Decoder(nn.Module):

    def __init__(self, num_classes, num_frcs, channels):
        super(Decoder, self).__init__()
        self.to_frcs = nn.Linear(num_classes + 1, num_frcs)
        self._edge_decoder = EdgeDecoder()
        self.final_conv = nn.Conv2d(1, channels, kernel_size=1)


    def forward(self, deformation, label, data):
        batch_size = data.size(0)
        z = torch.cat([deformation, label], dim=1)

        frcs = self.to_frcs(z)
        frcs = frcs.view(batch_size, -1, 3)
        
        edge_map = self._edge_decoder.decode_edge_map(frcs, data).float()

        return self.final_conv(edge_map)    


class Encoder(nn.Module):

    def __init__(self, num_classes, channels):
        super(Encoder, self).__init__()
        self.features = self.build_feature_extractor(channels)
        self.deformation_loc = nn.Linear(5408, 1)
        self.deformation_scale = nn.Linear(5408, 1)
        self.label_prior = nn.Linear(5408, num_classes)


    def forward(self, data):
        z = self.features(data)
        z = z.view(z.size(0), -1)

        d_loc = self.deformation_loc(z)
        d_scale = self.deformation_scale(z)

        label_prior = self.label_prior(z)

        return label_prior, d_loc, d_scale


    def build_feature_extractor(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 32, kernel_size=1)
        )


if __name__ == '__main__':
    bs = 1
    x = torch.rand([bs, 1, 105, 105])
    net = FIG(10)
    a = net.reconstruct(x)
    print(a.shape)