import Karanir
from Karanir.dklearn import *
from Karanir.dklearn.nn.cnn import ConvolutionUnits

class SceneNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # [Convolutional Bakcbone]
        self.convs_backbone = ConvolutionUnits(config.input_channels,config.conv_dim, config.latent_dim)

    def forward(self, x):
        return x

    def store_parameters(self, path):
        return
    
    def load_parameters(self, path, map_location):
        self.load_state_dict(torch.load(path, map_location= map_location))