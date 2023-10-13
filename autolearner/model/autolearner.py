from karanir import *
from .perception import *
from .knowledge import *
from .physics import *
from .reason import *

class AutoLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # [Knowledge]

        # [Perception]
        self.perception = SceneNet(config)

        # [Physics]

        # [Reasoning and Planning]
    
    def forward(self, x):
        return x

    def restore_parameters(self, path, map_location = None):
        if map_location is None: map_location = self.device

    
    def save_parameter(self, path, map_location = None):
        if map_location is None: map_location = self.device