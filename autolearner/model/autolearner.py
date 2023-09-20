from Karanir import *


class AutoLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config