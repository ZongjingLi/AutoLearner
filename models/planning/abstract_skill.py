import torch
import torch.nn as nn

class AbstractAction:
    def __init__(self):
        pass


class QuasiPDDL(nn.Module):
    def __init__(self):
        self.reader = reader