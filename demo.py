# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-20 05:14:36
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-20 05:21:40
from autolearner.model import *
from autolearner.config import *

agent = AutoLearner(config)

from autolearner.train import *

print(agent)