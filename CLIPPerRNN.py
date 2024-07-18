import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from collections import namedtuple
import numpy as np
from progress.bar import Bar
from config import device

# pylint: disable=E1101,E1102
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from model import PerformanceRNN
import config

# Import PerformanceRNN from the model.py file
from model import PerformanceRNN  # Adjust the import based on the actual content



class PerformanceRNNComparator:
    def __init__(self, model_path_1, model_path_2):

        model_config = config.model
        self.model_1 = PerformanceRNN(**model_config).to(device)
        self.model_2 = PerformanceRNN(**model_config).to(device)
        
        # Load model weights
        self.model_1.load_state_dict(torch.load(model_path_1))
        self.model_2.load_state_dict(torch.load(model_path_2))
        
        # Set models to evaluation mode
        self.model_1.eval()
        self.model_2.eval()
        
        # Cosine similarity metric
        self.cosine_similarity = CosineSimilarity(dim=1)

    def forward(self, input_1, input_2):
        # Ensure inputs are in the correct format (e.g., tensors)
        if not isinstance(input_1, torch.Tensor):
            input_1 = torch.tensor(input_1)
        if not isinstance(input_2, torch.Tensor):
            input_2 = torch.tensor(input_2)
        
        # Forward pass through the models
        output_1 = self.model_1(input_1)
        output_2 = self.model_2(input_2)
        
        return output_1, output_2

    def compute_cosine_similarity(self, input_1, input_2):
        # Get model outputs
        output_1, output_2 = self.forward(input_1, input_2)
        
        # Compute cosine similarity between outputs
        similarity = self.cosine_similarity(output_1, output_2)
        
        return similarity