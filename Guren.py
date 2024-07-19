import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from collections import namedtuple
import numpy as np
from config import device

# pylint: disable=E1101,E1102
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from model import PerformanceRNN
import config

from Utils.ControlSeq import ControlSeq

# Import PerformanceRNN from the model.py file
from model import PerformanceRNN  # Adjust the import based on the actual content


# Gated Unit REcurrent Nerwork


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
    
class Guren:
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
        output = self.model_1(input_1)
        #controls = output.getcontrol()
        _, _,compressed_controls, _ = torch.load(output)
        controls = ControlSeq.recover_compressed_array(compressed_controls)
        controls = torch.tensor(controls, dtype=torch.float32)
        controls = controls.unsqueeze(1).repeat(1, 64, 1).to(device)
        #control = f'control sequence from "{controls}"'

        outputs = self.generate(1,2,controls=controls,
                                 greedy=0.5, #commentata
                                 temperature=0.8,# impostabile a run time ogni 30 sec
                                 verbose=True)

        
        return outputs
    
    def generate(self, init, steps, events=None, controls=None, greedy=1.0,
                    temperature=1.0, teacher_forcing_ratio=1.0, output_type='index', verbose=False):
            # init [batch_size, init_dim]
            # events [steps, batch_size] indeces
            # controls [1 or steps, batch_size, control_dim]

            batch_size = init.shape[0]
            assert init.shape[1] == self.init_dim
            assert steps > 0

            use_teacher_forcing = events is not None
            if use_teacher_forcing:
                assert len(events.shape) == 2
                assert events.shape[0] >= steps - 1
                events = events[:steps-1]

            event = self.get_primary_event(batch_size)
            use_control = controls is not None
            if use_control:
                controls = self.expand_controls(controls, steps)
            hidden = self.init_to_hidden(init)

            outputs = []
            step_iter = range(steps)
            
            for step in step_iter:
                control = controls[step].unsqueeze(0) if use_control else None
                output, hidden = self.forward(event, control, hidden)

                use_greedy = np.random.random() < greedy
                event = self._sample_event(output, greedy=use_greedy,
                                        temperature=temperature)

                if output_type == 'index':
                    outputs.append(event)
                elif output_type == 'softmax':
                    outputs.append(self.output_fc_activation(output))
                elif output_type == 'logit':
                    outputs.append(output)
                else:
                    assert False

                if use_teacher_forcing and step < steps - 1: # avoid last one
                    if np.random.random() <= teacher_forcing_ratio:
                        event = events[step].unsqueeze(0)
            
            return torch.cat(outputs, 0)