import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from collections import namedtuple
import numpy as np
from progress.bar import Bar
from config import device

class PerformanceRNN2(nn.Module):
    def __init__(self, event_dim, control_dim, init_dim, hidden_dim,
                 inithid_fc = None, gru_layers=3, gru_dropout=0.3):
        super().__init__()

        self.event_dim = event_dim
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.concat_dim = event_dim + event_dim + 1 + control_dim
        self.input_dim = hidden_dim
        self.output_dim = event_dim

        self.primary_event = self.event_dim - 1

        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)

        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          num_layers=gru_layers, dropout=gru_dropout)
        self.output_fc = nn.Linear(hidden_dim * gru_layers, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forward(self, event, event2, control=None, hidden=None, condition = None):
        # One step forward

        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        event = self.event_embedding(event)#mix here

        if control is None:
            default = torch.ones(1, batch_size, 1).to(device)
            control = torch.zeros(1, batch_size, self.control_dim).to(device)
        else:
            default = torch.zeros(1, batch_size, 1).to(device)
            assert control.shape == (1, batch_size, self.control_dim)

        concat = torch.cat([event, default, control, event2], -1) # TODO add dimension to consider event output
        input = self.concat_input_fc(concat)  #nn.Linear(self.concat_dim, self.input_dim + event_dim)
        input = self.concat_input_fc_activation(input)  #nn.LeakyReLU(0.1, inplace=True)

        _, hidden = self.gru(input, hidden)  #nn.GRU(self.input_dim, self.hidden_dim,num_layers=gru_layers, dropout=gru_dropout)
        output = hidden.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output) #nn.Linear(hidden_dim * gru_layers, self.output_dim)
        return output, hidden # output is under the form of a logit
    
    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)
    
    def init_to_hidden(self, init):
        # [batch_size, init_dim]
        batch_size = init.shape[0]
        out = self.inithid_fc(init)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out
    
    def expand_controls(self, controls, steps):
        # [1 or steps, batch_size, control_dim]
        assert len(controls.shape) == 3
        assert controls.shape[2] == self.control_dim
        if controls.shape[0] > 1:
            assert controls.shape[0] >= steps
            return controls[:steps]
        return controls.repeat(steps, 1, 1)
    
    def generate(self, init, batch_size, init_dim, steps, events=None, events2=None, controls=None, greedy=1.0,
                 temperature=1.0, teacher_forcing_ratio=1.0, output_type='softmax', verbose=False):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        # controls [1 or steps, batch_size, control_dim]

        self.init_dim = init_dim

        batch_size = batch_size
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[:steps-1]

        event = self.get_primary_event(batch_size)
        event2 = self.get_primary_event(batch_size)
        use_control = controls is not None
        if use_control:
            controls = self.expand_controls(controls, steps)
        hidden = self.init_to_hidden(init)

        outputs = []
        step_iter = range(steps)
        if verbose:
            step_iter = Bar('Generating').iter(step_iter)

        for step in step_iter:
            control = controls[step].unsqueeze(0) if use_control else None
            output, hidden = self.forward(event, event2, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy,
                                       temperature=temperature)

            if output_type == 'index':
                outputs.append((event,output))
                return outputs
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

class Guren(nn.Module):
    def __init__(self, modelA = None, modelB = None):
        super().__init__()

        self.modelA = modelA.to('cuda')
        self.modelB = modelB.to('cuda')
        

    def generate(self, e1, e2, c1, c2, batch_size,
                 window_size, control_ratio = 1.0):
        outputs = []
        events = torch.LongTensor(e1).to(device)
        assert events.shape[0] == window_size
        assert len(events.shape) == 2
        assert events.shape[0] >= window_size - 1

        if np.random.random() < control_ratio:
            controls = torch.FloatTensor(c1).to(device)
            assert controls.shape[0] == window_size
        else:
            controls = None

        init = torch.randn(batch_size, self.modelA.init_dim).to(device)

        outputsA = self.modelA.generate(init, batch_size, self.modelA.init_dim, 
                                     window_size, events=events[:-1], controls=controls,output_type='logit')

        events2 = torch.LongTensor(e2).to(device)
        assert events2.shape[0] == window_size
        assert len(events2.shape) == 2
        assert events2.shape[0] >= window_size - 1

        if np.random.random() < control_ratio:
            controls = torch.FloatTensor(c2).to(device)
            assert controls.shape[0] == window_size
        else:
            controls = None

        init = torch.randn(batch_size, self.modelB.init_dim).to(device)

        outputsB = self.modelB.generate(init,batch_size, self.modelB.init_dim, window_size, 
                            events=events2[:-1], controls=controls)
        
        assert outputsB.shape[:2] == events2.shape[:2]
        outputsF = torch.cat((outputsA, outputsB),2)
        #print(outputsF)

        softed = self.modelB.output_fc_activation(outputsF)
        outputs.append(softed)

        return torch.cat(outputs, 0)