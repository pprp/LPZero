import json
import math

import networkx as nx
import numpy as np
import torch
import torch.nn
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from multilinear import MultiLinear
from weight_drop import ParameterListWeightDrop, WeightDrop


# From NAS-Bench-NLP https://github.com/fmsnew/nas-bench-nlp-release
class CustomRNNCell(torch.nn.Module):
    elementwise_ops_dict = {'prod': torch.mul, 'sum': torch.add}

    def __init__(self, input_size, hidden_size, recepie):
        super(CustomRNNCell, self).__init__()

        self.activations_dict = {
            'tanh': torch.nn.Tanh(),
            'sigm': torch.nn.Sigmoid(),
            'leaky_relu': torch.nn.LeakyReLU(),
        }

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recepie = recepie
        self.hidden_tuple_size = 0

        components_dict = {}

        self.G = nx.DiGraph()
        for k in recepie.keys():
            if k not in components_dict:
                component = self._make_component(recepie[k])
                if component is not None:
                    components_dict[k] = component
                if k.startswith('h_new'):
                    suffix = k.replace('h_new_', '')
                    if suffix.isdigit():
                        self.hidden_tuple_size = max(
                            [self.hidden_tuple_size, int(suffix) + 1]
                        )

                if k not in self.G.nodes():
                    self.G.add_node(k)
                for i, n in enumerate(recepie[k]['input']):
                    if n not in self.G.nodes():
                        self.G.add_node(k)
                    self.G.add_edge(n, k)

        self.components = torch.nn.ModuleDict(components_dict)
        self.nodes_order = list(nx.algorithms.dag.topological_sort(self.G))

    def forward(self, x, hidden_tuple):
        calculated_nodes = {}
        # Modified to be able to get hidden states
        hidden_tuple[0].requires_grad_()
        hidden_tuple[0].retain_grad()
        hidden_states.append(hidden_tuple[0])
        for n in self.nodes_order:
            if n == 'x':
                calculated_nodes['x'] = x.unsqueeze(0)
            elif n.startswith('h_prev') and n.replace('h_prev_', '').isdigit():
                calculated_nodes[n] = hidden_tuple[
                    int(n.replace('h_prev_', ''))
                ].unsqueeze(0)
            elif n in self.components:
                inputs = [calculated_nodes[k]
                          for k in self.recepie[n]['input']]
                calculated_nodes[n] = self.components[n](*inputs)
            else:
                # simple operations
                op = self.recepie[n]['op']
                inputs = [calculated_nodes[k]
                          for k in self.recepie[n]['input']]
                if op in ['elementwise_prod', 'elementwise_sum']:
                    op_func = CustomRNNCell.elementwise_ops_dict[
                        op.replace('elementwise_', '')
                    ]
                    calculated_nodes[n] = op_func(inputs[0], inputs[1])
                    for inp in range(2, len(inputs)):
                        calculated_nodes[n] = op_func(
                            calculated_nodes[n], inputs[i])
                elif op == 'blend':
                    calculated_nodes[n] = (
                        inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]
                    )
                elif op.startswith('activation'):
                    op_func = self.activations_dict[op.replace(
                        'activation_', '')]
                    calculated_nodes[n] = op_func(inputs[0])
                    # calculate and store K codes for activations in RNN - LeakyReLU, TanH, Sigmoid
                    calculate_activations(calculated_nodes[n])
        return tuple(
            [calculated_nodes[f'h_new_{i}'][0]
                for i in range(self.hidden_tuple_size)]
        )

    def _make_component(self, spec):
        if spec['op'] == 'linear':
            input_sizes = [
                self.input_size if inp == 'x' else self.hidden_size
                for inp in spec['input']
            ]
            return MultiLinear(input_sizes, self.hidden_size)


class CustomRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, recepie):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell = CustomRNNCell(input_size, hidden_size, recepie)
        self.reset_parameters()

    def forward(self, inputs, hidden_tuple=None):
        batch_size = inputs.size(1)
        if hidden_tuple is None:
            hidden_tuple = tuple(
                [
                    self.init_hidden(batch_size)
                    for _ in range(self.cell.hidden_tuple_size)
                ]
            )

        self.check_hidden_size(hidden_tuple, batch_size)

        hidden_tuple = tuple([x[0] for x in hidden_tuple])
        outputs = []
        for x in torch.unbind(inputs, dim=0):
            hidden_tuple = self.cell(x, hidden_tuple)
            outputs.append(hidden_tuple[0].clone())

        return torch.stack(outputs, dim=0), tuple(
            [x.unsqueeze(0) for x in hidden_tuple]
        )

    def init_hidden(self, batch_size):
        # num_layers == const (1)
        return torch.zeros(1, batch_size, self.hidden_size).to(
            next(self.parameters()).device
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            torch.nn.init.uniform_(param, -stdv, stdv)

    def check_hidden_size(self, hidden_tuple, batch_size):
        expected_hidden_size = (1, batch_size, self.hidden_size)
        msg = 'Expected hidden size {}, got {}'
        for hx in hidden_tuple:
            if hx.size() != expected_hidden_size:
                raise RuntimeError(msg.format(
                    expected_hidden_size, tuple(hx.size())))


# From NAS-Bench-NLP https://github.com/fmsnew/nas-bench-nlp-release
class AWDRNNModel(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    # add batch_size parameter
    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        wdrop=0,
        tie_weights=False,
        recepie=None,
        verbose=True,
    ):
        super(AWDRNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = torch.nn.Dropout(dropouti)
        self.hdrop = torch.nn.Dropout(dropouth)
        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(ntoken, ninp)
        self.wdrop = wdrop
        self.verbose = verbose
        self.ntoken = ntoken

        if recepie is not None:
            recepie = json.loads(recepie)

        self.rnns = []
        for i in range(nlayers):
            input_size = ninp if i == 0 else nhid
            hidden_size = nhid if i != nlayers - \
                1 else (ninp if tie_weights else nhid)
            if rnn_type == 'LSTM':
                self.rnns.append(torch.nn.LSTM(input_size, hidden_size))
            elif rnn_type == 'CustomRNN':
                self.rnns.append(CustomRNN(input_size, hidden_size, recepie))

        if wdrop:
            if rnn_type == 'LSTM':
                self.rnns = [
                    WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                    for rnn in self.rnns
                ]
            elif rnn_type == 'CustomRNN':
                wd_rnns = []
                for rnn in self.rnns:
                    multilinear_components = []
                    for k, v in rnn.cell.components.items():
                        if rnn.cell.recepie[k]['op'] == 'linear':
                            for i in np.where(
                                np.array(rnn.cell.recepie[k]['input']) != 'x'
                            )[0]:
                                multilinear_components.append(
                                    f'cell.components.{k}.weights.{i}'
                                )
                    wd_rnns.append(
                        ParameterListWeightDrop(
                            rnn, multilinear_components, dropout=wdrop
                        )
                    )
                    self.rnns = wd_rnns

        if self.verbose:
            print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = torch.nn.Linear(nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.recepie = recepie

    def reset(self):
        pass

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, skip_embedding=False):
        emb = (
            input
            if skip_embedding
            else embedded_dropout(
                self.encoder, input, dropout=self.dropoute if self.training else 0
            )
        )

        # store embedding output
        self.embeddings = emb
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for i, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[i])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.nlayers - 1:
                # self.hdrop(raw_output) add???
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        result = output.view(output.size(0) * output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden = []
        for i in range(self.nlayers):
            if self.rnn_type == 'LSTM':
                hidden_tuple_size = 2
            elif self.rnn_type == 'CustomRNN':
                if self.wdrop:
                    # wrapped with ParameterListWeightDrop
                    hidden_tuple_size = self.rnns[0].module.cell.hidden_tuple_size
                else:
                    hidden_tuple_size = self.rnns[0].cell.hidden_tuple_size
            hidden_size = (
                self.nhid
                if i != self.nlayers - 1
                else (self.ninp if self.tie_weights else self.nhid)
            )
            hidden.append(
                tuple(
                    [
                        weight.new(1, bsz, hidden_size).zero_()
                        for _ in range(hidden_tuple_size)
                    ]
                )
            )

        return hidden
