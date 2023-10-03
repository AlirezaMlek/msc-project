from enum import Enum
from torch import nn
import torch

class BlockType(Enum):
    Input = 1
    Network = 2
    Output = 3

class Branch:
    def __init__(self, name, path, app, out_main, in_main, in_branch=None, out_branch=None, is_main=True,
                 is_residual=True):
        self.is_main = is_main
        self.name = name
        self.path = path
        self.app = app
        self.out_main = out_main
        self.in_branch = in_branch
        self.out_branch = out_branch
        self.in_main = in_main
        self.is_residual = is_residual


class InputType(Enum):
    D1 = 1
    D2 = 2


class Gate(nn.Module):
    def __init__(self, link, node):
        super(Gate, self).__init__()
        self.link = link
        self.nextNode = node


class Node(nn.Module):
    def __init__(self, name, _id, owner, block, inputShape, outputShape, blockType, forward,
                 inputType=InputType.D1, tokenizer=None):
        super(Node, self).__init__()

        self.name = name
        self.id = _id
        self.owner = owner
        self.inputType = inputType
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.inputs = []
        self.outputGates = nn.ModuleDict()
        self.inputGates = nn.ModuleDict()
        self.subscriptions = []
        self.type = blockType
        self.tokenizer = tokenizer
        self.forward_fn = forward
        self.device = torch.device('cpu')
        self.block = nn.ModuleList()
        for b in block: self.block.append(b)

    def add_output_gate(self, path, node, link):
        if link is not None:
            link.to(self.device)
        self.outputGates[path] = Gate(link, node)
        node.add_input(self)
        self.subscriptions.append(path)

    def add_input(self, node):
        self.inputs.append(node)

    def add_input_gate(self, branch, link):
        link.to(self.device)
        self.inputGates[branch] = link

    def forward(self, data, branch):
        # forward input link

        if self.inputGates.keys().__contains__(branch.name):
            if isinstance(data, dict) and data.keys().__contains__('data1'):
                data = self.inputGates[branch.name](**data)
            else:
                data = self.inputGates[branch.name](data)

        x = self.forward_fn(data, self.block)
        if self.type == BlockType.Output:
            return x

        # get next node
        if self.outputGates.keys().__contains__(branch.name):
            next_name = branch.name
        else:
            next_name = branch.path

        # forward gate
        gate = self.outputGates[next_name]
        if gate.link is not None:
            x = self.outputGates[next_name].link(x)

        # if it has another owner then return data
        if self.name in [branch.out_main, branch.out_branch]:
            return x

        if gate.nextNode.name == branch.in_main:
            return x

        return self.outputGates[next_name].nextNode(x, branch)
