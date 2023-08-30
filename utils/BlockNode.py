from enum import Enum


class BlockType(Enum):
    Input = 1
    Network = 2
    Output = 3
    Bridge = 4



class InputType(Enum):
    D1 = 1
    D2 = 2


class Gate:
    def __init__(self, fc, node):
        self.fc = fc
        self.nextNode = node


class Node:
    def __init__(self, name, _id, owner, block, inputShape, outputShape, blockType,
                 inputType=InputType.D1, tokenizer=None):
        self.name = name
        self.id = _id
        self.owner = owner
        self.block = block
        self.inputType = inputType
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.inputs = []
        self.outputGates = {}
        self.subscriptions = []
        self.type = blockType
        self.tokenizer = tokenizer

    def add_output_gate(self, path, node, fc):
        self.outputGates[path] = Gate(fc, node)
        node.add_input(self)
        self.subscriptions.append(path)

    def add_input(self, node):
        self.inputs.append(node)
