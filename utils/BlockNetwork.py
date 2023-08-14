import torch
import torch.nn as nn
from utils.BlockNode import *
import os


"""
    Getting access to the input nodes. All application has this shared attribute
"""
class BlockNetwork:
    inputNodes = {}

    def __init__(self, name):
        self.name = name

    def get_input_node(self, owner):
        if self.inputNodes.__contains__(owner):
            return self.inputNodes[owner]
        else:
            return None

    def update_input_node(self, owner, node):
        BlockNetwork.inputNodes[owner] = node



"""
    Each App has some path. a main path and other shared paths.
    * for training it first fetches all the path's nodes (from its own nodes and another path's nodes)
"""
class Path(nn.Module):

    App = {}

    def __init__(self, name, app, nodes, forward=None, forward_backup=None, forward_label=None):
        super(Path, self).__init__()
        self.app = app
        self.name = name
        self.acc = None
        self.ops = -1
        self.update_nodes(name, nodes)
        self.load_fcs()
        self.layers = nn.ModuleList()
        self.fc_layers = []
        self.forward_label = forward_label
        self.forward_fn = forward
        self.forward_backup_fn = forward_backup

    def forward(self, inputs):
        return self.forward_fn(inputs, self.layers)


    def forward_backup(self, inputs):
        return self.forward_backup_fn(inputs, self.name, self.App, self.app)


    def fetch_fc(self):
        currentNode = self.App[self.app].get_input_node()

        while currentNode.outputGates.keys().__contains__(self.name):
            if isinstance(currentNode.block, list):
                for l in currentNode.block:
                    self.layers.append(l)
                    self.layers[-1].requires_grad_(False)
            else:
                self.layers.append(currentNode.block)
                self.layers[-1].requires_grad_(False)

            gate = currentNode.outputGates[self.name]
            currentNode = gate.nextNode
            if gate.fc is not None:
                for fc in gate.fc:
                    self.fc_layers.append(len(self.layers))
                    self.layers.append(fc)
                    self.layers[-1].requires_grad_(True)
                batch_norm = nn.BatchNorm1d(num_features=gate.fc[-1].out_features)
                self.layers.append(batch_norm)

        if currentNode.type == BlockType.Output:
            for l in currentNode.block:
                self.layers.append(l)
                self.layers[-1].requires_grad_(False)


    def get_input_node(self):
        return self.App[self.app].get_input_node()



    """
    update nodes when a path is created
    """
    def update_nodes(self, name, nodes):

        for i in range(len(nodes) - 1):
            if len(nodes[i + 1].inputs) != 0 and nodes[i + 1].inputs[0] != nodes[i]:
                fc = [nn.Linear(nodes[i].outputShape, nodes[i + 1].inputShape) for _ in range(5)]
            else:
                fc = None

            nodes[i].add_output_gate(name, nodes[i + 1], fc)

        self.App[self.app].update_nodes(nodes[0], self.name)


    def save_fcs(self):
        for i, index in enumerate(self.fc_layers):
            file_name = f'./cache/{self.name}/{str(i)}.pth'
            torch.save(self.layers[index].state_dict(), file_name)

    def load_fcs(self):
        currentNode = self.App[self.app].get_input_node()
        fc_counter = 0
        while currentNode.outputGates.keys().__contains__(self.name):

            gate = currentNode.outputGates[self.name]
            currentNode = gate.nextNode
            if gate.fc is not None:
                for fc in gate.fc:
                    file_name = f'./cache/{self.name}/{str(fc_counter)}.pth'
                    if os.path.exists(file_name):
                        fc.load_state_dict(torch.load(file_name))

                    fc_counter += 1

        self.App[self.app].update_nodes(find_root(currentNode), self.name)


class DnnApp:

    network = None

    def __init__(self, name, tag, description=None, predictor=None):
        self.name = name
        self.description = description
        self.tag = tag
        self.paths = []
        self.predictor = predictor
        Path.App[name] = self

    def update_nodes(self, node, pathName):
        self.network.update_input_node(self.name, node)
        self.paths.append(pathName)

        currentNode = node
        while currentNode.outputGates.keys().__contains__(pathName) and currentNode.type != BlockType.Output:
            gate = currentNode.outputGates[pathName]
            currentNode = gate.nextNode
            if currentNode.owner != self.name:
                rootNode = find_root(currentNode)
                self.network.update_input_node(rootNode.owner, rootNode)
                break



    def get_input_node(self):
        return self.network.get_input_node(self.name)


    def instantiate(self, tokenizer, embBlock, networkLayers, outputBlock, inputSize, outputSize, forward=None,
                    forward_backup=None):

        outputSizeEmb = embBlock.word_embeddings.embedding_dim

        inputNode = Node("{}:input".format(self.tag), 0, self.name, embBlock,
                         None, outputSizeEmb, BlockType.Input, tokenizer)

        nodeList = [inputNode]
        numEncoders = len(networkLayers)

        for i in range(numEncoders):
            block = [networkLayers[i]]
            node = Node('{}:{}'.format(self.tag, i+1), i+1, self.name, block, inputSize, outputSize, BlockType.Network)
            nodeList.append(node)

        outputNode = Node('{}:output'.format(self.tag), numEncoders+1, self.name, outputBlock, inputSize,
                          outputSize, BlockType.Output)
        nodeList.append(outputNode)

        mainPath = Path('{}:path-main'.format(self.name), self.name, nodeList, forward=forward,
                        forward_backup=forward_backup)
        self.paths = [mainPath.name]

        self.network.update_input_node(self.name, inputNode)

        return mainPath, self


    def predict(self, scores):
        return self.predictor(scores)




def find_root(node):
    if node.type == BlockType.Input:
        return node
    else:
        return find_root(node.inputs[0])