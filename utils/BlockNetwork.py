from utils.BlockNode import *
import os
from models.Link import ConcatLayer
from utils.Utils import *

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

    def find_node(self, app, name):
        node = self.get_input_node(app)
        path = app + ':path-main'
        while node.name != name:
            node = node.outputGates[path].nextNode

        return node


"""
    Each App has some path. a main path and other shared paths.
    * for training it first fetches all the path's nodes (from its own nodes and another path's nodes)
"""
class Path(nn.Module):
    App = {}

    def __init__(self, name, app, nodes, link=None, forward_label=None):
        super(Path, self).__init__()
        self.app = app
        self.name = name
        self.acc = None
        self.ops = -1
        self.link = link
        self.update_nodes(name, nodes)
        self.inputNode = self.App[self.app].get_input_node()
        self.load_links()
        self.layers_host = nn.ModuleList()
        self.forward_label = forward_label
        self.num_nodes = len(nodes) - 1

        main_branch = Branch(self.name, self.name, self.app, 0, 0)
        self.branches = {'main': main_branch}

    def forward(self, inputs, branch_name=None):

        if branch_name is not None:
            branch = self.branches[branch_name]
        else:
            branch = self.branches['main']

        node = self.inputNode
        data = node(inputs, branch)

        if branch.is_main:
            return data

        out_main_node = node
        while out_main_node.name != branch.out_main:
            out_main_node = out_main_node.outputGates[self.name].nextNode

        in_branch_node = out_main_node.outputGates[branch_name].nextNode
        data_copy = copy_data(data)
        branch_data = in_branch_node(data_copy, branch)

        branch.name = self.name
        out_main_node_next = out_main_node.outputGates[self.name].nextNode

        if branch.is_residual:
            main_data = out_main_node_next(data, branch)

        in_main_node = out_main_node_next
        while in_main_node.name != branch.in_main:
            in_main_node = in_main_node.outputGates[self.name].nextNode

        branch.name = branch_name
        if branch.is_residual:
            comb_data = {'data1': main_data, 'data2': branch_data}
            final_data = in_main_node(comb_data, branch)
        else:
            final_data = in_main_node(branch_data, branch)

        return final_data

    """
    update nodes when a path is created
    """
    def update_nodes(self, name, nodes, is_branch=False, residual=True):

        for i in range(len(nodes) - 1):
            if nodes[i].owner != nodes[0].owner or \
                    (nodes[i + 1].id - 1 != nodes[i].id and nodes[i + 1].owner == nodes[0].owner):

                if nodes[i].inputType == InputType.D1 and nodes[i + 1].owner != nodes[0].owner:
                    #                     nodes[i].block[0].output.LayerNorm = nn.Sequential()
                    #                     nodes[i].block[0].output.dropout = nn.Sequential()
                    nodes[i].block[0].layer_norm2 = nn.Sequential()

                    add_norm = True
                else:
                    add_norm = False

                link = self.link(nodes[i], nodes[i + 1], add_norm)
                nodes[i].add_output_gate(name, nodes[i + 1], link)

            elif not is_branch or (nodes[i].owner == nodes[0].owner and nodes[i + 1].owner != nodes[0].owner):
                link = None
                nodes[i].add_output_gate(name, nodes[i + 1], link)

            if nodes[i].owner == nodes[0].owner and nodes[i - 1].owner != nodes[0].owner:
                link = ConcatLayer(nodes[i])
                nodes[i].add_input_gate(name, link)

            elif nodes[i].owner != nodes[0].owner and nodes[i - 1].owner == nodes[0].owner:
                link = self.link(nodes[i - 1], nodes[i])
                nodes[i].add_input_gate(name, link)

        if nodes[-1].owner == nodes[0].owner and nodes[-2].owner != nodes[0].owner and residual:
            link = ConcatLayer(nodes[-1])
            nodes[-1].add_input_gate(name, link)

        self.App[self.app].update_nodes(nodes[0], self.name)

    def save_links(self, currentNode=None):
        if currentNode is None:
            currentNode = self.inputNode

        for gate in currentNode.inputGates.keys():
            name = f'./cache/{self.name}/{currentNode.name}-in-{gate}.pth'
            link = currentNode.inputGates[gate]
            torch.save(link, name)

        outputGates = currentNode.outputGates
        for gate in outputGates.keys():
            if gate.__contains__(self.name):
                link = outputGates[gate].link
                if link is not None:
                    name = f'./cache/{self.name}/{currentNode.name}-{outputGates[gate].nextNode.name}-out-{gate}.pth'
                    torch.save(link, name)

                self.save_links(outputGates[gate].nextNode)

    def load_links(self):
        currentNode = self.inputNode
        while currentNode.outputGates.keys().__contains__(self.name):
            outputGates = currentNode.outputGates
            for gate in outputGates:
                if gate.__contains__(self.name):
                    link = outputGates[gate].link
                    if link is not None:
                        name = f'./cache/{self.name}/{currentNode.name}-{outputGates[gate].nextNode.name}-out-{gate}.pth'
                        if os.path.exists(name):
                            outputGates[gate].link.load_state_dict(torch.load(name))

            currentNode = outputGates[self.name].nextNode

        self.App[self.app].update_nodes(find_root(currentNode), self.name)

    def new_branch(self, App, out_main, in_host, out_host, in_main, residual=True):

        inputNode1 = self.inputNode
        inputNode2 = App.get_input_node()
        nodes = collect_new_path_nodes_cross_path(inputNode1, inputNode2, out_main, in_host, out_host, in_main,
                                                  self.name)

        out_main_name = nodes[out_main].name
        in_host_name = nodes[out_main + 1].name
        out_host_name = nodes[out_main + out_host - in_host + 1].name
        in_main_name = nodes[out_main + out_host - in_host + 2].name

        branchName = self.name + '-b' + str(len(self.branches))
        self.update_nodes(branchName, nodes, is_branch=True, residual=residual)

        self.branches[branchName] = Branch(branchName, self.name, App, out_main_name, in_main_name, in_host_name,
                                           out_host_name, False, residual)

    def link_require_grad(self, branch_name='main', require=True, currentNode=None):

        if currentNode is None:
            currentNode = self.inputNode

        for gate in currentNode.inputGates.keys():
            link = currentNode.inputGates[gate]
            for param in link.parameters():
                param.requires_grad = require

        outputGates = currentNode.outputGates
        for gate in outputGates.keys():
            if gate in [self.name, branch_name]:
                link = outputGates[gate].link
                if link is not None:
                    for param in link.parameters():
                        param.requires_grad = require

                self.link_require_grad(branch_name, require, outputGates[gate].nextNode)


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

    def instantiate(self, tokenizer, embBlock, networkLayers, outputBlock, inputSize, outputSize=None,
                    inputType=InputType.D1, forward=None, link=None):

        if isinstance(inputSize, list):
            outSize = inputSize[0]
        else:
            outSize = outputSize
        inputNode = Node("{}:input".format(self.tag), 0, self.name, [embBlock],
                         None, outSize, BlockType.Input, forward, inputType, tokenizer)

        nodeList = [inputNode]
        numEncoders = len(networkLayers)

        for i in range(numEncoders):
            block = [networkLayers[i]]

            if isinstance(inputSize, list):
                inSize = inputSize[i]
                outSize = inputSize[i + 1]
            else:
                inSize = inputSize
                outSize = outputSize

            node = Node('{}:{}'.format(self.tag, i + 1), i + 1, self.name, block, inSize, outSize,
                        BlockType.Network, inputType=inputType, forward=forward)
            nodeList.append(node)

        if isinstance(inputSize, list):
            inSize = inputSize[-1]
        else:
            inSize = inputSize
        outputNode = Node('{}:output'.format(self.tag), numEncoders + 1, self.name, outputBlock, inSize,
                          None, BlockType.Output, inputType=inputType, forward=forward)
        nodeList.append(outputNode)

        mainPath = Path('{}:path-main'.format(self.name), self.name, nodeList, link=link)
        self.paths = [mainPath.name]

        self.network.update_input_node(self.name, inputNode)

        return mainPath, self

    def predict(self, scores):
        return self.predictor(scores)

    def find_node(self, app, name):
        return self.network.find_node(app, name)

    def set_device(self, device):
        main_path = self.paths[0]
        node = self.get_input_node()
        node.to(device)
        node.device = device

        while node.outputGates.keys().__contains__(main_path):
            node = node.outputGates[main_path].nextNode
            node.to(device)
            node.device = device
