from utils.BlockNetwork import *
import copy

"""
    new path contains some nodes of path1 and path2
    @param inputNode1: input node of path1
    @param inputNode2: input node of path2
    @param idOutBranch1: id of output node of branch1. new path cross from path1 to path2
    @param idInBranch2: id of input node of branch2. new path cross from path1 to path2
    @param idOutBranch2: id of output node of branch2. new path cross from path2 to path1
    @param idInBranch1: id of input node of branch1. new path cross from path2 to path1 and continue to the output node of path1
    @return: list of nodes in the new path
"""
def collect_new_path_nodes_cross_path(inputNode1, inputNode2, idOutBranch1, idInBranch2, idOutBranch2,
                                      idInBranch1, addBridge):

    # collect all nodes in the path
    path = [inputNode1]


    # nodes must be in either path-main1 or path-main2
    mainPath1 = inputNode1.owner + ':path-main'
    mainPath2 = inputNode2.owner + ':path-main'

    # collect nodes in path1
    nodeOutBranch1 = inputNode1
    while nodeOutBranch1.id != idOutBranch1:
        nodeOutBranch1 = nodeOutBranch1.outputGates[mainPath1].nextNode
        path.append(nodeOutBranch1)


    # find inBranch node of branch2
    nodeInBranch2 = inputNode2
    while nodeInBranch2.id != idInBranch2:
        nodeInBranch2 = nodeInBranch2.outputGates[mainPath2].nextNode


    if addBridge:
        bridge = make_bridge(nodeInBranch2.inputs[0])
        path.append(bridge)

    # collect nodes in path2
    path.append(nodeInBranch2)

    nodeOutBranch2 = nodeInBranch2
    while nodeOutBranch2.id != idOutBranch2:
        nodeOutBranch2 = nodeOutBranch2.outputGates[mainPath2].nextNode
        path.append(nodeOutBranch2)


    # find inBranch node of branch1
    nodeInBranch1 = nodeOutBranch1.outputGates[mainPath1].nextNode
    while nodeInBranch1.id != idInBranch1:
        nodeInBranch1 = nodeInBranch1.outputGates[mainPath1].nextNode

    if addBridge:
        bridge = make_bridge(nodeInBranch1.inputs[0])
        path.append(bridge)

    # collect nodes in path1
    path.append(nodeInBranch1)

    while nodeInBranch1.type != BlockType.Output:
        nodeInBranch1 = nodeInBranch1.outputGates[mainPath1].nextNode
        path.append(nodeInBranch1)

    return path



"""
    new path skips some nodes of path
    @param inputNode: input node of path
    @param idOutBranch: id of output node of branch. skip point
    @param idInBranch: id of input node of branch. merge point
    @return: list of nodes in the new path
"""
def collect_new_path_nodes_single_path(inputNode, idOutBranch, idInBranch, addBridge):
    path = [inputNode]

    # nodes must be in either path-main1 or path-main2
    mainPath = inputNode.owner + ':path-main'

    # collect nodes in path1
    nodeOutBranch = inputNode
    while nodeOutBranch.id != idOutBranch:
        nodeOutBranch = nodeOutBranch.outputGates[mainPath].nextNode
        path.append(nodeOutBranch)

    # find inBranch node of branch1
    nodeInBranch = nodeOutBranch.outputGates[mainPath].nextNode
    while nodeInBranch.id != idInBranch:
        nodeInBranch = nodeInBranch.outputGates[mainPath].nextNode

    if addBridge:
        bridge = make_bridge(nodeInBranch.inputs[0])
        path.append(bridge)


    # collect nodes in path1
    path.append(nodeInBranch)

    while nodeInBranch.type != BlockType.Output:
        nodeInBranch = nodeInBranch.outputGates[mainPath].nextNode
        path.append(nodeInBranch)

    return path


def collect_new_path_nodes_host_path(inputNode1, inputNode2, idOutBranch, idInBranch):
    path = [inputNode1]

    # nodes must be in either path-main1 or path-main2
    mainPath2 = inputNode2.owner + ':path-main'
    mainPath1 = inputNode1.owner + ':path-main'

    # collect nodes in path1
    nodeOutBranch = inputNode2
    while nodeOutBranch.id != idOutBranch:
        nodeOutBranch = nodeOutBranch.outputGates[mainPath2].nextNode
        path.append(nodeOutBranch)

    # find inBranch node of branch1
    nodeInBranch = nodeOutBranch.outputGates[mainPath2].nextNode
    while nodeInBranch.id != idInBranch:
        nodeInBranch = nodeInBranch.outputGates[mainPath2].nextNode

    # collect nodes in path1
    path.append(nodeInBranch)

    while nodeInBranch.outputGates[mainPath2].nextNode.type != BlockType.Output:
        nodeInBranch = nodeInBranch.outputGates[mainPath2].nextNode
        path.append(nodeInBranch)

    nodeOutput = inputNode1
    while nodeOutput.type != BlockType.Output:
        nodeOutput = nodeOutput.outputGates[mainPath1].nextNode

    path.append(nodeOutput)

    return path




def collect_new_path_nodes_cross_path_reverse(inputNode1, inputNode2, idOutBranch2, idInBranch1, idOutBranch1,
                                              idInBranch2):

    # collect all nodes in the path
    path = [inputNode1]


    # nodes must be in either path-main1 or path-main2
    mainPath1 = inputNode1.owner + ':path-main'
    mainPath2 = inputNode2.owner + ':path-main'

    # collect nodes in path1
    nodeOutBranch2 = inputNode2
    while nodeOutBranch2.id != idOutBranch2:
        nodeOutBranch2 = nodeOutBranch2.outputGates[mainPath2].nextNode
        path.append(nodeOutBranch2)


    # find inBranch node of branch2
    nodeInBranch1 = inputNode1
    while nodeInBranch1.id != idInBranch1:
        nodeInBranch1 = nodeInBranch1.outputGates[mainPath1].nextNode


    # collect nodes in path2
    path.append(nodeInBranch1)

    nodeOutBranch1 = nodeInBranch1
    while nodeOutBranch1.id != idOutBranch1:
        nodeOutBranch1 = nodeOutBranch1.outputGates[mainPath1].nextNode
        path.append(nodeOutBranch1)


    # find inBranch node of branch1
    nodeInBranch2 = nodeOutBranch2.outputGates[mainPath2].nextNode
    while nodeInBranch2.id != idInBranch2:
        nodeInBranch2 = nodeInBranch2.outputGates[mainPath2].nextNode


    # collect nodes in path1
    path.append(nodeInBranch2)

    while (len(nodeInBranch2.outputGates) != 0) and \
            nodeInBranch2.outputGates[mainPath2].nextNode.type != BlockType.Output:

        nodeInBranch2 = nodeInBranch2.outputGates[mainPath2].nextNode
        path.append(nodeInBranch2)


    while nodeOutBranch1.type != BlockType.Output:
        nodeOutBranch1 = nodeOutBranch1.outputGates[mainPath1].nextNode

    if nodeInBranch2.block == BlockType.Output:
        path[-1] = nodeOutBranch1
    else:
        path.append(nodeOutBranch1)


    for i in range(1, len(path)-1):
        if path[i].owner != inputNode2.owner:
            for b in path[i].block:
                for param in b.parameters():
                    param.requires_grad = True

    return path



def create_new_path(name, App, idOutBranch1, idInBranch1, App2=None, idInBranch2=None,
                    idOutBranch2=None, forward=None, link=None, reverse=False, addBridge=False):

    inputNode1 = App.get_input_node()

    if idInBranch2 is not None:
        inputNode2 = App2.get_input_node()
        if reverse:
            pathNodes = collect_new_path_nodes_cross_path_reverse(inputNode1, inputNode2, idOutBranch1, idInBranch2,
                                                      idOutBranch2, idInBranch1)
        else:
            pathNodes = collect_new_path_nodes_cross_path(inputNode1, inputNode2, idOutBranch1, idInBranch2,
                                                          idOutBranch2, idInBranch1, addBridge)
    else:
        if reverse:
            inputNode2 = App2.get_input_node()
            pathNodes = collect_new_path_nodes_host_path(inputNode1, inputNode2, idOutBranch1, idInBranch1)
        else:
            pathNodes = collect_new_path_nodes_single_path(inputNode1, idOutBranch1, idInBranch1, addBridge)


    path = Path(name, App.name, pathNodes, forward=forward, link=link)

    return path



def make_bridge(node):
    bridge = copy.copy(node)
    bridge.name = 'bridge'
    bridge.id = -1
    bridge.outputGates = {}
    bridge.subscriptions = []
    bridge.type = BlockType.Bridge
    for b in bridge.block:
        for param in b.parameters():
            param.requires_grad = True

    return bridge
