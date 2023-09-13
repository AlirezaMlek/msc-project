from utils.BlockNetwork import *


"""
    new path skips some nodes of path
    @param inputNode: input node of path
    @param idOutBranch: id of output node of branch. skip point
    @param idInBranch: id of input node of branch. merge point
    @return: list of nodes in the new path
"""
def collect_new_path_nodes_single_path(inputNode, idOutBranch, idInBranch):
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
        if nodeInBranch.type == BlockType.Output:
            raise AttributeError('invalid block index')
        nodeInBranch = nodeInBranch.outputGates[mainPath].nextNode



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




def create_new_path(name, App, idOutBranch1, idInBranch1, App2=None, link=None):

    inputNode1 = App.get_input_node()
    if App2 is not None:
        inputNode2 = App2.get_input_node()
        pathNodes = collect_new_path_nodes_host_path(inputNode1, inputNode2, idOutBranch1, idInBranch1)
    else:
        pathNodes = collect_new_path_nodes_single_path(inputNode1, idOutBranch1, idInBranch1)


    path = Path(name, App.name, pathNodes, link=link)

    return path
