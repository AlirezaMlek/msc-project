
"""
    new path contains some nodes of path1 and path2
    @param inputNode2: input node of path2
    @param idInBranch2: id of input node of branch2. new path cross from path1 to path2
    @param idOutBranch2: id of output node of branch2. new path cross from path2 to path1
    @return: list of nodes in the new path
"""
from utils.BlockNode import BlockType


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
                                      idInBranch1, path_main):

    # collect all nodes in the path
    path = [inputNode1]


    # nodes must be in either path-main1 or path-main2
    mainPath1 = path_main
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


    # collect nodes in path1
    path.append(nodeInBranch1)

    while nodeInBranch1.type != BlockType.Output:
        nodeInBranch1 = nodeInBranch1.outputGates[mainPath1].nextNode
        path.append(nodeInBranch1)

    return path



def find_root(node):
    if node.type == BlockType.Input:
        return node
    else:
        return find_root(node.inputs[0])



