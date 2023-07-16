import networkx as nx
import matplotlib.pyplot as plt
from BlockNetwork import BlockType

def scan_topology(inputNodes, visitedNodes):
    for node in inputNodes:
        visitedNodes.add(node)
        if len(node.outputGates) == 0:
            continue
        else:
            for gate in node.outputGates.values():
                scan_topology([gate.nextNode], visitedNodes)

    return visitedNodes


def collect_input_nodes(paths):
    inputNodes = []
    for path in paths:
        inputNodes.append(path.inputNode)
    return inputNodes


def plot_topology(paths):
    colors = ['orange', 'green', 'blue']
    owners = set()
    inputNodes = collect_input_nodes(paths)
    visitedNodes = scan_topology(inputNodes, set())

    for node in visitedNodes:
        owners.add(node.owner)

    owners = list(owners)

    G = nx.DiGraph()
    for node in visitedNodes:
        G.add_node(node.name, label=node.owner, color=colors[owners.index(node.owner)])
        for gate in node.outputGates.values():
            G.add_edge(node.name, gate.nextNode.name)

    pos = {i.name: [i.id*5 - 5*(i.type==BlockType.Input) + 5*(i.type==BlockType.Output), owners.index(i.owner)] for i in visitedNodes}
    labels = {i.name: i.id for i in visitedNodes}

    color_index = {i.name: colors[owners.index(i.owner)] for i in visitedNodes}

    # Draw the graph using matplotlib
    plt.figure(figsize=(8, 3))
    # nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=[color_index[node] for node in G.nodes()])
    nx.draw_networkx_edges(G, pos, width=2.0, edge_color="black", style="solid")
    nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.axvline(x=-2, color='r', linestyle='--')
    plt.text(-3, 0.5, 'input layer', rotation=90, ha='center', va='center')
    plt.text(0, 0.5, 'network layers', rotation=90, ha='center', va='center')
    plt.show()


