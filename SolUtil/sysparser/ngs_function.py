import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ._array_utils import to_writable_array


def visualize_ngs(G: nx.DiGraph):
    virtual_pipe = [(u, v) for (u, v, d) in G.edges(data=True) if d["type"] == 0]
    real_pipe = [(u, v) for (u, v, d) in G.edges(data=True) if d["type"] == 1]
    s_node = [n for (n, data) in G.nodes(data=True) if data['type'] == 2]
    ns_node = [n for (n, data) in G.nodes(data=True) if data['type'] == 1]

    # Visualize the graph and the minimum spanning tree
    pos = nx.planar_layout(G, scale=100)
    # pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=s_node, node_color='#ff7f0e')
    nx.draw_networkx_nodes(G, pos, nodelist=ns_node, node_color='#17becf')
    nx.draw_networkx_labels(G, pos, font_family="sans-serif")
    nx.draw_networkx_edges(G, pos, edgelist=virtual_pipe, edge_color='grey', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=real_pipe, edge_color='green')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): d["idx"] for u, v, d in G.edges(data=True)},
        verticalalignment='top'
    )
    ax = plt.gca()
    ax.margins(10)
    plt.axis("off")
    plt.show()


def load_ngs(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    gc = dict()

    type_node = to_writable_array(df['node']['type'])
    type_pipe = to_writable_array(df['pipe']['type'])
    idx_node = to_writable_array(df['node']['idx'])
    idx_pipe = to_writable_array(df['pipe']['idx'])
    idx_from = to_writable_array(df['pipe']['fnode'])
    compress_fac = to_writable_array(df['pipe']['Compress factor'])
    gc['compress_fac'] = compress_fac
    idx_to = to_writable_array(df['pipe']['tnode'])
    ns_node_cond = df['node']['type'] == 1
    ns_node = to_writable_array(df['node'][ns_node_cond]['idx'])
    gc['ns_node'] = ns_node
    s_node_cond = df['node']['type'] != 1
    s_node = to_writable_array(df['node'][s_node_cond]['idx'])
    gc['s_node'] = s_node
    slack_cond = df['node']['type'] == 3
    slack_node = to_writable_array(df['node'][slack_cond]['idx'])
    non_slack_source_node = np.setdiff1d(s_node, slack_node)
    non_slack_node = np.setdiff1d(np.arange(len(df['node'])), slack_node)
    gc['slack'] = slack_node
    gc['n_slack'] = len(slack_node)
    gc['non_slack_source'] = non_slack_source_node
    gc['non_slack_node'] = non_slack_node
    n_node = len(df['node'])
    gc['n_node'] = n_node
    gc['n_pipe'] = len(df['pipe'])
    gc['fs'] = to_writable_array(df['node']['fs'], dtype=np.float64)
    gc['fl'] = to_writable_array(df['node']['fl'], dtype=np.float64)
    gc['delta'] = to_writable_array(df['pipe']['delta'])

    # loop detection and conversion
    if 'loop' in df:
        pipe_in_loop = df['loop']['loop1'] == 1
        pinloop = to_writable_array(df['loop'][pipe_in_loop]['idx'])
        gc['pinloop'] = pinloop
    else:
        gc['pinloop'] = []

    lam = to_writable_array(df['pipe']['Friction'])
    D = to_writable_array(df['pipe']['Diameter'])
    Piset = to_writable_array(df['node'][slack_cond]['p'])
    L = to_writable_array(df['pipe']['Length'])
    va = 340
    S = np.pi * (D / 2) ** 2
    C = lam * va ** 2 * L / D / S ** 2 / (1e6 ** 2)
    gc['lam'] = lam
    gc['D'] = D
    gc['Piset'] = Piset
    gc['L'] = L
    gc['va'] = va
    gc['S'] = S
    gc['C'] = C

    G = nx.DiGraph()

    for i in range(len(idx_pipe)):
        G.add_node(idx_from[i], type=type_node[idx_from[i]])
        G.add_node(idx_to[i], type=type_node[idx_to[i]])
        G.add_edge(idx_from[i],
                   idx_to[i],
                   idx=idx_pipe[i],
                   type=type_pipe[i],
                   c=C[i])

    A = nx.incidence_matrix(G,
                            nodelist=idx_node,
                            edgelist=sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)),
                            oriented=True)
    gc['A'] = A
    rpipe_from = []
    rpipe_to = []
    idx_rpipe = []
    pipe_from = []
    pipe_to = []
    idx_pipe = []
    for x, y, z in sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)):
        if z['type'] == 1:
            rpipe_from.append(x)
            rpipe_to.append(y)
            idx_rpipe.append(z['idx'])
        pipe_from.append(x)
        pipe_to.append(y)
        idx_pipe.append(z['idx'])

    gc['rpipe_from'] = rpipe_from
    gc['rpipe_to'] = rpipe_to
    gc['idx_rpipe'] = idx_rpipe
    gc['pipe_from'] = pipe_from
    gc['pipe_to'] = pipe_to
    gc['idx_pipe'] = idx_pipe
    gc['G'] = G

    return gc


def load_GT(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    gtc = dict()

    for column_name in df['GT'].columns:
        gtc[column_name] = to_writable_array(df['GT'][column_name])

    return gtc


def load_P2G(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    p2gc = dict()

    for column_name in df['P2G'].columns:
        p2gc[column_name] = to_writable_array(df['P2G'][column_name])

    return p2gc
