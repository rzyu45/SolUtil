import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csc_array

from ._array_utils import to_writable_array


def load_hs(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    hc = dict()

    type_node = to_writable_array(df['node']['type'])
    idx_node = to_writable_array(df['node']['idx'])
    idx_pipe = to_writable_array(df['pipe']['idx'])
    idx_from = to_writable_array(df['pipe']['fnode'])
    idx_to = to_writable_array(df['pipe']['tnode'])
    I_node_cond = df['node']['type'] == 1
    I_node = to_writable_array(df['node'][I_node_cond]['idx'])
    hc['I_node'] = I_node
    s_node_cond = df['node']['type'] == 0
    s_node = to_writable_array(df['node'][s_node_cond]['idx'])
    hc['s_node'] = s_node
    l_node_cond = df['node']['type'] == 2
    l_node = to_writable_array(df['node'][l_node_cond]['idx'])
    hc['l_node'] = l_node
    if len(l_node) == 0:
        warnings.warn('No DHS load node!')
    slack_node_cond = df['node']['type'] == 3
    slack_node = to_writable_array(df['node'][slack_node_cond]['idx'])
    hc['slack_node'] = slack_node
    n_node = len(df['node'])
    hc['n_node'] = n_node
    n_pipe = len(df['pipe'])
    hc['n_pipe'] = n_pipe
    non_slack_nodes = np.setdiff1d(np.arange(n_node), slack_node)
    hc['non_slack_nodes'] = non_slack_nodes

    if 'delta' in df['node'].columns:
        delta_node = to_writable_array(df['node']['delta'])
    else:
        delta_node = np.zeros(n_node)
    hc['delta_node'] = delta_node

    if 'delta' in df['pipe'].columns:
        delta_pipe = to_writable_array(df['pipe']['delta'])
    else:
        delta_pipe = np.zeros(n_pipe)
    hc['delta_pipe'] = delta_pipe

    # loop detection and conversion
    if 'loop' in df:
        hc['pinloop'] = to_writable_array(df['loop']['loop1'])
    else:
        hc['pinloop'] = np.zeros(n_pipe)

    lam = to_writable_array(df['pipe']['lambda (W/mK)'])
    D = to_writable_array(df['pipe']['D (mm)']) / 1000
    Ts = to_writable_array(df['node']['Ts'])
    Tr = to_writable_array(df['node']['Tr'])
    L = to_writable_array(df['pipe']['L (m)'])
    S = np.pi * (D / 2) ** 2
    m = to_writable_array(df['pipe']['Massflow (kg/s)'])
    hc['m'] = m

    density = 958.4
    g = 10
    mu = 0.294e-6
    epsilon = 1.25e-3

    # calculate K
    # Calculate absolute velocity
    v = np.abs(m) / (density * np.pi * (D ** 2) / 4)

    # Calculate Reynolds number
    Re = v * D / mu

    # Initialize friction factor array
    f = np.zeros(hc['n_pipe'])

    # Indices where Re < 2300
    low_Re_indices = np.where(Re < 2300)[0]
    f[low_Re_indices] = 64. / Re[low_Re_indices]

    # Indices where 2300 <= Re <= 4000
    mid_Re_indices = np.where((Re >= 2300) & (Re <= 4000))[0]
    f[mid_Re_indices] = (((colebrook(4000, epsilon / D[mid_Re_indices]) - 64 / 2300) / (4000 - 2300)) *
                         (Re[mid_Re_indices] - 2300) + 64 / 2300)

    # Indices where Re > 4000
    high_Re_indices = np.where(Re > 4000)[0]
    f[high_Re_indices] = colebrook(Re[high_Re_indices], epsilon / D[high_Re_indices])

    # Calculate K
    K = 8 * L * f / (D ** 5 * density ** 2 * np.pi ** 2 * g)

    hc['K'] = K

    G = nx.DiGraph()

    for i in range(len(idx_pipe)):
        G.add_node(idx_from[i], type=type_node[idx_from[i]])
        G.add_node(idx_to[i], type=type_node[idx_to[i]])
        G.add_edge(idx_from[i],
                   idx_to[i],
                   idx=idx_pipe[i],
                   c=K[i])

    A = nx.incidence_matrix(G,
                            nodelist=idx_node,
                            edgelist=sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)),
                            oriented=True)
    hc['A'] = A
    pipe_from = []
    pipe_to = []
    idx_pipe = []
    for x, y, z in sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)):
        pipe_from.append(x)
        pipe_to.append(y)
        idx_pipe.append(z['idx'])

    hc['pipe_from'] = pipe_from
    hc['pipe_to'] = pipe_to
    hc['idx_pipe'] = idx_pipe
    hc['G'] = G

    hc['lam'] = lam
    hc['D'] = D
    hc['Ts'] = Ts
    hc['Tr'] = Tr
    hc['L'] = L
    hc['S'] = S
    hc['Ta'] = to_writable_array(df['setting']['Ta'])
    hc['Tsource'] = to_writable_array(df['setting']['Tsource'])
    hc['Tload'] = to_writable_array(df['setting']['Tload'])
    hc['phi'] = to_writable_array(df['node']['phi (MW)'], dtype=float)
    if 'Hset' in df['node']:
        hc['Hset'] = to_writable_array(df['node']['Hset'][slack_node])
    else:
        hc['Hset'] = np.zeros(slack_node.shape[0])

    s_slack_node = s_node.tolist() + slack_node.tolist()
    Cs = csc_array((np.ones(len(s_slack_node)), (s_slack_node, s_slack_node)),
                   shape=(hc['n_node'], hc['n_node']))
    Cl = csc_array((np.ones(len(hc['l_node'])), (hc['l_node'], hc['l_node'])),
                   shape=(hc['n_node'], hc['n_node']))
    Ci = csc_array((np.ones(len(hc['I_node'])), (hc['I_node'], hc['I_node'])),
                   shape=(hc['n_node'], hc['n_node']))
    hc['Cs'] = Cs
    hc['Cl'] = Cl
    hc['Ci'] = Ci

    return hc


def colebrook(R, K=None):
    """
    Compute the Darcy-Weisbach friction factor according to the Colebrook formula.

    Parameters:
    R : array_like
        Reynolds' number (should be > 2300).
    K : array_like or None
        Equivalent sand roughness height divided by the hydraulic diameter.
        If None, default value is set to 0.

    Returns:
    F : array_like
        Friction factor.

    Raises:
    ValueError
        If any value in R is non-positive or any value in K is negative.
    """

    # Check for input validity
    if np.any(R <= 0):
        raise ValueError("The Reynolds number must be positive (R>2300).")
    if K is None:
        K = np.zeros_like(R)
    elif np.any(K < 0):
        raise ValueError("The relative sand roughness must be non-negative.")

    # Constants used in initialization
    X1 = K * R * 0.123968186335417556  # X1 ≈ K * R * log(10) / 18.574
    X2 = np.log(R) - 0.779397488455682028  # X2 ≈ log(R*log(10) / 5.02)

    # Initial guess for F
    F = X2 - 0.2  # F ≈ X2 - 1/5

    # First iteration
    E = (np.log(X1 + F) + F - X2) / (1 + X1 + F)
    F = F - (1 + X1 + F + 0.5 * E) * E * (X1 + F) / (1 + X1 + F + E * (1 + E / 3))

    # Second iteration for higher accuracy
    E = (np.log(X1 + F) + F - X2) / (1 + X1 + F)
    F = F - (1 + X1 + F + 0.5 * E) * E * (X1 + F) / (1 + X1 + F + E * (1 + E / 3))

    # Finalizing the solution
    F = 1.151292546497022842 / F  # F ≈ 0.5 * log(10) / F
    F = F ** 2  # Square F to get the friction factor

    return F
