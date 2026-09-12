import os
import warnings

import numpy as np
from numpy import exp, diagflat, abs
import pandas as pd
from scipy.sparse import csc_array, hstack, spdiags
from typing import Union, Dict
from Solverz.solvers.solution import aesol

from ._array_utils import to_writable_array

# MATPOWER bus types
PQ, PV, REF, NONE = 1, 2, 3, 4

# The columns of the ``bus``, ``branch`` and ``gen`` sheets of a SolUtil xlsx case
BUS_COLS = ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va',
            'baseKV', 'zone', 'Vmax', 'Vmin']
BRANCH_COLS = ['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC',
               'ratio', 'angle', 'status', 'angmin', 'angmax']
GEN_COLS = ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax',
            'Pmin', 'Pc1', 'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max',
            'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf']


def Vm_updater(Vm_pv, Vm_pq, Vm_slack):
    return np.concatenate((Vm_pv, Vm_pq, Vm_slack))


def Va_updater(Va_pvpq, Va_slack):
    return np.concatenate((Va_pvpq, Va_slack))


def S_updater(Ybus, Vm_pv, Vm_pq, Vm_slack, Va_pvpq, Va_slack):
    Vm = Vm_updater(Vm_pv, Vm_pq, Vm_slack)
    Va = Va_updater(Va_pvpq, Va_slack)
    V = Vm * exp(1j * Va)
    Ibus = Ybus @ V
    S = V * Ibus.conj()
    return np.column_stack((S.real, S.imag))


def dSbus_dV_updater(Ybus: Union[csc_array, np.ndarray], Vm_pv, Vm_pq, Vm_slack, Va_pvpq, Va_slack):
    Vm = Vm_updater(Vm_pv, Vm_pq, Vm_slack)
    Va = Va_updater(Va_pvpq, Va_slack)
    n = len(Vm)
    V = Vm * exp(1j * Va)
    Ibus = Ybus @ V
    if isinstance(Ybus, csc_array):
        diagV = csc_array((V, (np.arange(0, n), np.arange(0, n))), shape=(n, n))
        diagIbus = csc_array((Ibus, (np.arange(0, n), np.arange(0, n))), shape=(n, n))
        diagVnorm = csc_array((V / abs(V), (np.arange(0, n), np.arange(0, n))), shape=(n, n))
    else:
        diagV = diagflat(V)
        diagIbus = diagflat(Ibus)
        diagVnorm = diagflat(V / abs(V))

    dSbusdVa = 1j * diagV @ (diagIbus - Ybus @ diagV).conj()
    dSbusdVm = diagV @ (Ybus @ diagVnorm).conj() + diagIbus.conj() @ diagVnorm

    if isinstance(Ybus, csc_array):
        return csc_array(hstack([dSbusdVa.real, dSbusdVm.real, dSbusdVa.imag, dSbusdVm.imag],
                                format='csc'))
    else:
        return np.hstack([dSbusdVa.real, dSbusdVm.real, dSbusdVa.imag, dSbusdVm.imag])


def makeYbus(baseMVA: np.ndarray, bus: pd.DataFrame, branch: pd.DataFrame):
    nb = len(bus)
    nl = len(branch)
    stat = np.asarray(branch['status'])
    r = np.asarray(branch['r'])
    x = np.asarray(branch['x'])
    b = np.asarray(branch['b'])
    Ys = stat / (r + 1j * x)
    Bc = stat * b
    tap = np.ones((nl,))
    ratio = np.asarray(branch['ratio'])
    i = np.argwhere(ratio)
    tap[i] = ratio[i]
    angle = np.asarray(branch['angle'])
    tap = tap * np.exp(1j * np.pi / 180 * angle)
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * tap.conj())
    Yft = -Ys / tap.conj()
    Ytf = -Ys / tap

    Ysh = (bus['Gs'] + 1j * bus['Bs']) / baseMVA
    f = np.asarray(branch['fbus'])
    t = np.asarray(branch['tbus'])
    Cf = csc_array((np.ones((nl,)), (np.arange(0, nl), f)),
                   (nl, nb))
    Ct = csc_array((np.ones((nl,)), (np.arange(0, nl), t)),
                   (nl, nb))
    YffD = csc_array((Yff, (np.arange(0, nl), np.arange(0, nl))), (nl, nl))
    YftD = csc_array((Yft, (np.arange(0, nl), np.arange(0, nl))), (nl, nl))
    YtfD = csc_array((Ytf, (np.arange(0, nl), np.arange(0, nl))), (nl, nl))
    YttD = csc_array((Ytt, (np.arange(0, nl), np.arange(0, nl))), (nl, nl))
    Yf = YffD @ Cf + YftD @ Ct
    Yt = YtfD @ Cf + YttD @ Ct
    Ybus = Cf.T @ Yf + Ct.T @ Yt + csc_array((Ysh, (np.arange(0, nb), np.arange(0, nb))), (nb, nb))

    return Ybus


def read_matpower(file_name):
    """Read a MATPOWER case file into the tables of a SolUtil case.

    The file is parsed by matpowercaseframes, and the case is then prepared as
    MATPOWER ``runpf`` prepares it before its Newton iteration, so that
    ``PowerFlow`` solves the problem that MATPOWER solves:

    - generators out of service are dropped;
    - the generators in service at one bus are summed into one row, whose
      voltage setpoint is that of the first of them;
    - a PV or reference bus without a generator in service becomes a PQ bus,
      and if no reference bus remains, the first PV bus becomes the reference
      bus, with a warning, as in MATPOWER's ``bustypes``;
    - every bus with a generator in service starts at that generator's
      voltage setpoint;
    - the buses are numbered 0 to nb-1 in file order, and the bus columns of
      the branch and gen tables are numbered accordingly.

    Branches out of service stay in the branch table with status 0, as in the
    xlsx format. Isolated buses, of type 4, are refused, because MATPOWER
    removes them from the case and the xlsx format has no place for them.

    Returns
    -------
    baseMVA : ndarray of shape (1,)
    bus, branch, gen : pandas.DataFrame
        With the columns of the ``bus``, ``branch`` and ``gen`` sheets of a
        SolUtil xlsx case.
    """
    from matpowercaseframes import CaseFrames

    cf = CaseFrames(os.fspath(file_name))
    bus = cf.bus.reset_index(drop=True)
    gen = cf.gen.reset_index(drop=True)
    br = cf.branch.reset_index(drop=True)
    base_mva = float(np.asarray(cf.baseMVA).ravel()[0])
    nb = len(bus)

    ext_ids = bus['BUS_I'].to_numpy().astype(np.int64)
    if len(np.unique(ext_ids)) != nb:
        raise ValueError(f'{file_name}: duplicate bus numbers')
    btype = bus['BUS_TYPE'].to_numpy().astype(np.int64).copy()
    if np.any(btype == NONE):
        raise ValueError(f'{file_name}: {int(np.sum(btype == NONE))} isolated buses of type 4; '
                         f'remove them, with their branches and generators, before the power flow')
    e2i = pd.Series(np.arange(nb), index=ext_ids)

    def remap(col: pd.Series) -> np.ndarray:
        ids = col.to_numpy().astype(np.int64)
        unknown = ~np.isin(ids, ext_ids)
        if unknown.any():
            raise ValueError(f'{file_name}: {int(unknown.sum())} references to unknown bus numbers')
        return e2i.loc[ids].to_numpy()

    # generators: drop those out of service, sum those of one bus
    on = gen['GEN_STATUS'].to_numpy() > 0
    gen_on = gen.loc[on].reset_index(drop=True)
    gbus = remap(gen_on['GEN_BUS'])
    has_gen = np.zeros(nb, dtype=bool)
    has_gen[gbus] = True

    def bus_sum(col: str) -> np.ndarray:
        acc = np.zeros(nb)
        np.add.at(acc, gbus, gen_on[col].to_numpy(dtype=float))
        return acc

    first = ~pd.Series(gbus).duplicated().to_numpy()
    Vg = np.full(nb, np.nan)
    Vg[gbus[first]] = gen_on['VG'].to_numpy(dtype=float)[first]

    # bus types, as MATPOWER bustypes()
    demoted = ((btype == PV) | (btype == REF)) & ~has_gen
    btype[demoted] = PQ
    if not np.any(btype == REF):
        pv = np.flatnonzero(btype == PV)
        if pv.size == 0:
            raise ValueError(f'{file_name}: no reference bus and no PV bus to make one')
        btype[pv[0]] = REF
        warnings.warn(f'{file_name}: no reference bus with a generator in service; '
                      f'bus {ext_ids[pv[0]]} is promoted to the reference bus', stacklevel=2)

    # voltage start, as MATPOWER runpf
    Vm = bus['VM'].to_numpy(dtype=float).copy()
    Vm[has_gen] = Vg[has_gen]

    bus_out = pd.DataFrame({
        'bus_i': np.arange(nb),
        'type': btype,
        'Pd': bus['PD'].to_numpy(dtype=float),
        'Qd': bus['QD'].to_numpy(dtype=float),
        'Gs': bus['GS'].to_numpy(dtype=float),
        'Bs': bus['BS'].to_numpy(dtype=float),
        'area': bus['BUS_AREA'].to_numpy(),
        'Vm': Vm,
        'Va': bus['VA'].to_numpy(dtype=float),
        'baseKV': bus['BASE_KV'].to_numpy(dtype=float),
        'zone': bus['ZONE'].to_numpy(),
        'Vmax': bus['VMAX'].to_numpy(dtype=float),
        'Vmin': bus['VMIN'].to_numpy(dtype=float),
    })[BUS_COLS]

    branch_out = pd.DataFrame({
        'fbus': remap(br['F_BUS']),
        'tbus': remap(br['T_BUS']),
        'r': br['BR_R'].to_numpy(dtype=float),
        'x': br['BR_X'].to_numpy(dtype=float),
        'b': br['BR_B'].to_numpy(dtype=float),
        'rateA': br['RATE_A'].to_numpy(dtype=float),
        'rateB': br['RATE_B'].to_numpy(dtype=float),
        'rateC': br['RATE_C'].to_numpy(dtype=float),
        'ratio': br['TAP'].to_numpy(dtype=float),
        'angle': br['SHIFT'].to_numpy(dtype=float),
        'status': br['BR_STATUS'].to_numpy().astype(np.int64),
        'angmin': br['ANGMIN'].to_numpy(dtype=float),
        'angmax': br['ANGMAX'].to_numpy(dtype=float),
    })[BRANCH_COLS]

    gen_bus = np.flatnonzero(has_gen)
    gen_out = pd.DataFrame({
        'bus': gen_bus,
        'Pg': bus_sum('PG')[gen_bus],
        'Qg': bus_sum('QG')[gen_bus],
        'Qmax': bus_sum('QMAX')[gen_bus],
        'Qmin': bus_sum('QMIN')[gen_bus],
        'Vg': Vg[gen_bus],
        'mBase': np.full(gen_bus.size, base_mva),
        'status': np.ones(gen_bus.size, dtype=np.int64),
        'Pmax': bus_sum('PMAX')[gen_bus],
        'Pmin': bus_sum('PMIN')[gen_bus],
    })
    for col in GEN_COLS:
        if col not in gen_out:
            gen_out[col] = 0.0
    gen_out = gen_out[GEN_COLS]

    return np.array([base_mva]), bus_out, branch_out, gen_out


def load_mpc(file_name) -> Dict[str, Union[np.ndarray, csc_array]]:
    """Read a power-flow case into the arrays that ``PowerFlow`` uses.

    ``file_name`` is either a MATPOWER ``.m`` case file, which
    :func:`read_matpower` reads and prepares as MATPOWER ``runpf`` does, or a
    SolUtil ``.xlsx`` workbook with the sheets ``setting``, ``bus``,
    ``branch`` and ``gen``, whose buses are numbered 0 to nb-1 in file order
    and whose ``gen`` sheet holds one row per bus with a generator in
    service. Both give the same arrays for the same case. The ``.m`` file is
    the faster one to read, because openpyxl spends seconds parsing the XML
    of a large workbook (rzyu45/SolUtil#5).
    """
    if isinstance(file_name, (str, os.PathLike)) and os.fspath(file_name).lower().endswith('.m'):
        baseMVA, bus, branch, gen = read_matpower(file_name)
    else:
        df = pd.read_excel(file_name,
                           sheet_name=None,
                           engine='openpyxl',
                           index_col=None
                           )
        bus = df['bus']
        branch = df['branch']
        gen = df['gen']
        baseMVA = to_writable_array(df['setting']['baseMVA'])
    return _mpc_from_tables(baseMVA, bus, branch, gen)


def _mpc_from_tables(baseMVA, bus: pd.DataFrame, branch: pd.DataFrame,
                     gen: pd.DataFrame) -> Dict[str, Union[np.ndarray, csc_array]]:
    """The arrays of a case, from its base power and its bus, branch and gen
    tables in the layout of the SolUtil xlsx format."""
    mpc = dict()
    pq = bus['type'] == 1
    pv = bus['type'] == 2
    slack = bus['type'] == 3
    nb = len(bus)
    mpc['baseMVA'] = baseMVA
    idx_pq = to_writable_array(bus[pq]['bus_i'])
    mpc['idx_pq'] = idx_pq
    idx_pv = to_writable_array(bus[pv]['bus_i'])
    mpc['idx_pv'] = idx_pv
    idx_slack = to_writable_array(bus[slack]['bus_i'])
    mpc['idx_slack'] = idx_slack
    Vm = to_writable_array(bus['Vm'], dtype=float)
    mpc['Vm'] = Vm
    Va = np.deg2rad(to_writable_array(bus['Va'], dtype=float))
    mpc['Va'] = Va
    Pd = to_writable_array(bus['Pd'], dtype=float) / baseMVA
    mpc['Pd'] = Pd
    Qd = to_writable_array(bus['Qd'], dtype=float) / baseMVA
    mpc['Qd'] = Qd
    Pg = np.zeros((nb,), dtype=float)
    idx_gen = to_writable_array(gen['bus'])
    Pg[idx_gen] = gen['Pg'] / baseMVA
    mpc['Pg'] = Pg
    Qg = np.zeros((nb,), dtype=float)
    Qg[idx_gen] = gen['Qg'] / baseMVA
    mpc['Qg'] = Qg
    mpc['Ybus'] = makeYbus(baseMVA, bus, branch)
    mpc['nb'] = nb

    return mpc


def bus_injections(Ybus: Union[csc_array, np.ndarray], Vm: np.ndarray, Va: np.ndarray):
    """Net complex bus injections of a voltage profile, vectorized.

    With ``V = Vm * exp(1j * Va)`` the injection is ``S = V * conj(Ybus @ V)``.
    Returns ``(P, Q)`` in per unit. ``P[i]`` is the active power that bus ``i``
    feeds into the network, so at a slack bus ``Pg[i] = P[i] + Pd[i]`` and at
    a slack or PV bus ``Qg[i] = Q[i] + Qd[i]``.

    This replaces the element-wise double loop over ``G[i, j]`` / ``B[i, j]``
    that scaled as ``O(n_gen * n_bus)`` sparse scalar lookups and took hours on
    networks with tens of thousands of buses. One sparse matrix-vector product
    gives every bus at once.
    """
    V = Vm * np.exp(1j * Va)
    S = V * np.conj(Ybus @ V)
    return S.real, S.imag


def parse_data_post_pf(sys: dict, sol: aesol):
    Vm = sys['Vm']
    Va = sys['Va']
    Ybus = sys["Ybus"]
    ref = sys["idx_slack"].tolist()
    pv = sys["idx_pv"].tolist()
    pq = sys["idx_pq"].tolist()
    Pg = sys["Pg"]
    Qg = sys["Qg"]
    Pd = sys["Pd"]
    Qd = sys["Qd"]
    Vm[pq] = sol.y['Vm']
    Va[pv + pq] = sol.y['Va']

    # update slack Pg and slack / PV Qg from the net bus injections
    P, Q = bus_injections(Ybus, Vm, Va)
    Pg[ref] = P[ref] + Pd[ref]
    Qg[ref + pv] = Q[ref + pv] + Qd[ref + pv]

    sys['Vm'] = Vm
    sys['Va'] = Va
    sys['Pg'] = Pg
    sys['Qg'] = Qg
    return sys


def plus_load_impedance(Y, Pd, Qd, Vm):
    Yload = (Pd - 1j * Qd) / (Vm ** 2)
    return Y + spdiags(Yload, 0, *Y.shape)


def load_mac(file_name) -> Dict[str, Union[np.ndarray, csc_array]]:
    mpc = dict()
    df = pd.read_excel(file_name,
                       sheet_name='machine',
                       engine='openpyxl',
                       index_col=None
                       )
    mpc['ra'] = to_writable_array(df['ra'])
    mpc['xd'] = to_writable_array(df['xd'])
    mpc['xdp'] = to_writable_array(df['xdp'])
    mpc['xq'] = to_writable_array(df['xq'])
    mpc['xqp'] = to_writable_array(df['xqp'])
    mpc['D'] = to_writable_array(df['D'])
    mpc['Tj'] = to_writable_array(df['Tj'])
    mpc['Tdp'] = to_writable_array(df['Tdp'])
    mpc['Tqp'] = to_writable_array(df['Tqp'])
    mpc['nm'] = mpc['Tqp'].shape[0]
    mpc['bus'] = to_writable_array(df['bus'])
    return mpc


def load_GT(file_name) -> Dict[str, Union[np.ndarray, csc_array]]:
    mpc = dict()
    df = pd.read_excel(file_name,
                       sheet_name='GT',
                       engine='openpyxl',
                       index_col=None
                       )
    mpc['bus'] = to_writable_array(df['bus'])
    mpc['node'] = to_writable_array(df['node'])

    mpc['ngt'] = mpc['bus'].shape[0]
    
    param_list = ['qmax', 'qmin', 'b', 'c', 'TFS', 'K1', 'K2', 'T1', 'T2', 'kp', 'ki', 'W', 'Y', 'Z', 'kNL', 'TCD', 
                  'Cop', 'A', 'B', 'C', 'D', 'E', 'TRbase', 'TG', 'Tref']
    for param in param_list:
        mpc[param] = to_writable_array(df[param])
    
    return mpc
