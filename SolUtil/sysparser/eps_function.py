import numpy as np
from numpy import exp, diagflat, abs
import pandas as pd
from scipy.sparse import csc_array, hstack, spdiags
from typing import Union, Dict
from Solverz.solvers.solution import aesol


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


def load_mpc(file_name) -> Dict[str, Union[np.ndarray, csc_array]]:
    mpc = dict()
    df = pd.read_excel(file_name,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    bus = df['bus']
    branch = df['branch']
    gen = df['gen']

    pq = bus['type'] == 1
    pv = bus['type'] == 2
    slack = bus['type'] == 3
    nb = len(bus)
    baseMVA = np.asarray(df['setting']['baseMVA'])
    mpc['baseMVA'] = baseMVA
    idx_pq = np.asarray(bus[pq]['bus_i'])
    mpc['idx_pq'] = idx_pq
    idx_pv = np.asarray(bus[pv]['bus_i'])
    mpc['idx_pv'] = idx_pv
    idx_slack = np.asarray(bus[slack]['bus_i'])
    mpc['idx_slack'] = idx_slack
    Vm = np.asarray(bus['Vm'], dtype=float)
    mpc['Vm'] = Vm
    Va = np.deg2rad(np.asarray(bus['Va'], dtype=float))
    mpc['Va'] = Va
    Pd = np.asarray(bus['Pd'], dtype=float) / baseMVA
    mpc['Pd'] = Pd
    Qd = np.asarray(bus['Qd'], dtype=float) / baseMVA
    mpc['Qd'] = Qd
    Pg = np.zeros((nb,), dtype=float)
    idx_gen = np.asarray(gen['bus'])
    Pg[idx_gen] = gen['Pg'] / baseMVA
    mpc['Pg'] = Pg
    Qg = np.zeros((nb,), dtype=float)
    Qg[idx_gen] = gen['Qg'] / baseMVA
    mpc['Qg'] = Qg
    mpc['Ybus'] = makeYbus(baseMVA, bus, branch)
    mpc['nb'] = nb

    return mpc


def parse_data_post_pf(sys: dict, sol: aesol):
    nb = sys['nb']
    Vm = sys['Vm']
    Va = sys['Va']
    Ybus = sys["Ybus"]
    G = Ybus.real
    B = Ybus.imag
    ref = sys["idx_slack"].tolist()
    pv = sys["idx_pv"].tolist()
    pq = sys["idx_pq"].tolist()
    Pg = sys["Pg"]
    Qg = sys["Qg"]
    Pd = sys["Pd"]
    Qd = sys["Qd"]
    npv = len(pv)
    npq = len(pq)
    Vm[pq] = sol.y['Vm']
    Va[pv + pq] = sol.y['Va']

    # update slack pg qg

    for i in ref:
        Pinj = 0
        Vmi = Vm[i]
        Vai = Va[i]
        for j in range(nb):
            Vmj = Vm[j]
            Vaj = Va[j]
            Pinj += Vmi * Vmj * (G[i, j] * np.cos(Vai - Vaj) + B[i, j] * np.sin(Vai - Vaj))
        Pg[i] = Pinj + Pd[i]

    for i in ref+pv:
        Qinj = 0
        Vmi = Vm[i]
        Vai = Va[i]
        for j in range(nb):
            Vmj = Vm[j]
            Vaj = Va[j]
            Qinj += Vmi * Vmj * (G[i, j] * np.sin(Vai - Vaj) - B[i, j] * np.cos(Vai - Vaj))
        Qg[i] = Qinj + Qd[i]

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
    mpc['ra'] = np.asarray(df['ra'])
    mpc['xd'] = np.asarray(df['xd'])
    mpc['xdp'] = np.asarray(df['xdp'])
    mpc['xq'] = np.asarray(df['xq'])
    mpc['xqp'] = np.asarray(df['xqp'])
    mpc['D'] = np.asarray(df['D'])
    mpc['Tj'] = np.asarray(df['Tj'])
    mpc['Tdp'] = np.asarray(df['Tdp'])
    mpc['Tqp'] = np.asarray(df['Tqp'])
    mpc['nm'] = mpc['Tqp'].shape[0]
    mpc['bus'] = np.asarray(df['bus'])
    return mpc


def load_GT(file_name) -> Dict[str, Union[np.ndarray, csc_array]]:
    mpc = dict()
    df = pd.read_excel(file_name,
                       sheet_name='GT',
                       engine='openpyxl',
                       index_col=None
                       )
    mpc['bus'] = np.asarray(df['bus'])
    mpc['node'] = np.asarray(df['node'])

    mpc['ngt'] = mpc['bus'].shape[0]
    
    param_list = ['qmax', 'qmin', 'b', 'c', 'TFS', 'K1', 'K2', 'T1', 'T2', 'kp', 'ki', 'W', 'Y', 'Z', 'kNL', 'TCD', 
                  'Cop', 'A', 'B', 'C', 'D', 'E', 'TRbase', 'TG', 'Tref']
    for param in param_list:
        mpc[param] = np.asarray(df[param])
    
    return mpc
