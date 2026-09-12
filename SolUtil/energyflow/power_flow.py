import os
import tempfile
import sys
import uuid
import warnings

from Solverz import (Var as SolVar, Param as SolParam, Eqn, Model,
                     made_numerical, nr_method, module_printer, sin, cos,
                     Set, LoopEqn, Sum)
from Solverz.solvers.solution import aesol
from SolUtil.sysparser import load_mpc
from SolUtil.sysparser.eps_function import bus_injections
from scipy.sparse import csc_array
import numpy as np
import pandas as pd

__all__ = ["PowerFlow"]


class PowerFlow:
    """
    The electric power flow with Newton-Raphson method.

    Parameters
    ----------
    file : str or path
        The case: a MATPOWER ``.m`` case file, which is read with
        matpowercaseframes and prepared as MATPOWER ``runpf`` prepares it,
        or a SolUtil ``.xlsx`` case file. ``load_mpc`` parses both, and the
        ``.m`` file is much faster to read for large cases.
    mdl : dict or None
        Optional pre-built ``{'mdl': numerical_model, 'y0': Vars}`` to
        skip the symbolic build entirely.
    loopeqn : bool, default True
        Formulate the polar power-flow equations as a small number of
        ``LoopEqn`` blocks over a flat full-bus ``Vm`` / ``Va`` state
        (``P_eqn`` over pv+pq, ``Q_eqn`` over pq, plus ``Vm_pin`` /
        ``Va_pin`` pinning the ref+pv magnitudes and ref angles),
        instead of expanding ``O(nb)`` scalar ``Eqn``s. The two
        formulations are algebraically identical, but the LoopEqn form
        builds and compiles to a handful of vector kernels rather than
        one scalar ``Eqn`` per bus, so ``create_instance`` and the
        Jacobian assembly scale with the number of equation *blocks*
        rather than the number of buses. This is the default since 0.9.0.

        ``LoopEqn`` is only supported by the Numba-JIT module path
        (``module_printer``), not by the inline ``made_numerical`` path,
        so the model is rendered to a temporary directory and imported.
        This pays a one-off Numba compile cost.

        ``loopeqn=False`` selects the legacy per-bus scalar inline build.
        **It is deprecated and will be removed in a future release**; it
        emits a ``DeprecationWarning``. The scalar build is much slower
        to construct and lambdify on non-trivial networks and exists only
        for backward compatibility.
    """
    def __init__(self,
                 file: str,
                 mdl=None,
                 loopeqn: bool = True):

        if not loopeqn and mdl is None:
            warnings.warn(
                "PowerFlow(loopeqn=False) selects the legacy per-bus scalar "
                "inline power-flow build, which is deprecated and will be "
                "removed in a future release. It is significantly slower to "
                "construct and lambdify than the default LoopEqn build. Drop "
                "the loopeqn=False argument to use the LoopEqn formulation.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.loopeqn = loopeqn
        self.Vm = None
        self.Va = None
        self.Pg = None
        self.Pd = None
        self.Qg = None
        self.Qd = None
        self.nb = None
        self.Ybus = None
        self.idx_slack = None
        self.idx_pv = None
        self.idx_pq = None
        self.sol = None
        self.baseMVA = None
        self.U = None
        self.S = None
        self.ux = None
        self.uy = None
        self.ix = None
        self.iy = None

        self.mpc = load_mpc(file)
        self.__dict__.update(self.mpc)
        self.Gbus = self.Ybus.real
        self.Bbus = self.Ybus.imag

        self.run_succeed = False

        if mdl is None:
            if self.loopeqn:
                self.pfmdl, self.y0 = loopeqn_pf_mdl(self)
            else:
                self.pfmdl, self.y0 = inline_pf_mdl(self)
        else:
            self.pfmdl = mdl['mdl']
            self.y0 = mdl['y0']

    def mdlpf(self):
        if self.loopeqn:
            return self.mdlpf_loopeqn()
        return self.mdlpf_scalar()

    def mdlpf_scalar(self):
        Vm = self.Vm
        Va = self.Va
        nb = self.nb
        Ybus = self.Ybus
        G = Ybus.real
        B = Ybus.imag
        ref = self.idx_slack.tolist()
        pv = self.idx_pv.tolist()
        pq = self.idx_pq.tolist()
        Pg = self.Pg
        Qg = self.Qg
        Pd = self.Pd
        Qd = self.Qd

        m = Model()
        m.Va = SolVar('Va', Va[pv + pq])
        m.Vm = SolVar('Vm', Vm[pq])
        m.Pg = SolParam('Pg', Pg)
        m.Qg = SolParam('Qg', Qg)
        m.Pd = SolParam('Pd', Pd)
        m.Qd = SolParam('Qd', Qd)

        def get_Vm(idx):
            if idx in ref + pv:
                return Vm[idx]
            elif idx in pq:
                return m.Vm[pq.index(idx)]

        def get_Va(idx):
            if idx in ref:
                return Va[idx]
            elif idx in pv + pq:
                return m.Va[(pv + pq).index(idx)]

        for i in pv + pq:
            expr = 0
            Vmi = get_Vm(i)
            Vai = get_Va(i)
            for j in range(nb):
                Vmj = get_Vm(j)
                Vaj = get_Va(j)
                expr += Vmi * Vmj * (G[i, j] * cos(Vai - Vaj) + B[i, j] * sin(Vai - Vaj))
            m.__dict__[f'P_eqn_{i}'] = Eqn(f'P_eqn_{i}', expr + m.Pd[i] - m.Pg[i])

        for i in pq:
            expr = 0
            Vmi = get_Vm(i)
            Vai = get_Va(i)
            for j in range(nb):
                Vmj = get_Vm(j)
                Vaj = get_Va(j)
                expr += Vmi * Vmj * (G[i, j] * sin(Vai - Vaj) - B[i, j] * cos(Vai - Vaj))
            m.__dict__[f'Q_eqn_{i}'] = Eqn(f'Q_eqn_{i}', expr + m.Qd[i] - m.Qg[i])

        spf, y0 = m.create_instance()

        return spf, y0

    def mdlpf_loopeqn(self):
        """Polar power flow as LoopEqn blocks over a flat full-bus state.

        ``Vm`` / ``Va`` carry every bus. ``P_eqn`` iterates the pv+pq
        buses and ``Q_eqn`` the pq buses; ``Vm_pin`` / ``Va_pin`` pin
        the ref+pv magnitudes and ref angles to their setpoints. The
        inner ``Sum`` over all buses reads the sparse ``Gbus`` / ``Bbus``
        rows, so each LoopEqn compiles to a single vector kernel rather
        than one scalar Eqn per bus.
        """
        Vm = self.Vm
        Va = self.Va
        nb = self.nb
        Ybus = self.Ybus

        ref = self.idx_slack.tolist()
        pv = self.idx_pv.tolist()
        pq = self.idx_pq.tolist()

        pv_pq_arr = np.array(pv + pq, dtype=int)
        pq_arr = np.array(pq, dtype=int)
        ref_pv_arr = np.array(ref + pv, dtype=int)
        ref_arr = np.array(ref, dtype=int)

        m = Model()
        m.Vm = SolVar('Vm', Vm.copy())
        m.Va = SolVar('Va', Va.copy())

        m.Gbus = SolParam('Gbus', csc_array(Ybus.real), dim=2, sparse=True)
        m.Bbus = SolParam('Bbus', csc_array(Ybus.imag), dim=2, sparse=True)
        m.Pg = SolParam('Pg', self.Pg)
        m.Qg = SolParam('Qg', self.Qg)
        m.Pd = SolParam('Pd', self.Pd)
        m.Qd = SolParam('Qd', self.Qd)

        m.Bus = Set('Bus', nb)
        m.PVPQ = Set('PVPQ', pv_pq_arr)
        m.PQ = Set('PQ', pq_arr)
        m.RefPV = Set('RefPV', ref_pv_arr)
        m.Ref = Set('Ref', ref_arr)

        m.Vm_pinned = SolParam('Vm_pinned', Vm[ref_pv_arr])
        m.Va_pinned = SolParam('Va_pinned', Va[ref_arr])

        i_p = m.PVPQ.idx('i_p')
        i_q = m.PQ.idx('i_q')
        j = m.Bus.idx('j')

        body_P = (
            m.Vm[i_p] * Sum(m.Vm[j] * m.Gbus[i_p, j]
                            * cos(m.Va[i_p] - m.Va[j]), j)
            + m.Vm[i_p] * Sum(m.Vm[j] * m.Bbus[i_p, j]
                              * sin(m.Va[i_p] - m.Va[j]), j)
            + m.Pd[i_p] - m.Pg[i_p]
        )
        m.P_eqn = LoopEqn('P_eqn', outer_index=i_p, body=body_P, model=m)

        body_Q = (
            m.Vm[i_q] * Sum(m.Vm[j] * m.Gbus[i_q, j]
                            * sin(m.Va[i_q] - m.Va[j]), j)
            - m.Vm[i_q] * Sum(m.Vm[j] * m.Bbus[i_q, j]
                              * cos(m.Va[i_q] - m.Va[j]), j)
            + m.Qd[i_q] - m.Qg[i_q]
        )
        m.Q_eqn = LoopEqn('Q_eqn', outer_index=i_q, body=body_Q, model=m)

        i_vp = m.RefPV.idx('i_vp')
        i_vr = m.Ref.idx('i_vr')
        m.Vm_pin = LoopEqn('Vm_pin', outer_index=i_vp,
                           body=m.Vm[i_vp] - m.Vm_pinned[i_vp], model=m)
        m.Va_pin = LoopEqn('Va_pin', outer_index=i_vr,
                           body=m.Va[i_vr] - m.Va_pinned[i_vr], model=m)

        spf, y0 = m.create_instance()

        return spf, y0

    def run(self):
        self.pfmdl.p['Pg'] = self.Pg
        self.pfmdl.p['Qg'] = self.Qg
        self.pfmdl.p['Pd'] = self.Pd
        self.pfmdl.p['Qd'] = self.Qd
        self.sol = nr_method(self.pfmdl, self.y0)
        if self.sol.stats.succeed:
            self.run_succeed = True
        self.parse_data_post_pf(self.sol)

    def parse_data_post_pf(self, sol: aesol):
        Vm = self.Vm
        Va = self.Va
        Ybus = self.Ybus
        ref = self.idx_slack.tolist()
        pv = self.idx_pv.tolist()
        pq = self.idx_pq.tolist()
        Pg = self.Pg
        Qg = self.Qg
        Pd = self.Pd
        Qd = self.Qd
        if self.loopeqn:
            # full-bus state; ref / pv pinned by the pin LoopEqns.
            Vm = np.asarray(sol.y['Vm']).copy()
            Va = np.asarray(sol.y['Va']).copy()
        else:
            Vm[pq] = sol.y['Vm']
            Va[pv + pq] = sol.y['Va']

        # update slack Pg and slack / PV Qg from the net bus injections
        # S = V conj(Ybus V); one sparse matrix-vector product instead of
        # an O(n_gen * n_bus) double loop of sparse scalar lookups.
        P, Q = bus_injections(Ybus, Vm, Va)
        Pg[ref] = P[ref] + Pd[ref]
        Qg[ref + pv] = Q[ref + pv] + Qd[ref + pv]

        self.Vm = Vm
        self.Va = Va
        self.Pg = Pg
        self.Qg = Qg

        self.U: np.ndarray = self.Vm * np.exp(1j * self.Va)
        self.S: np.ndarray = (self.Pg - self.Pd) + 1j * (self.Qg - self.Qd)
        I = (self.S / self.U).conjugate()
        self.ux = self.U.real
        self.uy = self.U.imag
        self.ix = I.real
        self.iy = I.imag


def generate_pf_module(pf: PowerFlow, module_name, jit=True, directory=None):
    spf, y0 = pf.mdlpf()
    pyprinter = module_printer(spf,
                               y0,
                               module_name,
                               directory=directory,
                               jit=jit)
    pyprinter.render()


def inline_pf_mdl(pf: PowerFlow):
    spf, y0 = pf.mdlpf_scalar()
    npf = made_numerical(spf, y0, sparse=True)
    return npf, y0


def loopeqn_pf_mdl(pf: PowerFlow):
    """Build and import a LoopEqn power-flow module.

    ``LoopEqn`` is only supported by the module printer path, so the
    model is rendered to a temporary directory and imported back. The
    returned ``mdl`` exposes the usual ``F`` / ``J`` / ``p`` interface
    so ``PowerFlow.run`` works unchanged.
    """
    spf, y0 = pf.mdlpf_loopeqn()
    tmpdir = tempfile.mkdtemp(prefix='solutil_pf_loopeqn_')
    # One module name per PowerFlow instance. A fixed name made every later
    # PowerFlow() in the same process reuse the first rendered module: the
    # package was found in sys.modules and reloaded, but reload() only
    # re-executes __init__.py, and the cached num_func / dependency
    # submodules kept the first case's F, J, p and y.
    module_name = f'pf_loopeqn_mdl_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, module_name, directory=tmpdir, jit=True).render()
    if tmpdir not in sys.path:
        sys.path.insert(0, tmpdir)
    import importlib
    mod = importlib.import_module(module_name)
    return mod.mdl, mod.y
