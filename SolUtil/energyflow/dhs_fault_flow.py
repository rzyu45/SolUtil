from .dhs_flow import DhsFlow
from .hydraulic_mdl import HydraFlow
from Solverz.num_api.Array import Array
from Solverz import Var as SolVar, Param as SolParam, Model, heaviside, Abs, Eqn, exp, made_numerical, nr_method, Sign, \
    Opt
import networkx as nx
import numpy as np
import copy


# %%
class DhsFaultFlow:

    def __init__(self,
                 df: DhsFlow,
                 fault_pipe,
                 fault_location,
                 fault_sys='s',
                 dH=0):
        """

        :param df:
        :param fault_pipe: 故障管道在正常网络中的编号
        :param fault_location: 始端节点到故障位置的距离占管道总长度的百分比
        :param fault_sys: 故障在供水网络还是回水网络
        """
        self.yf0 = None
        self.smdl_full = None
        self.nfault = None
        if not df.run_succeed:
            df.run()

        self.HydraSup: HydraFlow = df.HydraFlow
        fs = self.HydraSup.fl.copy()
        fl = self.HydraSup.fs.copy()
        fl[df.slack_node] = 0
        self.HydraRet: HydraFlow = HydraFlow(df.slack_node,
                                             df.non_slack_nodes,
                                             self.HydraSup.c,
                                             fs,
                                             fl,
                                             self.HydraSup.Hset - dH,
                                             -self.HydraSup.delta,
                                             [1],
                                             self.HydraSup.G.reverse(),
                                             2)

        self.dH = dH

        self.df = df
        self.Hset = df.Hset
        self.fault_sys = fault_sys
        self.fault_location = fault_location
        self.fault_pipe = fault_pipe
        self.G = df.G

        # run power flow considering leakage
        self.HydraFault_mdl()
        self.fs_injection = None
        self.stemp = None
        self.temp_mdl = None
        self.mdl_full = None
        self.y0 = None
        self.HydraSup.run()
        self.HydraRet.run()
        self.run_succeed = False
        self.mdl_temp()

    def run_hydraulic(self):

        self.HydraSup.run()
        self.HydraRet.run()
        self.fs_injection = self.HydraSup.f[-1] + self.HydraRet.f[-1]

    def run(self):

        self.run_succeed = False
        done = False
        Tamb = self.df.Ta
        Ts = self.y0['Ts'].copy()
        Tr = self.y0['Tr'].copy()
        Touts = self.y0['Touts'].copy()
        Toutr = self.y0['Toutr'].copy()
        Tsource = self.df.Tsource * np.ones(self.HydraSup.G.number_of_nodes() - 1) - Tamb
        Tload = self.df.Tload * np.ones(self.HydraSup.G.number_of_nodes() - 1) - Tamb
        min_slack_0 = self.df.minset[self.df.slack_node]

        nt = 0
        while not done:
            nt += 1
            phi = self.df.phi
            # the last node is the leak node
            dT = np.sum(self.df.Cs, axis=0) * Tsource[:-1] + (self.df.Cl + self.df.Ci) @ Ts[:-1] \
                 - (self.df.Ci + self.df.Cs) @ Tr[:-1] - np.sum(self.df.Cl, axis=0) * Tload[:-1]
            minset = np.append(phi * 1e6 / (4182 * dT), [0])

            fs = np.zeros(self.df.n_node + 1, dtype=np.float64)
            for i in self.df.s_node.tolist():
                fs[i] = minset[i]
            fl = np.zeros(self.df.n_node + 1, dtype=np.float64)
            for i in self.df.I_node.tolist() + self.df.l_node.tolist():
                fl[i] = minset[i]

            self.HydraSup.update_fs(fs)
            self.HydraSup.update_fl(fl)
            self.HydraRet.update_fs(fl)
            self.HydraRet.update_fl(fs)

            self.run_hydraulic()
            self.temp_mdl.p['m_leak'] = np.array([self.HydraSup.f[-1], self.HydraRet.f[-1]])

            ms = self.HydraSup.f[0:self.df.n_pipe + 1]
            mr = self.HydraRet.f[0:self.df.n_pipe + 1]
            self.temp_mdl.p['ms'] = ms
            self.temp_mdl.p['mr'] = mr

            minset[self.df.slack_node] = self.HydraSup.fs[self.HydraSup.slack_nodes][0]

            self.temp_mdl.p['min'] = minset

            sol = nr_method(self.temp_mdl, self.y0)

            if not sol.stats.succeed:
                print("Temperature not found")
                break

            Ts = sol.y['Ts']
            Tr = sol.y['Tr']
            Touts = sol.y['Touts']
            Toutr = sol.y['Toutr']
            self.y0.array[:] = sol.y.array[:]

            phi_slack = (4182 * abs(minset[self.df.slack_node]) *
                         (Tsource[self.df.slack_node] - Tr[self.df.slack_node]) / 1e6)

            dF = np.abs(minset[self.df.slack_node] - min_slack_0)

            if dF < 1e-5:
                done = True
                self.run_succeed = True
            if nt > 100:
                done = True

            min_slack_0 = minset[self.df.slack_node]

        if self.run_succeed:
            self.Ts = Ts + Tamb
            self.Tr = Tr + Tamb
            self.Touts = Touts + Tamb
            self.Toutr = Toutr + Tamb
            self.ms = ms
            self.mr = mr
            self.minset = minset
            self.phi = np.append(phi, [0])
            self.phi[self.df.slack_node] = phi_slack
            self.phi_slack = phi_slack
            self.Touts = Touts + self.df.Ta
            self.Toutr = Toutr + self.df.Ta
            self.Hs = self.HydraSup.H[:-1]
            self.Hr = self.HydraRet.H[:-1]
            self.m_leak = np.array([self.HydraSup.f[-1], self.HydraRet.f[-1]])
        else:
            print("Solution not found")

    def HydraFault_mdl(self):

        # supply network
        Gs, c, nenviron, nfault = self.add_fault_node(copy.deepcopy(self.HydraSup.G), sys='s')
        fs = self.HydraSup.fs  # assign fs and fl of node nfault and nenviron
        fl = self.HydraSup.fl
        fs = np.append(fs, [0, 0])
        fl = np.append(fl, [0, 0])
        if self.fault_sys == 's':
            Hslack = np.append(self.HydraSup.Hset, [0])
            slack_node = [*self.df.slack_node, nenviron]
            non_slack_node = np.setdiff1d(np.arange(Gs.number_of_nodes()), slack_node)
        else:
            Hslack = self.HydraSup.Hset
            slack_node = self.df.slack_node
            non_slack_node = np.setdiff1d(np.arange(Gs.number_of_nodes()), slack_node)
        # shared
        delta = np.zeros(Gs.number_of_nodes())
        self.HydraSup = HydraFlow(slack_node,
                                  non_slack_node,
                                  c,
                                  fs,
                                  fl,
                                  Hslack,
                                  delta,
                                  [1],
                                  Gs,
                                  2)

        # return network
        Gr, c, nenviron, nfault = self.add_fault_node(copy.deepcopy(self.HydraRet.G), sys='r')
        fs = self.HydraRet.fs  # assign fs and fl of node nfault and nenviron
        fl = self.HydraRet.fl
        fs = np.append(fs, [0, 0])
        fl = np.append(fl, [0, 0])
        if self.fault_sys == 'r':
            Hslack = np.append(self.HydraRet.Hset, [0])
            slack_node = [*self.df.slack_node, nenviron]
            non_slack_node = np.setdiff1d(np.arange(Gs.number_of_nodes()), slack_node)
        else:
            Hslack = self.HydraRet.Hset
            slack_node = self.df.slack_node
            non_slack_node = np.setdiff1d(np.arange(Gs.number_of_nodes()), slack_node)
        # shared
        delta = np.zeros(Gr.number_of_nodes())
        self.HydraRet = HydraFlow(slack_node,
                                  non_slack_node,
                                  c,
                                  fs,
                                  fl,
                                  Hslack,
                                  delta,
                                  [1],
                                  Gr,
                                  2)

        self.nfault = nfault

    def add_fault_node(self, G, sys):

        nfault = G.number_of_nodes()
        for u, v, data in G.edges(data=True):
            if data.get('idx') == self.fault_pipe:
                edge_to_remove = [u, v, data]

        u, v, data = edge_to_remove
        c = data['c']
        G.remove_edge(u, v)

        if sys == 's':
            f = u
            t = nfault
            G.add_edge(f,
                       t,
                       idx=self.fault_pipe,
                       c=c * self.fault_location)

            f = nfault
            t = v
            G.add_edge(f,
                       t,
                       idx=G.number_of_edges(),
                       c=c * (1 - self.fault_location))
        elif sys == 'r':
            f = nfault
            t = v
            G.add_edge(f,
                       t,
                       idx=self.fault_pipe,
                       c=c * self.fault_location)

            f = u
            t = nfault
            G.add_edge(f,
                       t,
                       idx=G.number_of_edges(),
                       c=c * (1 - self.fault_location))
        else:
            raise ValueError(f'Unknown sys type {sys}')

        nenviron = G.number_of_nodes()
        g = 10
        G.add_edge(nfault,
                   nenviron,
                   idx=G.number_of_edges(),
                   c=1 / (2 * g * self.df.S[self.fault_pipe] ** 2))

        c = [np.array(k['c']).reshape(-1) for i, j, k in
             sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1))]
        c = np.asarray(c).reshape(-1)

        return G, c, nenviron, nfault

    def mdl_temp(self):
        """
        Temperature model based on Solverz, with mass flow as parameters
        """
        m = Model()
        Tamb = self.df.Ta

        m.ms = SolParam('ms', self.HydraSup.f[0: self.df.n_pipe + 1])
        m.mr = SolParam('mr', self.HydraRet.f[0: self.df.n_pipe + 1])
        m_leak = np.zeros(2)
        m_leak[0] = self.HydraSup.f[-1]
        m_leak[1] = self.HydraRet.f[-1]
        m.m_leak = SolParam('m_leak', m_leak)
        Ts = self.df.Ts - Tamb
        Ts = np.append(Ts, [self.df.Tsource - Tamb])  # add Ts of leak node
        Tr = self.df.Tr - Tamb
        Tr = np.append(Tr, [self.df.Tload - Tamb])  # add Tr of leak node
        m.Ts = SolVar('Ts', Ts)
        m.Tr = SolVar('Tr', Tr)
        Touts = self.df.Touts - Tamb
        Touts = np.append(Touts, [Ts[-1]])
        Toutr = self.df.Toutr - Tamb
        Toutr = np.append(Toutr, [Tr[-1]])
        m.Touts = SolVar('Touts', Touts)
        m.Toutr = SolVar('Toutr', Toutr)
        min_leak = np.zeros(2)
        minset = np.append(self.df.minset, min_leak)
        m.min = SolParam('min', minset)
        Tsource = self.df.Tsource * np.ones(self.HydraSup.G.number_of_nodes() - 1) - Tamb
        m.Tsource = SolParam('Tsource', Tsource)
        Tload = self.df.Tload * np.ones(self.HydraSup.G.number_of_nodes() - 1) - Tamb
        m.Tload = SolParam('Tload', Tload)
        lam = self.df.lam
        lam = np.append(lam, self.df.lam[self.fault_pipe])
        m.lam = SolParam('lam', lam)
        Ls = np.append(self.df.L, [0])
        Ls[self.fault_pipe] = self.df.L[self.fault_pipe] * self.fault_location
        Ls[-1] = self.df.L[self.fault_pipe] * (1 - self.fault_location)
        m.Ls = SolParam('Ls', Ls)
        Lr = np.append(self.df.L, [0])
        Lr[self.fault_pipe] = self.df.L[self.fault_pipe] * (1 - self.fault_location)
        Lr[-1] = self.df.L[self.fault_pipe] * self.fault_location
        m.Lr = SolParam('Lr', Lr)
        m.Cp = SolParam('Cp', 4182)

        # Supply temperature
        for node in range(self.df.n_node + 1):
            # skip the leak node
            lhs = 0
            rhs = 0

            if node in self.df.s_node.tolist() + self.df.slack_node.tolist():
                lhs += Abs(m.min[node])
                rhs += m.Tsource[node] * Abs(m.min[node])

            for edge in self.HydraSup.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += heaviside(m.ms[pipe]) * Abs(m.ms[pipe])  #
                rhs += heaviside(m.ms[pipe]) * (m.Touts[pipe] * Abs(m.ms[pipe]))  #

            for edge in self.HydraSup.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                if pipe != self.df.n_pipe + 1:
                    lhs += (1 - heaviside(m.ms[pipe])) * Abs(m.ms[pipe])  #
                    rhs += (1 - heaviside(m.ms[pipe])) * (m.Touts[pipe] * Abs(m.ms[pipe]))  #

            lhs *= m.Ts[node]

            m.__dict__[f"Ts_{node}"] = Eqn(f"Ts_{node}", lhs - rhs)

        # Return temperature
        for node in range(self.df.n_node + 1):
            # skip the leak node
            lhs = 0
            rhs = 0

            if node in self.df.l_node:
                lhs += Abs(m.min[node])
                rhs += m.Tload[node] * Abs(m.min[node])

            for edge in self.HydraRet.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += heaviside(m.mr[pipe]) * Abs(m.mr[pipe])  #
                rhs += heaviside(m.mr[pipe]) * (m.Toutr[pipe] * Abs(m.mr[pipe]))  #

            for edge in self.HydraRet.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                if pipe != self.df.n_pipe + 1:
                    lhs += (1 - heaviside(m.mr[pipe])) * Abs(m.mr[pipe])  #
                    rhs += (1 - heaviside(m.mr[pipe])) * (m.Toutr[pipe] * Abs(m.mr[pipe]))  #

            lhs *= m.Tr[node]

            m.__dict__[f"Tr_{node}"] = Eqn(f"Tr_{node}", lhs - rhs)

        # Temperature drop
        for edge in self.HydraSup.G.edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            pipe = edge[2]['idx']
            if pipe != self.df.n_pipe + 1:  # DISCARD THE PIPE FROM LEAKAGE TO THE ENVIRONMENT
                attenuation = exp(- m.lam[pipe] * m.Ls[pipe] / (m.Cp * Abs(m.ms[pipe])))
                Tstart = m.Ts[fnode] * heaviside(m.ms[pipe]) + m.Ts[tnode] * (1 - heaviside(m.ms[pipe]))  #
                rhs = m.Touts[pipe] - Tstart * attenuation
                m.__dict__[f"Touts_{pipe}"] = Eqn(f"Touts_{pipe}", rhs)

                attenuation = exp(- m.lam[pipe] * m.Lr[pipe] / (m.Cp * Abs(m.mr[pipe])))
                Tstart = m.Tr[tnode] * heaviside(m.mr[pipe]) + m.Tr[fnode] * (1 - heaviside(m.mr[pipe]))  #
                rhs = m.Toutr[pipe] - Tstart * attenuation
                m.__dict__[f"Toutr_{pipe}"] = Eqn(f"Toutr_{pipe}", rhs)

        stemp, y0 = m.create_instance()
        self.stemp = stemp
        temp = made_numerical(stemp, y0, sparse=True)
        self.temp_mdl = temp
        self.y0 = y0

    def mdl_full_dhspf(self):
        """
        Temperature-hydraulic model based on Solverz, with mass flow as parameters
        """
        m = Model()
        Tamb = self.df.Ta

        m.ms = SolVar('ms', self.HydraSup.f[0: self.df.n_pipe + 1])
        m.mr = SolVar('mr', self.HydraRet.f[0: self.df.n_pipe + 1])
        m.K = SolParam('K', self.HydraSup.c)
        m_leak = np.zeros(2)
        m_leak[0] = self.HydraSup.f[-1]
        m_leak[1] = self.HydraRet.f[-1]
        m.m_leak = SolVar('m_leak', m_leak)
        m.S_leak = SolParam('S_leak', self.df.S[self.fault_pipe])
        m.g = SolParam('g', 10)
        m.Hs = SolVar('Hs', self.HydraSup.H[:-1])
        m.Hr = SolVar('Hr', self.HydraRet.H[:-1])
        m.Hset_s = SolParam('Hset_s', self.HydraSup.Hset[0])
        m.Hset_r = SolParam('Hset_r', self.HydraSup.Hset[0] - self.dH)
        m.fs_injection = SolVar('fs_injection', self.HydraSup.f[-1] + self.HydraRet.f[-1])
        m.phi_slack = SolVar('phi_slack', np.zeros(1))
        m.phi = SolParam('phi', self.df.phi)
        Ts = self.df.Ts - Tamb
        Ts = np.append(Ts, [self.df.Tsource - Tamb])  # add Ts of leak node
        Tr = self.df.Tr - Tamb
        Tr = np.append(Tr, [self.df.Tload - Tamb])  # add Tr of leak node
        m.Ts = SolVar('Ts', Ts)
        m.Tr = SolVar('Tr', Tr)
        Touts = self.df.Touts - Tamb
        Touts = np.append(Touts, [Ts[-1]])
        Toutr = self.df.Toutr - Tamb
        Toutr = np.append(Toutr, [Tr[-1]])
        m.Touts = SolVar('Touts', Touts)
        m.Toutr = SolVar('Toutr', Toutr)
        m.min = SolVar('min', self.df.minset)
        Tsource = self.df.Tsource * np.ones(self.HydraSup.G.number_of_nodes() - 1) - Tamb
        m.Tsource = SolParam('Tsource', Tsource)
        Tload = self.df.Tload * np.ones(self.HydraSup.G.number_of_nodes() - 1) - Tamb
        m.Tload = SolParam('Tload', Tload)
        lam = self.df.lam
        lam = np.append(lam, self.df.lam[self.fault_pipe])
        m.lam = SolParam('lam', lam)
        Ls = np.append(self.df.L, [0])
        Ls[self.fault_pipe] = self.df.L[self.fault_pipe] * self.fault_location
        Ls[-1] = self.df.L[self.fault_pipe] * (1 - self.fault_location)
        m.Ls = SolParam('Ls', Ls)
        Lr = np.append(self.df.L, [0])
        Lr[self.fault_pipe] = self.df.L[self.fault_pipe] * (1 - self.fault_location)
        Lr[-1] = self.df.L[self.fault_pipe] * self.fault_location
        m.Lr = SolParam('Lr', Lr)
        m.Cp = SolParam('Cp', 4182)

        # Supply temperature
        for node in range(self.df.n_node + 1):
            # skip the leak node
            lhs = 0
            rhs = 0

            if node in self.df.s_node.tolist() + self.df.slack_node.tolist():
                lhs += Abs(m.min[node])
                rhs += m.Tsource[node] * Abs(m.min[node])

            for edge in self.HydraSup.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += heaviside(m.ms[pipe]) * Abs(m.ms[pipe])  #
                rhs += heaviside(m.ms[pipe]) * (m.Touts[pipe] * Abs(m.ms[pipe]))  #

            for edge in self.HydraSup.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                # skip the virtual leak pipe (from leakage position to the environment), since its direction
                # cannot be reversed.
                if pipe != self.df.n_pipe + 1:
                    lhs += (1 - heaviside(m.ms[pipe])) * Abs(m.ms[pipe])  #
                    rhs += (1 - heaviside(m.ms[pipe])) * (m.Touts[pipe] * Abs(m.ms[pipe]))  #

            lhs *= m.Ts[node]

            m.__dict__[f"Ts_{node}"] = Eqn(f"Ts_{node}", lhs - rhs)

        # Return temperature
        for node in range(self.df.n_node + 1):
            # skip the leak node
            lhs = 0
            rhs = 0

            if node in self.df.l_node:
                lhs += Abs(m.min[node])
                rhs += m.Tload[node] * Abs(m.min[node])

            for edge in self.HydraRet.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += heaviside(m.mr[pipe]) * Abs(m.mr[pipe])  #
                rhs += heaviside(m.mr[pipe]) * (m.Toutr[pipe] * Abs(m.mr[pipe]))  #

            for edge in self.HydraRet.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                # skip the virtual leak pipe (from leakage position to the environment), since its direction
                # cannot be reversed.
                if pipe != self.df.n_pipe + 1:
                    lhs += (1 - heaviside(m.mr[pipe])) * Abs(m.mr[pipe])  #
                    rhs += (1 - heaviside(m.mr[pipe])) * (m.Toutr[pipe] * Abs(m.mr[pipe]))  #

            lhs *= m.Tr[node]

            m.__dict__[f"Tr_{node}"] = Eqn(f"Tr_{node}", lhs - rhs)

        # Temperature drop
        for edge in self.HydraSup.G.edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            pipe = edge[2]['idx']
            if pipe != self.df.n_pipe + 1:  # DISCARD THE PIPE FROM LEAKAGE TO THE ENVIRONMENT
                attenuation = exp(- m.lam[pipe] * m.Ls[pipe] / (m.Cp * Abs(m.ms[pipe])))
                Tstart = m.Ts[fnode] * heaviside(m.ms[pipe]) + m.Ts[tnode] * (1 - heaviside(m.ms[pipe]))  #
                rhs = m.Touts[pipe] - Tstart * attenuation
                m.__dict__[f"Touts_{pipe}"] = Eqn(f"Touts_{pipe}", rhs)

                attenuation = exp(- m.lam[pipe] * m.Lr[pipe] / (m.Cp * Abs(m.mr[pipe])))
                Tstart = m.Tr[tnode] * heaviside(m.mr[pipe]) + m.Tr[fnode] * (1 - heaviside(m.mr[pipe]))  #
                rhs = m.Toutr[pipe] - Tstart * attenuation
                m.__dict__[f"Toutr_{pipe}"] = Eqn(f"Toutr_{pipe}", rhs)

        # mass flow continuity
        # supply
        for node in range(self.df.n_node):
            if node in self.df.slack_node.tolist() + self.df.s_node.tolist():
                rhs = Abs(m.min[node])
            elif node in self.df.l_node.tolist() + self.df.I_node.tolist():
                rhs = -Abs(m.min[node])
            else:
                raise ValueError(f"Unknown Node type {node}")

            for edge in self.HydraSup.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs + m.ms[pipe]

            for edge in self.HydraSup.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs - m.ms[pipe]
            m.__dict__[f"Mass_flow_continuity_sup_{node}"] = Eqn(f"Mass_flow_continuity_sup_{node}", rhs)

        rhs = m.ms[self.fault_pipe] - (m.ms[self.df.n_pipe] + m.m_leak[0])
        m.__dict__[f"Mass_flow_continuity_sup_{self.df.n_node}"] = Eqn(f"Mass_flow_continuity_sup_{self.df.n_node}",
                                                                       rhs)

        # return
        for node in range(self.df.n_node):
            if node in self.df.slack_node:
                rhs = - (m.min[node] - m.fs_injection)
            elif node in self.df.s_node:
                rhs = - m.min[node]
            elif node in self.df.l_node.tolist() + self.df.I_node.tolist():
                rhs = m.min[node]
            else:
                raise ValueError(f"Unknown Node type {node}")

            for edge in self.HydraRet.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs + m.mr[pipe]

            for edge in self.HydraRet.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs - m.mr[pipe]
            m.__dict__[f"Mass_flow_continuity_ret_{node}"] = Eqn(f"Mass_flow_continuity_ret_{node}", rhs)

        rhs = m.mr[self.df.n_pipe] - (m.mr[self.fault_pipe] + m.m_leak[1])
        m.__dict__[f"Mass_flow_continuity_ret_{self.df.n_node}"] = Eqn(f"Mass_flow_continuity_ret_{self.df.n_node}",
                                                                       rhs)

        # pressure drop
        for edge in self.HydraSup.G.edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            pipe = edge[2]['idx']
            if pipe != self.df.n_pipe + 1:  # DISCARD THE PIPE FROM LEAKAGE TO THE ENVIRONMENT
                rhs = m.Hs[fnode] - m.Hs[tnode] - m.K[pipe] * m.ms[pipe] ** 2 * Sign(m.ms[pipe])
                m.__dict__[f"Hs_{pipe}"] = Eqn(f"Hs_{pipe}", rhs)

                rhs = m.Hr[tnode] - m.Hr[fnode] - m.K[pipe] * m.mr[pipe] ** 2 * Sign(m.mr[pipe])
                m.__dict__[f"Hr_{pipe}"] = Eqn(f"Hr_{pipe}", rhs)

        m.Hs_slack = Eqn(f"Hs_slack", m.Hs[self.df.slack_node[0]] - m.Hset_s)
        m.Hr_slack = Eqn(f"Hr_slack", m.Hr[self.df.slack_node[0]] - m.Hset_r)

        # leak mass flow
        if self.fault_sys == 's':
            rhs_s = m.m_leak[0] - m.S_leak * (2 * m.g * m.Hs[self.df.n_node]) ** (1 / 2)
            rhs_r = m.m_leak[1]
        elif self.fault_sys == 'r':
            rhs_s = m.m_leak[0]
            rhs_r = m.m_leak[1] - m.S_leak * (2 * m.g * m.Hr[self.df.n_node]) ** (1 / 2)
        else:
            raise ValueError("Unknown fault sys")
        m.leak_mass_flow_sup = Eqn("leak_mass_flow_sup", rhs_s)
        m.leak_mass_flow_ret = Eqn("leak_mass_flow_ret", rhs_r)

        # heat power
        for node in range(self.df.n_node):
            if node in self.df.slack_node:
                phi = m.phi_slack
            else:
                phi = m.phi[node]

            if node in self.df.s_node.tolist() + self.df.slack_node.tolist():
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
            elif node in self.df.l_node:
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts[node] - m.Tload[node])
            elif node in self.df.I_node:
                rhs = m.min[node]

            m.__dict__[f'phi_{node}'] = Eqn(f"phi_{node}", rhs)

        sae, y0 = m.create_instance()
        self.smdl_full = sae
        nae = made_numerical(sae, y0, sparse=True)
        self.mdl_full = nae
        self.yf0 = y0

    def verify_results(self):
        if self.run_succeed:
            Tamb = self.df.Ta
            if self.mdl_full is None:
                self.mdl_full_dhspf()
            yf0 = self.yf0
            yf0['ms'] = self.ms
            yf0['mr'] = self.mr
            yf0['Ts'] = self.Ts - Tamb
            yf0['Tr'] = self.Tr - Tamb
            yf0['Hs'] = self.HydraSup.H[:-1]
            yf0['Hr'] = self.HydraRet.H[:-1]
            yf0['min'] = self.minset[: -1]
            m_leak = np.zeros(2)
            m_leak[0] = self.HydraSup.f[-1]
            m_leak[1] = self.HydraRet.f[-1]
            yf0['m_leak'] = m_leak
            yf0['fs_injection'] = (self.HydraSup.f[-1] + self.HydraRet.f[-1])
            yf0['Touts'] = self.Touts - Tamb
            yf0['Toutr'] = self.Toutr - Tamb
            yf0['phi_slack'] = self.phi_slack
            dF = self.mdl_full.F(yf0, self.mdl_full.p)
            return np.max(np.abs(dF))
        else:
            return 1


def remove_edge_by_idx(G, target_idx):
    """
    从图 G 中移除具有给定 idx 属性值的边。

    :param G: networkx.Graph 或其子类的实例
    :param target_idx: 要移除的边的 idx 属性值
    """
    edge_to_remove = None
    for u, v, data in G.edges(data=True):
        if data.get('idx') == target_idx:
            edge_to_remove = (u, v)
            break

    if edge_to_remove is not None:
        G.remove_edge(*edge_to_remove)
        print(f"Edge with idx={target_idx} removed.")
    else:
        print(f"No edge found with idx={target_idx}.")


def cal_H(slack_nodes,
          Hset,
          G: nx.DiGraph):
    """
    calculate H using depth-first search, for graph without self-loop and multiple edges
    """
    H = dict()
    slack_nodes = slack_nodes.tolist()
    H[slack_nodes[0]] = Hset[slack_nodes.index(0)]

    def dfs(i):
        for neighbor in list(G.successors(i)) + list(G.predecessors(i)):
            if neighbor not in H:
                if neighbor in G.successors(i):
                    f = i
                    t = neighbor
                    fij = G[f][t]['f']
                    cij = G[f][t]['c']
                    H[t] = H[f] - cij * fij ** 2 * np.sign(fij)
                elif neighbor in G.predecessors(i):
                    f = neighbor
                    t = i
                    fij = G[f][t]['f']
                    cij = G[f][t]['c']
                    H[f] = H[t] + cij * fij ** 2 * np.sign(fij)
                else:
                    raise ValueError(f'{neighbor} neither successor nor predecessor!')
                dfs(neighbor)

    dfs(slack_nodes[0])

    sorted_H = sorted(H.items(), key=lambda item: item[0])
    H = [np.array(item[1]).reshape(-1) for item in sorted_H]

    return np.asarray(H).reshape(-1)
