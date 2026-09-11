"""Tests of the MATPOWER ``.m`` input of ``load_mpc`` and ``PowerFlow``.

``test_load_matpower/case9.m`` is MATPOWER 8.1's ``case9.m``, unchanged. The
other cases are written by the tests, each to exercise one rule of the
preparation that ``read_matpower`` copies from MATPOWER ``runpf``.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from Solverz import Opt, nr_method
from SolUtil import PowerFlow
from SolUtil.sysparser import load_mpc
from SolUtil.sysparser.eps_function import read_matpower

# MATPOWER 8.1 ``runpf`` on case9, in the bus order of the file
CASE9_VM = [1.04, 1.025, 1.025, 1.02578839284401, 1.01265432401778,
            1.03235294900237, 1.0158825836275, 1.02576937238645, 0.995630858048295]
CASE9_VA_DEG = [0.0, 9.28000548164279, 4.66475133313676, -2.21678779994979,
                -3.68739617015706, 1.96671607444908, 0.727536076874292,
                3.71970115462176, -3.98880527285147]

#       bus_i type  Pd  Qd Gs Bs area   Vm Va baseKV zone Vmax Vmin
BUS = [[10,   3,    0,  0, 0, 0, 1,   1.0, 0, 230,   1,   1.1, 0.9],
       [20,   2,    0,  0, 0, 0, 1,   1.0, 0, 230,   1,   1.1, 0.9],
       [30,   2,   20,  5, 0, 0, 1,  0.98, 0, 230,   1,   1.1, 0.9],
       [40,   1,   60, 20, 0, 0, 1,   1.0, 0, 230,   1,   1.1, 0.9],
       [50,   1,   40, 10, 0, 0, 1,   1.0, 0, 230,   1,   1.1, 0.9]]
#       bus  Pg Qg Qmax  Qmin    Vg mBase status Pmax Pmin, then 11 zero columns
GEN = [[20,  30, 5,  50,  -50, 1.03, 100, 1, 100, 0],
       [10,   0, 0, 300, -300, 1.02, 100, 1, 300, 0],
       [30,  25, 0,  50,  -50, 1.01, 100, 0, 100, 0],   # out of service
       [20,  20, 7,  50,  -50, 1.05, 100, 1, 100, 0]]   # a second generator at bus 20
#          fbus tbus    r    x     b rateA rateB rateC ratio angle status angmin angmax
BRANCH = [[10,  20,  0.01, 0.1, 0.02, 0, 0, 0, 0, 0, 1, -360, 360],
          [20,  30,  0.01, 0.1, 0.02, 0, 0, 0, 0, 0, 1, -360, 360],
          [30,  40,  0.01, 0.1, 0.02, 0, 0, 0, 0, 0, 1, -360, 360],
          [40,  50,  0.01, 0.1, 0.02, 0, 0, 0, 0, 0, 1, -360, 360],
          [10,  50,  0.01, 0.1, 0.02, 0, 0, 0, 0, 0, 0, -360, 360]]   # out of service


def _write_case(path, bus=BUS, gen=GEN, branch=BRANCH):
    """A MATPOWER case file with the given rows; generator rows are padded
    to the 21 columns of the version 2 format."""
    def block(name, rows):
        body = '\n'.join('\t' + '\t'.join(repr(v) for v in row) + ';' for row in rows)
        return f'mpc.{name} = [\n{body}\n];\n'
    gen = [list(row) + [0] * (21 - len(row)) for row in gen]
    path.write_text(f"function mpc = {path.stem}\nmpc.version = '2';\nmpc.baseMVA = 100;\n"
                    + block('bus', bus) + block('gen', gen) + block('branch', branch))
    return str(path)


def _assert_same_arrays(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if issparse(a[k]):
            assert a[k].shape == b[k].shape and (a[k] != b[k]).nnz == 0, k
        else:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_matpower_case9_solves_to_the_matpower_solution(datadir):
    pf = PowerFlow(datadir / 'case9.m')
    sol = nr_method(pf.pfmdl, pf.y0, Opt(ite_tol=1e-10))
    assert sol.stats.succeed
    pf.parse_data_post_pf(sol)
    np.testing.assert_allclose(pf.Vm, CASE9_VM, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(pf.Va), CASE9_VA_DEG, rtol=0, atol=1e-10)


def test_m_and_xlsx_of_one_case_give_the_same_arrays(datadir, tmp_path):
    """The two inputs share every step after reading the tables, so the
    xlsx written from the tables of a ``.m`` file loads to the same arrays."""
    m = datadir / 'case9.m'
    base_mva, bus, branch, gen = read_matpower(m)
    xlsx = tmp_path / 'case9.xlsx'
    with pd.ExcelWriter(xlsx, engine='openpyxl') as xw:
        pd.DataFrame({'baseMVA': base_mva}).to_excel(xw, sheet_name='setting', index=False)
        bus.to_excel(xw, sheet_name='bus', index=False)
        branch.to_excel(xw, sheet_name='branch', index=False)
        gen.to_excel(xw, sheet_name='gen', index=False)
    _assert_same_arrays(load_mpc(str(m)), load_mpc(str(xlsx)))


def test_runpf_preparation(tmp_path):
    """Generators out of service are dropped and those of one bus summed;
    a PV bus without a generator in service becomes a PQ bus; a generator bus
    starts at the setpoint of its first generator; buses are numbered in file
    order; a branch out of service adds nothing to the admittance matrix."""
    mpc = load_mpc(_write_case(tmp_path / 'case_rules.m'))
    assert mpc['nb'] == 5
    np.testing.assert_array_equal(mpc['idx_slack'], [0])
    np.testing.assert_array_equal(mpc['idx_pv'], [1])
    np.testing.assert_array_equal(mpc['idx_pq'], [2, 3, 4])
    np.testing.assert_allclose(mpc['Pg'], [0, 0.5, 0, 0, 0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(mpc['Qg'], [0, 0.12, 0, 0, 0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(mpc['Pd'], [0, 0, 0.2, 0.6, 0.4], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(mpc['Vm'], [1.02, 1.03, 0.98, 1.0, 1.0])
    Y = mpc['Ybus'].toarray()
    assert Y[0, 4] == 0 and Y[4, 0] == 0
    assert Y[0, 1] != 0 and Y[3, 4] != 0

    base_mva, bus, branch, gen = read_matpower(tmp_path / 'case_rules.m')
    np.testing.assert_array_equal(gen['bus'], [0, 1])
    np.testing.assert_array_equal(gen['Qmax'], [300, 100])
    np.testing.assert_array_equal(branch['status'], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(branch[['fbus', 'tbus']].to_numpy()[-1], [0, 4])

    pf = PowerFlow(tmp_path / 'case_rules.m')
    sol = nr_method(pf.pfmdl, pf.y0, Opt(ite_tol=1e-10))
    assert sol.stats.succeed


def test_a_case_without_a_reference_generator_promotes_the_first_pv_bus(tmp_path):
    gen = [row[:7] + [0] + row[8:] if row[0] == 10 else row for row in GEN]
    with pytest.warns(UserWarning, match='promoted to the reference bus'):
        mpc = load_mpc(_write_case(tmp_path / 'case_noref.m', gen=gen))
    np.testing.assert_array_equal(mpc['idx_slack'], [1])
    np.testing.assert_array_equal(mpc['idx_pv'], [])
    np.testing.assert_array_equal(mpc['idx_pq'], [0, 2, 3, 4])


@pytest.mark.parametrize('change, message', [
    ('isolated', 'isolated buses of type 4'),
    ('duplicate', 'duplicate bus numbers'),
    ('unknown', 'references to unknown bus numbers'),
])
def test_cases_the_power_flow_cannot_take_are_refused(tmp_path, change, message):
    bus, branch = [list(row) for row in BUS], [list(row) for row in BRANCH]
    if change == 'isolated':
        bus[4][1] = 4
    elif change == 'duplicate':
        bus[4][0] = 40
    else:
        branch[3][1] = 60
    with pytest.raises(ValueError, match=message):
        load_mpc(_write_case(tmp_path / f'case_{change}.m', bus=bus, branch=branch))
