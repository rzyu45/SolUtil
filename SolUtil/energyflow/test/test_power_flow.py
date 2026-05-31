"""Power-flow tests for ``SolUtil.PowerFlow``.

Ports the cookbook power-flow validation: build a MATPOWER case, solve
it, and cross-check the LoopEqn formulation against the legacy per-bus
scalar formulation. The two builds are algebraically identical, so their
complex bus voltages, slack injections, and reconstructed currents must
agree to Newton tolerance.

The LoopEqn path is only supported by the Numba-JIT module printer (the
inline ``made_numerical`` path does not accept ``LoopEqn``), so
``loopeqn=True`` pays a one-off compile cost; the test exercises both
paths.

The fixture ``test_power_flow/case9.xlsx`` is the IEEE 9-bus EPS case
shared with the cookbook IES chapter (a MATPOWER-format xlsx with
``bus`` / ``branch`` / ``gen`` / ``setting`` sheets parsed by
``load_mpc``).
"""
import numpy as np
import pytest

from SolUtil import PowerFlow


@pytest.fixture
def case_file(datadir):
    f = datadir / 'case9.xlsx'
    if not f.exists():
        pytest.skip(f'case9 fixture not available at {f}')
    return str(f)


def _complex_voltage(pf):
    return pf.Vm * np.exp(1j * pf.Va)


def test_pf_scalar_converges(case_file):
    pf = PowerFlow(case_file, loopeqn=False)
    pf.run()
    assert pf.run_succeed
    # voltage magnitudes are physical (per-unit, near 1.0)
    assert np.all(pf.Vm > 0.8) and np.all(pf.Vm < 1.2)


def test_pf_loopeqn_converges(case_file):
    pf = PowerFlow(case_file, loopeqn=True)
    pf.run()
    assert pf.run_succeed
    assert np.all(pf.Vm > 0.8) and np.all(pf.Vm < 1.2)


def test_pf_loopeqn_matches_scalar(case_file):
    """LoopEqn and legacy scalar formulations must converge to the same
    complex bus voltage and the same reconstructed slack / pv injections."""
    pf_scalar = PowerFlow(case_file, loopeqn=False)
    pf_scalar.run()
    assert pf_scalar.run_succeed

    pf_loop = PowerFlow(case_file, loopeqn=True)
    pf_loop.run()
    assert pf_loop.run_succeed

    V_scalar = _complex_voltage(pf_scalar)
    V_loop = _complex_voltage(pf_loop)
    np.testing.assert_allclose(V_loop, V_scalar, rtol=1e-6, atol=1e-6)

    np.testing.assert_allclose(pf_loop.Pg, pf_scalar.Pg, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pf_loop.Qg, pf_scalar.Qg, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pf_loop.ix, pf_scalar.ix, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pf_loop.iy, pf_scalar.iy, rtol=1e-6, atol=1e-6)


def test_pf_default_is_scalar():
    """Default formulation stays the lightweight inline scalar build, so
    existing callers do not silently pay the LoopEqn compile cost."""
    import inspect
    sig = inspect.signature(PowerFlow.__init__)
    assert sig.parameters['loopeqn'].default is False
