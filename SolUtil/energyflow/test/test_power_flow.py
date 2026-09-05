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


def test_pf_default_is_loopeqn():
    """Default formulation is the LoopEqn build (since 0.9.0)."""
    import inspect
    sig = inspect.signature(PowerFlow.__init__)
    assert sig.parameters['loopeqn'].default is True


def test_pf_scalar_path_is_deprecated(case_file):
    """The legacy inline scalar path warns it is deprecated."""
    with pytest.warns(DeprecationWarning):
        PowerFlow(case_file, loopeqn=False)


def test_pf_two_instances_are_independent(case_file, datadir):
    """Two PowerFlow objects in one process must not share a rendered module.

    The LoopEqn build renders a module under a per-instance name. With a
    fixed name the second instance got the first instance's F / J / y.
    """
    pf_a = PowerFlow(case_file)
    pf_b = PowerFlow(case_file)
    assert pf_a.pfmdl is not pf_b.pfmdl
    assert pf_a.pfmdl.F is not pf_b.pfmdl.F
    pf_b.run()
    assert pf_b.run_succeed
    assert pf_b.Vm.shape == (pf_b.nb,)
