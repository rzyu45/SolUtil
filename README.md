# Solverz Utilities

This lib provides the simulation routines for integrated energy systems, including electric power, natural gas and heat.

A simple usage of this library can be

```python
from SolUtil import GasFlow
gf = GasFlow('belgium.xlsx')
gf.run()
print(gf.Pi) # get the node Pressure results
print(gf.f) # get the pipe mass-flow results
```

The required `.xlsx` data format can be found in SolUtil/energyflow/test directory for reference.

## Requirements

This package requires the `ipopt` optimization solver. 

On `windows`, download the latest release [here](https://github.com/coin-or/Ipopt/releases). Then add the `bin` 
directory to the system path. Make sure that you can call `ipopt` in your terminal.

On `macos`, use `brew install ipopt` to perform the installation.

## MATPOWER case files

The power flow also reads a MATPOWER case file directly:

```python
from SolUtil import PowerFlow
pf = PowerFlow('case118.m')
pf.run()
print(pf.Vm, pf.Va) # get the bus voltage magnitudes and angles
```

A `.m` file is parsed with `matpowercaseframes` and prepared as MATPOWER `runpf` prepares it: generators out of service are dropped, the generators in service at one bus are summed, a PV or reference bus without a generator in service becomes a PQ bus, and every generator bus starts at its voltage setpoint. For large cases it is much faster to read than the `.xlsx` format.
