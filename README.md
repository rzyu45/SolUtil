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

## Installation

```
pip install SolUtil
```

## Requirements

This package requires the `ipopt` optimization solver. 

On `windows`, download the latest release [here](https://github.com/coin-or/Ipopt/releases). Then add the `bin` 
directory to the system path. Make sure that you can call `ipopt` in your terminal.

On `macos`, use `brew install ipopt` to perform the installation.
