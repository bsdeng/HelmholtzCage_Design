# Square Shaped Helmholtz-Cage Design

## Contents
- Features
- References

To install and compute the optimal parameter values, just run these commands on your terminal.
```
pip install -r requirements.txt \
python helmholtz_cage.py \
```

## 2 - Features
- This code performs the computation of the optimal parameter values of the Square shaped Helmholtz Cage.
- This optimization problem can be easily solved through [geometric programming (GP)](https://github.com/cvxpy/cvxpy#installation). The constraints of this problem are: (1) available space (2) the size of the spacecraft (3) upper limit of the power supply (4) number of turns.
- All these optimization variables may affect your helmholtz cage size, budget, and power consumption.

## 3 - Theory

## References

[1] E. Cayo, J. Pareja, P. E. R. Arapa, "Design and implementation of a geomagnetic field simulator
for small satellites ," Conference: III IAA Latin American Cubesat Workshop, Jan. 2019. \
[2] R. C. D. Silva, F. C. Guimaraes, J. V. L. D. Loiola, R. A. Borges, S. Battistini, and C.
Cappelletti, "Tabletop testbed for attitude determination and control of nanosatellites," Journal
of Aerospace Engineering, vol. 32, no. 1, 2018. \
[3] J. Stevens, "CubeSAT ADCS Validation and Testing Apparatus," Western Michigan University, 2016 \
[4] N. Theoret, "Attitude Determination Control Testing System," Western Michigan University, 2016
