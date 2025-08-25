# Wave energy device summer 2025

25-08-2025: Run AVF-code for Variational Boussinesq Model:
python3 wavebuoyhydynVBMGN2DH.py

Ealier Benney-Luke BLE-AVF model: wavebuoyhydyn.py

Change Amp and/or c and/or tstop to get convergence or not.
Note that when betaa=0.0 one has Green-Naghdi; for betaa=1 one has 
One estimate is dt ~ 1/sqrt{c} from bouncing ball case.
When t>tstop there is no forcing and AVF-energy conservation seen up to 10^(-14) for tight tolerances.
Otherwise it drops to ~10^(-9).

Question: better solver settings, convergence analysis for dt in terms of c, etc?
Now dt set with CFL and simple SWE phase speed as estimate.
