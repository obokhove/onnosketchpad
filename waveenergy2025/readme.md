# Wave energy device summer 2025

29-08: gif at CG3 made with python3 wavebuoyhydynVBMGN2DHancopy2808.py native venv
gif Plotting: ii, t 14740 3.997088523610295
E0, E1, |E1-E0|/E0: 0.0005186092432332517 0.0005186092432335049 4.881540593010651e-13
Hallo nonstatic
Elapsed time (min): 918.6930528799693
Gif-maker CG3 Elapsed time (min): 318.23679311672845
Static CG3: Elapsed time (min): 380.833709247907
Gif-maker CG2: Elapsed time (min): 88.67235361735025
CG1:about 23min.


27-08:
- ...han.py can make animations.
- see sample .gif file
- to do: better solvers, explorations and wave-breaking smoothing.

25-08-2025: Run AVF-code for Variational Boussinesq Model:
python3 wavebuoyhydynVBMGN2DH.py
Reproduces wavebuoyGNVBMA025c4000T1Ts2.png

Macbook 2020 via Docker 26-08:
- Step 3680, final SNES residual norm 1.7330e-15
- Converged reason: 4; Plotting: ii, t 3680 3.991665065641824
- E0, E1, |E1-E0|/E0: 0.0005181748452705241 0.000518174845270523 2.0923481376628265e-15
- Elapsed time (min): 52.5571262160937

M2-Macbook native 25-08:
- Step 3680, final SNES residual norm 6.5634e-16
- Converged reason: 4; Plotting: ii, t 3680 3.991665065641824
- E0, E1, |E1-E0|/E0: 0.0005181748452688466 0.0005181748452688488 4.1846962753392e-15
- Elapsed time (min): 23.418518352508546

M2-Macbook via Docker 25-08:
- Step 3680, final SNES residual norm 1.6366e-15
- Converged reason: 4; Plotting: ii, t 3680 3.991665065641824
- E0, E1, |E1-E0|/E0: 0.0005181748452683975 0.0005181748452683967 1.6738785101371308e-15
- Elapsed time (min): 39.863962813218436

Earlier Benney-Luke BLE-AVF model: wavebuoyhydyn.py

Change Amp, sigma and/or c and/or tstop to get convergence or not.
Shorten tstop, lengthen sigma, lower c for convergence.

Note that when betaa=0.0 one has Green-Naghdi; for betaa=1 one has 
One estimate is dt ~ 1/sqrt{c} from bouncing ball case.
When t>tstop there is no forcing and AVF-energy conservation seen up to 10^(-14) for tight tolerances.
Otherwise it drops to ~10^(-9).

Question: better solver settings, convergence analysis for dt in terms of c, etc?
Now dt set with CFL and simple SWE phase speed as estimate.

First tests show that when a crash occurs with the buoy motion and wave forcing it will also occur without buoy motyion but with the same wavemaker forcing. So the wave model may need taming.
