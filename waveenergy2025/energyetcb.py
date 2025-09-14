import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#------ User Input ------#
# Absolute adata_path (change to your settings), e.g.:
adata_path='/Users/onnobokhove/amtob/werk/vuurdraak2021/wavenergy2025/'

# fileEdata = np.array([[ t, Z00[0], W00[0], I00[0], E0, E1 ]])
fileEZWI = adata_path +'data10/VBMZWIenergy.csv' # 

save_figure=True
figure_name_1='energy.png'
figure_name_2='compZWIE.png'

with open(fileEZWI,'r') as MMP1:
    t1_MMP, E1_tot_MMP = np.loadtxt(MMP1, usecols=(0,1), unpack=True)

save_figure = True
figure_name = 'energy_panels.png'

# ------ Load Data ------ #
# Assuming file has columns: t, Z, W, I, E0, E1
data = np.loadtxt(fileEZWI)
t = data[:, 0]
Z = data[:, 1]
W = data[:, 2]
I = data[:, 3]
E0 = data[:, 4]
E1 = data[:, 5]
relE0E1 = data[:, 5]
rel_diff = np.abs(E1 - E0) / E0
# ------ Plot ------ #
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axs[0, 0].plot(t, Z,'.')
axs[0, 0].plot(t, Z, '-', label='Z')
axs[0, 0].set_ylabel('Z (m)')
axs[0, 0].legend()
axs[0, 1].plot(t, W, '-', label='W', color='orange')
axs[0, 1].plot(t, W, '.', color='orange')
axs[0, 1].set_ylabel('W (m/s)')
axs[0, 1].legend()
axs[1, 0].plot(t, I, '-', label='I', color='green')
axs[1, 0].plot(t, I, '.', color='green')
axs[1, 0].set_xlabel('t (s)')
axs[1, 0].set_ylabel('I (A)')
axs[1, 0].legend()
axs[1, 1].plot(t, relE0E1, '.', color='blue')
axs[1, 1].plot(t, relE0E1, '-', label=r'|E(t)-E(0)|/E(0)', color='blue')
axs[1, 1].set_xlabel('t (s)')
axs[1, 1].set_ylabel(r'|E(t)-E(0)|/E(0)')
axs[1, 1].legend()

Rl = 100
Rc = 45
# Add twin axis for power
ax2 = axs[1, 0].twinx()
ax2.plot(t, Rl*I**2, '-', label=r'$P=R_l I^2$', color='black')
ax2.plot(t, Rc*I**2, '-', label=r'$P_l=R_c I^2$', color='blue')
ax2.plot(t, Rl*I**2, '.', color='black')
ax2.set_ylabel(r'$P=R_l I^2$ (W)', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(loc='upper left') # Add legend for twin axis

P_Rl = Rl * I**2
dt = np.diff(t)
E_Rl = np.concatenate([[0], np.cumsum(0.5 * (P_Rl[1:] + P_Rl[:-1]) * dt)])
ax2.plot(t, E_Rl, '--', label=r'$\int_0^t R_l I^2 dt$', color='red')


# add secondary y-axis

nrel=1
# Inside loop
if nrel == 1:
    tstop = 2.0
    mask = t > tstop
    tshort = t[mask]
    E1short = E1[mask]
    E0short = E0[mask]
    rel_diff_short = relE0E1[mask]
    print(f"Original t length: {len(t)}")
    print(f"tshort length: {len(tshort)}")
    print(f"Number of points with t > tstop: {np.sum(mask)}")
    Emin = -10**(-8) # rel_diff_short.min()
    Emax = 10**(-8) # rel_diff_short.max()
    # Create secondary y-axis and plot using the shortened arrays
    ax2 = axs[1, 1].twinx()
    ax2.plot(tshort, rel_diff_short, '.', color='red')
    ax2.plot(tshort, rel_diff_short, '-', label='|E(t)-E(T)|/E(T)', color='red')
    # ax2.set_ylim([Emin * 0.95, Emax * 1.05])  # Add 5% padding
    ax2.set_ylabel('|E(t)-E(T)|/E(T)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='center')

plt.tight_layout()

if save_figure:
    plt.savefig(adata_path + figure_name, dpi=300)

plt.show()



