import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#------ User Input ------#
# Absolute adata_path (change to your settings), e.g.:
adata_path='/Users/onnobokhove/amtob/werk/vuurdraak2021/wavenergy2025/'

# fileEdata = np.array([[ t, Z00[0], W00[0], I00[0], E0, E1 ]])
fileEZWI = adata_path +'data7/VBMZWIenergy.csv' # 

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
rel_diff = np.abs(E1 - E0) / E0
# ------ Plot ------ #
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axs[0, 0].plot(t, Z, 'x', label='Z')
axs[0, 0].set_ylabel('Z')
axs[0, 0].legend()
axs[0, 1].plot(t, W, 'o', label='W', color='orange')
axs[0, 1].set_ylabel('W')
axs[0, 1].legend()
axs[1, 0].plot(t, I, '.', label='I', color='green')
axs[1, 0].set_xlabel('Time t')
axs[1, 0].set_ylabel('I')
axs[1, 0].legend()
axs[1, 1].plot(t, rel_diff, 'x', label='|E1 - E0| / E0', color='red')
axs[1, 1].set_xlabel('Time t')
axs[1, 1].set_ylabel('Relative Diff')
axs[1, 1].legend()

# add secondary y-axis
nrel = 1
nfa = 1
if nrel==1:
    tstop = 1.0
    if nfa == 1:
        # Find first E1 where corresponding t > tstop
        for i, ti in enumerate(t):
            if ti > tstop:
                E00 = E1[i]  # reference value
                first_idx = i  # index of E00
                nfa = 0
                break
    if nfa == 0:
        # mask for t strictly after the reference point
        mask = np.arange(len(t)) > first_idx
        rel_diff0 = np.abs(E1[mask] - E00) / E00
        ax2 = axs[1, 1].twinx()  # same subplot, new vertical axis
        ax2.plot(t[mask], rel_diff0, 'o', label='|E1 - E00| / E00 (t>tstop)', color='blue')
        ax2.set_ylabel('Same quantity for t>tstop')
        axs[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')

plt.tight_layout()

if save_figure:
    plt.savefig(adata_path + figure_name, dpi=300)

plt.show()



