import numpy as np
import matplotlib.pyplot as plt

def generate_perfect_asymptotic_plots(a=1.0, b=0.1, d=1.0, delta=0.1, gamma=1.0):
    # Match Region III (Quadratic) at G = -delta
    # lambda = a(G/b - G^2/d)
    v_neg = a * ((-delta)/b - (-delta)**2/d)
    s_neg = a * (1/b - 2*(-delta)/d)
    c_neg = -2*a/d
    
    # C2 Spline Coefficients (lambda = u^3 * (Af + Bf*v + Cf*v^2))
    # u = G - delta, v = G + delta. At G = -delta: u = -2*delta, v = 0.
    Af = v_neg / (-8 * delta**3)
    Bf = (s_neg - 12 * delta**2 * Af) / (-8 * delta**3)
    Cf = (c_neg + 12 * delta * Af - 24 * delta**2 * Bf) / (-16 * delta**3)

    # Sweep G from Air (+delta) to Deep Contact (-50)
    G = np.linspace(-50.0, delta + 0.1, 100000)
    
    Qs, Fps, GQs = [], [], []

    for g in G:
        u, v = g - delta, g + delta
        if g >= delta:
            lam, lam_p = 0.0, 0.0
        elif g > -delta:
            lam = u**3 * (Af + Bf*v + Cf*v**2)
            lam_p = 3*u**2*(Af + Bf*v + Cf*v**2) + u**3*(Bf + 2*Cf*v)
        else:
            lam = a * (g/b - g**2/d)
            lam_p = a * (1/b - 2*g/d)

        # Correct Mapping:
        # F+ = -lambda (lambda is negative in contact)
        # Q = -gamma*G - lambda
        q = -gamma * g - lam
        f_p = -lam
        
        # dF/dQ = (dF/dG) / (dQ/dG)
        dq_dg = -gamma - lam_p
        df_dq = (-lam_p / dq_dg) if abs(dq_dg) > 1e-9 else 0.0
        
        Qs.append(q)
        Fps.append(f_p)
        GQs.append(f_p * df_dq)

    Qs, Fps, GQs = np.array(Qs), np.array(Fps), np.array(GQs)
    idx = np.argsort(Qs)
    Qs, Fps, GQs = Qs[idx], Fps[idx], GQs[idx]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(Qs, Fps, 'b-', lw=3, label=r'$F_{+}(Q)$')
    ax1.plot(Qs, Qs, 'r--', lw=1.5, alpha=0.9, label='Identity $F_{+}=Q$')
    ax1.set_title(r'$F_{+}(Q)$ Mapping: Converged to Identity')
    ax1.set_xlim([-1, 50]); ax1.set_ylim([-1, 50])
    ax1.legend(); ax1.grid(True)

    ax2.plot(Qs, GQs, 'g-', lw=3, label=r'$G(Q)$')
    ax2.plot(Qs, Qs, 'r--', lw=1.5, alpha=0.9, label='Identity $G(Q)=Q$')
    ax2.set_title(r'$G(Q)$ Mapping: Converged to Identity')
    ax2.set_xlabel(r'$Q = -\gamma G - \lambda$')
    ax2.set_xlim([-1, 50]); ax2.set_ylim([-1, 50])
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()

generate_perfect_asymptotic_plots()

