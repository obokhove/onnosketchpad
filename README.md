# Onno's sketchpad

25-08-2025: see wave energy 2025 folder.
 

13-07: Typos corrected in pdf and updated code (...copy.py)

09-07-2025 update for Colin:
* two energy-conserving bouncing ball codes; one with Z fails since Z<0 in iterate I suspect; but I did not catch out Z^n+1=Z^n; but one with theta as in Z=exp(theta) seems to work. Terribly slow!
Note that in the graph above one can see in the middle eenrgy panel where Z~0 where L'Hopital is used; so some refinement seems needed.
* Notes updated; see red-colour and "Colin Cotter:" indicated parapragh starts!
* I have a Brown-MMP based Benney-Luke model; I will upload that one shortly and also work on energy-conserving code for that case but with lambda explicitly removed.

Billards codes:

...VI.py: file with VI without Lagrange multiplier (code runs but particle does not see wall).
Tried to follow drape.py (ball case) in: https://bitbucket.org/pefarrell/fascd/src/master/examples/drape.py

...VIlam.py: file with VI with Lagrange multiplier and all KKT-equations explicit (code runs but particle does not see wall).

....py: Burman solution with soft wall. Works when dt and softness are tuned such that "particle can turn around smoothly with resolved time stepping near the wall"

All plotting done via matplotlib on the fly.

See pdf.
