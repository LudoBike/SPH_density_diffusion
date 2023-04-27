# SPH density diffusion

This project implement density diffusion and free surface detection for SPH.

This is an improvement a code written by R. Carmigniani and D. Violeau. Their original work is visible in the commit c50203ad9311f3a2703b494469fca7e4dde067a5.

## Density diffusion
The density diffusion is implemented with the Molteni and Colagrossi scheme. It is tested in the case of a dam break (see `simulationDamBreakDensityDiffusion.ipynb` and compared to experimental results from [Zhou et al. (1999)](https://www.researchgate.net/profile/Bas-Buchner/publication/267414591_A_nonlinear_3-D_approach_to_simulate_green_water_dynamics_on_deck/links/5551d68208ae739bdb924383/A-nonlinear-3-D-approach-to-simulate-green-water-dynamics-on-deck.pdf) (see `simulationDamBreakComparaisonZhouetal.ipynb`).

## Free surface detection
The free surface detection is based on the work of [M. Doring (2005)](https://www.theses.fr/2005NANT2116).
