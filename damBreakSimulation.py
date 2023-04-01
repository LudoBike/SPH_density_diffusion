import numpy as np
from src.sphvar import (
    smthfc,
    nBound,
    FLUID,
    BOUND,
    POS,
    SPID,
    FORCES,
    INFO,
    DRHODT,
    VEL,
    RHO,
    aW,
)
from src.spyh import (
    init_spaces,
    init_particles,
    addBox,
    sortPart,
    getListNeib,
    CFLConditions,
    interpolateBoundary,
    computeForcesART,
    integrationStep,
    checkDensity,
)


# FLUID PARAMETERS
rhoF = 1000
c0 = 40
gamma = 7
B = rhoF * c0**2 / gamma
grav = np.array([0, -9.81])

# ARTIFICIAL VISCOSITY (voir plus loin)
alpha = 0.3
esp = 10 ** (-6)  # and it is multiplied by dr**2 in the code

# Density diffusion parameters
delta = 0.2

# DENSITY & SHEPARD THRESHOLDS :
shepardMin = 10 ** (-6)
rhoMin = 0.5 * rhoF
rhoMax = 1.5 * rhoF

# GEOMETRY
Lx = 2
Ly = 2
lx = 0.5
ly = 1

# PARTICLES & SPACES PARAMETERS :
dr = ly / 20
h = smthfc * dr
m = dr * dr * rhoF
lspace = 2 * h

# COMPUTATION DOMAIN :
xOrigin = -nBound * dr
yOrigin = -nBound * dr
xSize = Lx + 2 * nBound * dr
ySize = Ly + 2 * nBound * dr
xMax = xOrigin + xSize
yMax = yOrigin + ySize

# INIT SPACES & PART:
posSpace, neibSpace, partSpace, listNeibSpace = init_spaces(
    xOrigin, yOrigin, xSize, ySize, lspace, dr
)
part = init_particles()
part = addBox(part, [lx, ly], FLUID, dr, rhoF)
part = addBox(
    part,
    [-nBound * dr, -nBound * dr, Lx + 2 * nBound * dr, nBound * dr],
    BOUND,
    dr,
    rhoF,
)
part = addBox(part, [-nBound * dr, 0, nBound * dr, Ly], BOUND, dr, rhoF)
part = addBox(part, [Lx, 0, nBound * dr, Ly], BOUND, dr, rhoF)
part, partSpace = sortPart(
    part, posSpace, partSpace, xOrigin, yOrigin, xSize, ySize, lspace, dr
)
listNeibSpace = getListNeib(partSpace, neibSpace, listNeibSpace)

# time, iteration count and im_count
t = 0
it = 0
im_count = 0

# Here we specify the output frequencies
dt_figure = 2.0 / 100.0 * (ly / np.linalg.norm(grav)) ** 0.5
t_print = 0

# final time :
t_end = 10

while t < t_end:
    # STEP1 : Calcul de la CFL
    dt = CFLConditions(part[:, VEL], h, c0, grav)
    # STEP2 : Interpolation des conditions au bord
    part[:, RHO], part[:, VEL] = interpolateBoundary(
        (part[:, INFO] == BOUND),
        part[:, SPID],
        part[:, POS],
        part[:, VEL],
        part[:, RHO],
        listNeibSpace,
        aW,
        h,
        m,
        B,
        rhoF,
        gamma,
        grav,
        shepardMin,
    )
    # STEP3 : Calcul des forces et des termes de densité
    part[:, FORCES], part[:, DRHODT] = computeForcesART(
        (part[:, INFO] == FLUID),
        part[:, SPID],
        part[:, POS],
        part[:, VEL],
        part[:, RHO],
        listNeibSpace,
        aW,
        h,
        m,
        B,
        rhoF,
        gamma,
        grav,
        alpha,
        esp,
        dr,
        c0,
        delta,
        densityDiffusion=True,
    )
    # STEP4 : Integration en temps
    part[:, POS], part[:, VEL], part[:, RHO] = integrationStep(
        (part[:, INFO] == FLUID),
        part[:, POS],
        part[:, VEL],
        part[:, RHO],
        part[:, FORCES],
        part[:, DRHODT],
        dt,
    )
    # STEP5 : Corriger densité trop basse
    part[:, RHO] = checkDensity(part[:, RHO], rhoMin, rhoMax)
    # STEP6 : Mise à jour des voisins (pas forcément à tous les pas de temps)
    part, partSpace = sortPart(
        part, posSpace, partSpace, xOrigin, yOrigin, xSize, ySize, lspace, dr
    )
    listNeibSpace = getListNeib(partSpace, neibSpace, listNeibSpace)
    t += dt
    it += 1
