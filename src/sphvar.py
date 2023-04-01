###########################################################################
###########################################################################
#                             SPyH
###########################################################################
###########################################################################
# Original authors :  R. Carmigniani & D. Violeau
# Improvements for density diffusion : L. Druette
# Version : SPyH
# Contact : remi.carmigniani (at) enpc (dot) fr
###########################################################################
# In this file we list all the parameters stored in the array of particles
# and the flags
import numpy as np

# FLAGS :
# Type of particles
FLUID = 0
BOUND = 1
#
nBound = 4
#
# KERNEL PARAMETERS
aW = 7 / (4 * np.pi)
d = 2
smthfc = 2
nparameters = 10
POS = [0, 1]
VEL = [2, 3]
RHO = 4
FORCES = [5, 6]
DRHODT = 7
SPID = 8
INFO = 9
#
maxSpaceNeibs = 9
maxPartInSpace = 25

# Density diffusion parameters
delta = 0.2
