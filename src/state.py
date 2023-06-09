###########################################################################
###########################################################################
#                             SPyH
###########################################################################
###########################################################################
# Authors :  R. Carmigniani & D. Violeau
# Version : SPyH.0
# Contact : remi.carmigniani (at) enpc (dot) fr
###########################################################################
# Some useful imports
from numba import njit


@njit
def pressure(rho, B, rhoF, gamma):
    """
    return the pressure
    """
    # TODO COMPLETE HERE
    p = B * ((rho / rhoF) ** gamma - 1)
    # END
    return p


@njit
def density(p, B, rhoF, gamma):
    """
    return the density
    """
    # TODO COMPLETE HERE
    rho = rhoF * (p / B + 1) ** (1 / gamma)
    # END
    return rho


@njit
def soundSpeed(rho, B, rhoF, gamma):
    """
    return the sound speed
    """
    c = (gamma * B / rhoF * (rho / rhoF) ** (gamma - 1)) ** 0.5
    return c
