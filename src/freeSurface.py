###########################################################################
###########################################################################
#                 Free surface detection for SPyH
###########################################################################
###########################################################################
# Authors :  L. Druette & T. Contentin
###########################################################################

import numpy as np

from .spyh import Fw


def minEigenvalueR(volPart, er, rNorm, dwdr):
    nPart = len(rNorm)
    invR = np.zeros((2, 2))
    for j in range(nPart):
        invR += volPart[j] * dwdr[j] * rNorm[j] * np.outer(er, er)

    return np.min(np.linalg.eigvals(invR))


def freeSurfaceDetection(
    partBOUND, partSPID, partPos, partRho, listNeibSpace, aW, h, m
):
    """
    Detected isolated particles and free surface using renormalization matrix,
    based on results in Doring (2005)
    input :
        -
    output :
        - partLambda : table of fluid particules lambda
        - isolatedPart : table of ID of isolated particles (lambda <= 0.20)
        - freeSurface : table of ID of particles belonging to free surface
                        (0.20 < lambda < 0.75)
    """
    nPart = len(partBOUND)
    partLambda = -1 * np.ones_like(partBOUND)
    isolatedPart = []
    freeSurface = []
    for i in range(nPart):
        if not partBOUND[i]:
            spid_i = int(partSPID[i])
            # list neib
            listnb = listNeibSpace[spid_i, :]
            listnb = listnb[listnb > -1]
            listnb = listnb[listnb != i]  # no self contribution
            rPos = partPos[i, :] - partPos[listnb][:]
            rNorm = (rPos[:, 0] * rPos[:, 0] + rPos[:, 1] * rPos[:, 1]) ** 0.5
            q = rNorm / h
            dwdr = Fw(q, aW, h)
            er = np.zeros_like(rPos)
            er[:, 0] = rPos[:, 0] / rNorm
            er[:, 1] = rPos[:, 1] / rNorm
            volPart = m / partRho[listnb]
            partLambda[i] = minEigenvalueR(volPart, rPos, rNorm, dwdr)
            if partLambda[i] <= 0.20:
                isolatedPart.append(spid_i)
            elif partLambda[i] > 0.20 and partLambda[i] < 0.75:
                freeSurface.append(spid_i)

    return partLambda, isolatedPart, freeSurface
