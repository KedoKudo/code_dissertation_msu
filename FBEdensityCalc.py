#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch calculate misfit dislocation density in given Frank-Bilby framework.

Usage:
    FBEdensityCalc.py [-hv] <streakData>
                      [--weight=<float>]
                      [--density=<float>]
    FBEdensityCalc.py [--version]

Options:
    -h --help               print this help information
    -v --version            show version information
    --weight=<float>        weight for dislocation density in
                            objective function. [default: 1]
    --density=<float>       initial guess of dislocation density
                            [default: 1]
"""
import numpy as np
from docopt import docopt
from scipy.optimize import minimize
from cyxtal.cxtallite import Eulers
from cyxtal import Xtallite
from cyxtal import slip_systems
from cyxtal import bravais2cartesian


class VoxelData(object):
    """
    Container class bundle data for dislocation content analysis,
    all var are in APS coordinate system
    """

    def __init__(self, pos, eulers):
        self.pos = np.array(pos)
        self.eulers = np.array(eulers)
        self.lattice = 'hexagonal'

    def __str__(self):
        msg = "Voxel@({},{},{})_APS".format(*self.pos)
        msg += " with EulerAngles:({},{},{})\n".format(*self.eulers)
        msg += "  g matrix=\n" + str(self.orientationMatrix.T)
        return msg

    @property
    def rotationMatrix(self):
        return self.orientationMatrix.T

    @property
    def orientationMatrix(self):
        phi1, phi, phi2 = self.eulers
        return Eulers(phi1, phi, phi2).toOrientationMatrix()

    def getDisorientationFrom(self, other, mode='angleaxis'):
        myXtal = Xtallite(eulers=self.eulers,
                          pt=self.pos,
                          lattice=self.lattice)
        neiXtal = Xtallite(eulers=other.eulers,
                           pt=other.pos,
                           lattice=other.lattice)
        return myXtal.disorientation(neiXtal, mode=mode)


# ---- Help functions ----- #
def getSlipSystem(slipSystemName='hcp', covera=1.58):
    # use Ti as default
    ss = slip_systems(slipSystemName)
    ss_cart = np.zeros((ss.shape[0], 2, 3))
    matrixP = []
    for i in xrange(ss.shape[0]):
        m, n = ss[i]
        m, n = bravais2cartesian(m, n, covera)
        # store the slip system
        ss_cart[i, 0, :] = m
        ss_cart[i, 1, :] = n
        matrixP.append(0.5*(np.outer(n, m) + np.outer(m, n)))
    return matrixP, ss_cart


def fitnessFunc(Ci, T, wgt, ss, NoN):
    # total dislocation density is use as a penalty term here
    T_calc = np.zeros((3, 3))
    for i, ci in enumerate(Ci):
        b = ss[i, 0, :]
        n = ss[i, 1, :]
        vctr_ci = np.dot(NoN, n) - n  # vctr c_i in the formular
        T_calc += Ci[i] * np.outer(b, vctr_ci)

    residual = np.sqrt(np.mean(np.square(T_calc - T)))

    return residual + sum(np.absolute(Ci))*wgt


def calculate(refVoxel, neiVoxel, wgt, initGuess, ss, disp=False):
    # wgt is the importance of keeping total dislocation density low
    msg = "Importance of keeping low density:\twgt<-{:.2E}\n".format(wgt)
    msg += "Initial dislocation density guess:\tigs<-{:.2E}".format(initGuess)
    if disp:
        print msg

    # get interface normal, N
    N_aps = refVoxel.pos - neiVoxel.pos
    N_ref = np.dot(refVoxel.orientationMatrix.T, N_aps)

    # get the prefactor
    NoN_ref = np.outer(N_ref, N_ref)
    if disp:
        print "Interface plane normal (in ref) is: ", N_ref

    # build the target Frank-Bilby tensor (T)
    # Note: the bilby tensor is used to find the difference of the
    #       sampling vector (x,y,z)@xtl difference between reference
    #       voxel and neighbor
    cnt = 0
    T = np.eye(3) - np.dot(refVoxel.orientationMatrix.T,
                           neiVoxel.orientationMatrix)  # nei->aps->ref
    Ci = np.ones(24) * initGuess

    # Estimate density using single dislocation wall model
    # disorientation in rad
    estmDensity = refVoxel.getDisorientationFrom(neiVoxel)[0]

    # use adaptive wgt
    while cnt < 100:
        cnt += 1
        refine = minimize(fitnessFunc, Ci,
                          args=(T, wgt, ss, NoN_ref),
                          tol = 1e-8,
                          method='BFGS',
                          options={'disp': disp,
                                   'maxiter': int(1e8)})
        print refine

        cis = np.array(refine.x)
        dT = refine.fun-sum(cis)*wgt
        if sum(abs(cis)) > estmDensity*10:
            wgt = wgt*2
            print "***new weight", wgt
        else:
            break

    if disp:
        print "*"*20 + "\nci \tb\tn"
        for i in xrange(ss.shape[0]):
            if i in [0, 3, 6, 12]:
                print "*"*20
            print "{:+2.4E}: \t".format(cis[i]),
            print "[{:2.4},{:.3f},{:.3f}]\t".format(*ss[i, 0, :]),
            print "({:.3f},{:.3f},{:.3f})".format(*ss[i, 1, :])
    return cis, dT, wgt


# ----- MAIN ----- #
parser = docopt(__doc__, version="1.0.0")
datafile = parser['<streakData>']
wgt = float(parser['--weight'])
initDensity = float(parser['--density'])

# read in data
with open(datafile) as f:
    data = [map(float, line.split()) for line in f.readlines()[1:]]
data = np.array(data)

# process each voxel to get density per slip system
matrixP, ss_cart = getSlipSystem()
density = []
dTs = []
mywgts = []

for voxel in data:
    print voxel
    voxelRef = VoxelData(voxel[0:3], voxel[3:6])
    voxelNei = VoxelData(voxel[6:9], voxel[9:12])
    cis, dT, mywgt = calculate(voxelRef, voxelNei,
                               wgt, initDensity, ss_cart,
                               disp=True)
    density.append(cis)
    dTs.append(dT)
    mywgts.append(mywgt)

# prepare output
outstr = "1 header\n"
outstr += "\t".join(["{}_pos".format(i+1) for i in xrange(3)]) + "\t"
outstr += "\t".join(["{}_eulers".format(i+1) for i in xrange(3)]) + "\t"
outstr += "\t".join(["{}_rho".format(i+1) for i in xrange(24)]) + "\t"
outstr += "dT\twgt\n"
for i, voxel in enumerate(data):
    ci = density[i]
    # clearly this point does not have valid streak
    # data (total density should not be overly large)
    if sum(abs(ci)) > 1e-1:
        continue
    outstr += "\t".join(map(str, voxel[0:3])) + "\t"
    outstr += "\t".join(map(str, voxel[3:6])) + "\t"
    outstr += "\t".join(map(str, density[i])) + "\t"
    outstr += str(dTs[i]) + "\t" + str(mywgts[i]) + "\n"

with open('batchFBEresults.txt', 'w') as f:
    f.write(outstr)

# just to get some idea what the density looks like
tmp = np.array(density)
print np.amax(tmp, axis=0)
print np.amin(tmp, axis=0)
tmp = np.absolute(tmp)
print np.amax(tmp, axis=0)