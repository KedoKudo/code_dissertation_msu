#!/usr/bin/env python


import numpy as np
from ConfigParser import SafeConfigParser
from cyxtal import Xtallite


class VoxelData(object):
    """
    Container class bundle data for dislocation content analysis,
    all var are in APS coordinate system
    """

    def __init__(self, pos, eulers, astar, bstar, cstar,
                 lattice='hexagonal'):
        self.pos = np.array(pos)
        self.eulers = np.array(eulers)
        self.astar = np.array(astar)
        self.bstar = np.array(bstar)
        self.cstar = np.array(cstar)
        self.lattice = lattice

    def __str__(self):
        msg = "Voxel@({},{},{})_APS".format(*self.pos)
        msg += " with EulerAngles:({},{},{})\n".format(*self.eulers)
        msg += "  g matrix=\n" + str(self.orientationMatrix)
        return msg

    def getDisorientationFrom(self, other, mode='angleaxis'):
        myXtal = Xtallite(eulers=self.eulers,
                          pt=self.pos,
                          lattice=self.lattice)
        neiXtal = Xtallite(eulers=other.eulers,
                           pt=other.pos,
                           lattice=other.lattice)
        return myXtal.disorientation(neiXtal, mode=mode)


# ---- Help functions ----- #
def parseVoxel(datafile, sectionName):
    reader = SafeConfigParser()
    reader.read(datafile)
    pos = map(float, reader.get(sectionName, 'coordinateAPS').split())
    eulers = map(float, reader.get(sectionName, 'eulersAPS').split())
    astar = map(float, reader.get(sectionName, 'astar').split())
    bstar = map(float, reader.get(sectionName, 'bstar').split())
    cstar = map(float, reader.get(sectionName, 'cstar').split())
    lattice = reader.get(sectionName, 'lattice')
    return VoxelData(pos, eulers, astar, bstar, cstar, lattice=lattice)


# ---- calculate density ----- #
def estimate(refVoxel, neiVoxel):
    # small angle approximation is used here tan\theta = \theta
    print "Estimate density using single dislocation wall model."
    # disorientation in rad
    diso = refVoxel.getDisorientationFrom(neiVoxel)[0]  
    ci = 2.0 / diso
    print "require 1 b every {} b step".format(int(ci)+1)
    print "target area density should around {:.4E}".format(diso/2.0)


datafile = parser['<streakData>']  # need to switch to input data

# parsing data from <streakData>
refVoxel = parseVoxel(datafile, 'reference')
neiVoxel = parseVoxel(datafile, 'neighbor')
stressAPS = getStressAPS(datafile)
lattice = refVoxel.lattice
p, ss = getSlipSystem(slipSystemName=lattice)

# call func to report density 
estimate(refVoxel, neiVoxel)
