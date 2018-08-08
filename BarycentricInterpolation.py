#!/usr/bin/env python


import numpy as np 
import pandas as pd
from damask.orientation import Orientation 
from scipy import spatial


def calcBarycentricCoords(pt, verts):
    """calculate the Barycentric coordinates"""
    verts = np.array(verts)  # vertices formed by N+1 nearest voxels
    pt = np.array(pt)        # voxel of interest
    
    A = np.transpose(np.column_stack((verts, np.ones(verts.shape[0]))))
    b = np.append(pt, 1)
    
    return np.linalg.lstsq(A, b)[0]


# read in data
with open("seeds_cleaned.txt", 'r') as f:
    nHeaders = int(f.readline().split()[0])

seedsDF = pd.read_csv('seeds_cleaned.txt', sep='\t', skiprows=nHeaders)
seedsDF.describe()  # show summary info


# define a function to calculate the average Euler angles
def getAverageEulers(eulers):
    """return the average orientation of given list of euler angles"""
    olists = [Orientation(Eulers=np.radians(me), symmetry='hexagonal')
                          for me in eulers]
    return Orientation.average(orientations=olists).asEulers(degrees=True)

# assign orientation for each grain
# default 5 degree tolerance is used when assigning grain ID
grainIDs = seedsDF["grainID_eulers@5"].unique()
eulersLB = ["{}_eulers".format(i+1) for i in range(3)]

with open("texture.config", 'w') as f:
    txtstr = '<texture>\n'
    
    for i, gid in enumerate(grainIDs):
        tmp = seedsDF[seedsDF["grainID_eulers@5"] == gid]
        avgEuler = getAverageEulers(tmp[eulersLB].as_matrix())
        txtstr += "\n[Grain{}]\n".format(gid)
        txtstr += "(gauss) phi1 {} phi {} phi2 {} scatter 0.0 fraction 1.0\n".format(*avgEuler)
    
    f.write(txtstr)


geomGrids = np.zeros((142, 206, 130))

lbPos = ["{}_pos".format(i+1) for i in range(3)]
verts = seedsDF[lbPos].as_matrix()
gids = seedsDF["grainID_eulers@5"].as_matrix()
tree = spatial.KDTree(verts)

geomFilen = "vic_bary.geom"

with open(geomFilen, "w") as f:
    txtstr = """5 header
grid    a 142    b 206    c 130
size    x 142.0    y 206.0    z 130.0
origin    x 0.0    y 0.0    z -125.0
homogenization    1
microstructures    131
"""
    f.write(txtstr)
    
for z in range(130):
    print "\r{}/130".format(z),
    pts = [(i, j, z-125.0) 
                           for j in range(206)
                           for i in range(142)
          ]
    
    dsts, idxs = tree.query(pts, k=4)
    
    tmpgids = np.zeros(idxs.shape[0], dtype=np.int64)
    
    for k in range(idxs.shape[0]):
        dst = dsts[k,:]
        idx = idxs[k,:]
        
        candidateGIDs = gids[idx]
        candidateVerts = tree.data[idx]
        
        if z >= 0:
            tmpgids[k] = candidateGIDs[0]  # use neaest one for surface 
        else:
            wgts = calcBarycentricCoords(pts[k], candidateVerts)
            tmpgids[k] = candidateGIDs[np.argmax([sum(wgts[candidateGIDs==thisGid]) for thisGid in candidateGIDs])]
    
    tmpgids = tmpgids.reshape(142, 206)
    
    with open(geomFilen, 'a') as f:
        txtstr = "\n".join(["\t".join(map(str, row)) 
                             for row in tmpgids
                           ]) + "\n"
        f.write(txtstr)


