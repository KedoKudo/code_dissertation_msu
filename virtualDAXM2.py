#!/usr/bin/env python
# coding=utf-8

# NOTE:
# This version is using an improved version of the original
# strain quantification library (cyxtal -> daxmexplorer).
# The newer library provides more stable and robust strain
# quantification capability.

"""
Use virtual experiment to test the strain refinement algorithm.

Usage:
    virtualDAXM2.py    run        (JOBFILE) [-n]
    virtualDAXM2.py    -h | --help
    virtualDAXM2.py    --version

Options:
    -h --help            Show this screen.
    --version            Show version.
    -v --verbose         Detalied output.
    -n --new             Create new archive
"""

import sys
import json
import damask
import numpy as np
import pandas as pd
from copy import deepcopy
from numpy.linalg import norm
from docopt import docopt
from damask import Quaternion
from cyxtal import get_reciprocal_base
from daxmexplorer.voxel import DAXMvoxel
from daxmexplorer.vecmath import random_three_vector
from daxmexplorer.vecmath import normalize
from daxmexplorer.cm import get_deviatoric_defgrad


def read_config(configJSON):
    """read in config file in JSON"""
    with open(configJSON) as jsondata:
        configinfo = json.load(jsondata)
    return configinfo


def make_random_defgrad(angR, magU):
    """
    F = make_random_defgrad(ang, u)

    Generate a random deformation gradient with given angle (degree) and 
    stretch magnitude.     // F = RU
    """
    axis = random_three_vector()
    rotation = Quaternion().fromAngleAxis(angR, axis, degrees=True).asMatrix()
    udeviation = magU*(np.random.random(6)-0.5)
    udeviation = udeviation[np.array([0,5,4,5,1,3,4,3,2])].reshape(3,3)
    # stretch = np.identity(3)+magU*(np.random.random((3, 3)) - 0.5)  # this strech will make shear larger
    # return np.dot(rotation, 0.5*(stretch + stretch.T))
    stretch = np.identity(3) + udeviation
    return np.dot(rotation, stretch)


def perturb_vector(vector, ang, axis=None):
    """rotate the given vector by ang (degree) with a random axis"""
    # NOTE: the current DAXM setup => 409.6 × 409.6 mm2, 2.048 × 2.048 pixels
    #       90 deg reflection geometry 510.3 mm above the sample
    # assuming 1/10 pixel uncertainty on the detector, we have
    # delta_ang_max = 409.6/2048/10/510.3 = 0.000039 rad = 0.002 deg
    # construct the pertubation vector
    axis = random_three_vector() if axis is None else axis
    qw = np.cos(np.radians(ang/2.0))
    qx, qy, qz = np.array(axis) * np.sin(np.radians(ang/2.0))
    q = Quaternion(quatArray=[qw, qx, qy, qz]).normalized()
    return q*np.array(vector)


def random_select_npeaks(visible_hkls, n):
    """return n visible hkls based on give list of visible hkls."""
    df = pd.DataFrame(visible_hkls)
    df.columns = ['h', 'k', 'l']
    df['wgt'] = norm(visible_hkls, axis=1)
    return df.sample(n=n)[['h','k','l']].as_matrix()


def calc_visible_peaks(q_xtal,                   # orientation (quanternion) 
                       k0,                       # incident beam (lab)
                       n_detector,               # detector normal (lab)
                       E_xray,                   # in KeV
                       lattice_constants,        # in nm
                       diffractable_hkls,        # list of potential HKLs
                       detector_angularRange=22, # in degrees,
                      ):
    """
    Return a sorted list of hkls that is theoretically visible on the 
    detector with given k0 and detector normal.
    """
    # calculate wave length range based on given energy (in A)
    # https://en.wikibooks.org/wiki/Xray_Crystallography/Foundations
    # https://en.wikipedia.org/wiki/Photon_energy
    wavenumber_low, wavenumber_high = np.sort(np.array(E_xray)/12.398)  # in 1/A
    detectable = np.cos(np.radians(detector_angularRange))
 
    # expanding lattice constants
    a, b, c, alpha, beta, gamma =  lattice_constants
    recipUnit = 1.0/(a*10.0)  # nm -> 1/A
    
    # rotate k0 and n_detector to xtal frame (easy to workwith)
    k0 = q_xtal * k0
    n_detector = q_xtal * n_detector
    
    # go over each hkl to see if it hits the detector
    visible_peaks = []
    for i, hkl in enumerate(diffractable_hkls):
        recip_hkl = hkl*recipUnit
        # distance to the large Ewald sphere center
        dist2LEWcntr = np.linalg.norm(recip_hkl-wavenumber_high*(-k0))
        # distance to the small Ewald sphere center
        dist2SEWcntr = np.linalg.norm(recip_hkl-wavenumber_low *(-k0))
        # test if within Ewald spheres
        if dist2LEWcntr<=wavenumber_high and dist2SEWcntr>= wavenumber_low:
            # test if it hits detector
            n_hkl = recip_hkl/np.linalg.norm(recip_hkl)
            # quick hack for cubic materials
            wavenumber_hkl = np.linalg.norm(recip_hkl)/2.0/np.dot(-k0, n_hkl)  
            k = wavenumber_hkl * k0 + recip_hkl
            n_k = k/np.linalg.norm(k)
            if np.dot(n_k, n_detector) > detectable:
                visible_peaks.append(hkl)

    thePeaks = []
   
    # use collinearity to prune duplicated peaks
    visible_peaks.sort(key=lambda x: np.dot(x,x), reverse=True)
    for i,hkl in enumerate(visible_peaks):
        hasDup = False
        for j,duplicate in enumerate(visible_peaks[i+1:]):
            hasDup = abs(1.0 - abs(np.dot(normalize(hkl), 
                                          normalize(duplicate)
                                         )
                                  )
                        ) < 1e-4
            if hasDup: break
        if not hasDup: thePeaks.append(hkl)

    # sorted to make the low index one at the head of the list
    thePeaks.sort(key=lambda x: np.dot(x,x))    

    return thePeaks


def grad_student(jobConfig):
    """The poor grad student who will perform the DAXM experiment"""
    n_voxels = int(jobConfig["n_voxels"])
    hkls = pd.read_csv(jobConfig["DAXMConfig"]["hkllist"], sep='\t')[['h', 'k', 'l']].as_matrix()

    # extract config for each part
    DAXMConfig = jobConfig["DAXMConfig"]
    labConfig = jobConfig["labConfig"]
    StrainCalcConfig = jobConfig["StrainCalcConfig"]

    # setup virtual DAXM lab
    k0 = np.array(labConfig['k0'])  # incident beam direction
    n_CCD = np.array(labConfig['n_CCD'])  # detector normal
    E_xray = np.array(labConfig["X-ray Energy(KeV)"])  # in KeV
    detector_angularRange = float(labConfig["detector angular range"])  # in degrees

    # material: Al
    # NOTE: the shortcut used in the visible peak calculation requires a cubic material
    lc_AL = np.array([0.4050, 0.4050, 0.4050, 90, 90, 90])  # [nm,nm,nm,degree,degree,degree]

    # write header for the output file
    if ARGS["--new"]:
        with open(jobConfig["dataFileName"], 'w') as f:
            headerList = ["voxelName", 
                        "angR", "magU",
                        "n_visible", "n_fullq", "peakNoiseSigma",
                        ]
            headerList += ["{}_F0".format(i+1) for i in range(9)]   # target F
            headerList += ["{}_FL2".format(i+1) for i in range(9)]  # least-squares guess
            headerList += ["{}_Fopt".format(i+1) for i in range(9)]
            headerList += ["opt_successful", "opt_fun", "opt_nfev"]

            f.write("\t".join(headerList) + "\n")
    
    # convert the parameter range to log space
    _angR_lb, _angR_ub = map(np.log10, DAXMConfig["angR range"])
    _magU_lb, _magU_ub = map(np.log10, DAXMConfig["magU range"])
    
    for i in range(n_voxels):
        # _DAXMConfig is a per Voxel configuration
        _DAXMConfig = deepcopy(DAXMConfig)

        # generate visible peaks (planes, hkls)
        ## get n_visiblePeaks and n_fullQ: N(n_indexedPeaks) >= n(n_fullQ)
        unacceptable = True
        while unacceptable:
            n_indexedPeaks = int(np.random.choice(DAXMConfig["n_indexedPeaks"], 1))
            n_fullQ = int(np.random.choice(DAXMConfig["n_fullQ"], 1))
            if n_fullQ <= n_indexedPeaks:
                unacceptable = False
        ## get list of index planes based on a random crystal orientation
        unacceptable = True
        while unacceptable:
            xtal_orientation = Quaternion.fromRandom()
            visible_hkls = calc_visible_peaks(xtal_orientation, 
                                              k0, n_CCD, E_xray, 
                                              lc_AL, hkls,
                                              detector_angularRange=detector_angularRange,
                                              )
            if len(visible_hkls) > n_indexedPeaks:
                indexed_plane = random_select_npeaks(visible_hkls, n_indexedPeaks)
                # prevent degenerated matrix due to parallel hkls
                if np.linalg.matrix_rank(indexed_plane) >=3:
                    unacceptable = False
        ## get scattering vectors
        # NOTE:
        # Since the crystal orientation is only useful for selecting diffractable planes,
        # the rest of the calculation (q and q0) can be directly done within xtal lattice
        # frame (XTL) regardless of the actual crystal orientation (makes the calcualtion
        # a lot simpler).
        ### frist, randomly generate a deformation gradient in xtal frame
        _DAXMConfig["angR"] = np.power(10, np.random.random()*(_angR_ub - _angR_lb) +  _angR_lb)
        _DAXMConfig["magU"] = np.power(10, np.random.random()*(_magU_ub - _magU_lb) +  _magU_lb)
        defgrad = make_random_defgrad(_DAXMConfig['angR'], _DAXMConfig['magU'])
        ### use the randomly generated defgrad to strain the scattering vectors
        recip_base = get_reciprocal_base(lc_AL)
        T, Bstar = defgrad, recip_base
        Tstar = np.transpose(np.linalg.inv(T))
        Bstar_strained = np.dot(Tstar, Bstar)
        qs = np.zeros((3, n_indexedPeaks))
        # random select n_fullQ to be full scattering vectors
        idx_fullq = np.random.choice(n_indexedPeaks, n_fullQ, replace=False)
        for arrayidx, milleridx in enumerate(indexed_plane):
            q_strained = np.dot(Bstar_strained, milleridx)
            qs[:, arrayidx] = q_strained if arrayidx in idx_fullq else normalize(q_strained)

        # setup DAXMConfig for each run
        # NOTE:
        # Noise analysis can be done when we have more access to the facility
        _DAXMConfig["whiteNoiseLevel"] = np.random.choice(np.array(DAXMConfig["peakPositionUncertainty/deg"]))

        qs_correct = np.copy(qs)
        for qsIdx in range(qs.shape[1]):
            qs[:, qsIdx] = perturb_vector(qs[:, qsIdx], 
                                          np.random.normal(scale=_DAXMConfig["whiteNoiseLevel"]),
                                          )

        # construct the voxel
        thisVoxel = DAXMvoxel(name="voxel_{}".format(i), 
                              ref_frame='XTL', 
                              coords=np.zeros(3), 
                              pattern_image="voxel_{}".format(i),
                              scatter_vec=qs,
                              plane=indexed_plane.T,
                              recip_base=recip_base,
                              peak=np.random.random((2,n_indexedPeaks)),
                              depth=0,
                              lattice_constant=lc_AL,
                              )

        # save the voxel to H5 archive
        thisVoxel.write(h5file=jobConfig["voxelArchive"])

        # calculate the deformation gradient
        # NOTE:
        # notice the much cleaner and simpler interface in terms of strain quantification
        defgrad_L2 = thisVoxel.deformation_gradientL2()
        defgrad_opt = thisVoxel.deformation_gradient_opt(eps=1e0)

        # export data to file for analysis
        data = [thisVoxel.name, 
                _DAXMConfig["angR"], _DAXMConfig["magU"],
                n_indexedPeaks, n_fullQ, _DAXMConfig["whiteNoiseLevel"],
                ]
        data += list(defgrad.flatten())      # in XTAL frame!!!
        data += list(defgrad_L2.flatten())   # in XTAL frame!!!
        data += list(defgrad_opt.flatten())  # in XTAL frame!!!

        data += [int(thisVoxel.opt_rst.success),  # is optimization successful
                 thisVoxel.opt_rst.fun,           # fitness function final val
                 thisVoxel.opt_rst.nfev,          # number of iteration used
                ]

        with open(jobConfig["dataFileName"], 'a') as f:
            f.write("\t".join(map(str, data)) + "\n")

        # interactive monitoring
        if jobConfig["monitor"] and i%1000 == 0:
            sys.stderr.write("|")
        elif jobConfig["monitor"] and i%100 == 0:
            sys.stderr.write("*")
            
"""

        print "\nTESTING"
        print "correct fitval"
        print np.sqrt(np.mean(np.sqrt(np.linalg.norm(qs_correct - qs, axis=0))))
        print "optimized fitval"
        print thisVoxel.opt_rst.fun
        print "target FD"
        print get_deviatoric_defgrad(defgrad)
        print "optmized FD"
        print get_deviatoric_defgrad(defgrad_opt)
        print "quality:"
        print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))


        # iteratively kicking bad one out
        tmpFs = []
        print "1 header"
        print "a b"
        for endidx in range(qs.shape[1]):
            selectedIdx = np.random.choice(qs_correct.shape[1], qs_correct.shape[1]-1, replace=False)
            # construct the voxel
            thisVoxel = DAXMvoxel(name="voxel_{}".format(i), 
                                  ref_frame='XTL', 
                                  coords=np.zeros(3), 
                                  pattern_image="voxel_{}".format(i),
                                  scatter_vec=qs[:, selectedIdx],
                                  plane=indexed_plane.T[:, selectedIdx],
                                  recip_base=recip_base,
                                  peak=np.random.random((2,n_indexedPeaks)),
                                  depth=0,
                                  lattice_constant=lc_AL,
                                  )

            # save the voxel to H5 archive
            thisVoxel.write(h5file=jobConfig["voxelArchive"])

            # calculate the deformation gradient
            # NOTE:
            # notice the much cleaner and simpler interface in terms of strain quantification
            defgrad_L2 = thisVoxel.deformation_gradientL2()
            defgrad_opt = thisVoxel.deformation_gradient_opt(eps=1e0)
            print thisVoxel.opt_rst.fun, "\t", np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))

        return 1
        
        tmpFs.append(defgrad_opt.flatten())
        tmpdeviations_tmpFs = map(lambda x: np.linalg.norm(x - np.average(tmpFs, axis=0)), tmpFs)
        print "stats:"
        print "std: ", np.std(tmpdeviations_tmpFs)
        print "avg: ", np.average(tmpdeviations_tmpFs)
        print "max: ", np.amax(tmpdeviations_tmpFs)
        print "min: ", np.amin(tmpdeviations_tmpFs)

        # check the outlier
        defgrad_opt = tmpFs[np.argmax(map(lambda x: np.linalg.norm(x - np.average(tmpFs, axis=0)), tmpFs))].reshape((3,3))
        print "*****OUTLIER*****"
        print "target FD"
        print get_deviatoric_defgrad(defgrad)
        print "optmized FD"
        print get_deviatoric_defgrad(defgrad_opt)
        print "quality:"
        print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))

        # check the one closest to avearge
        defgrad_opt = tmpFs[np.argmin(map(lambda x: np.linalg.norm(x - np.average(tmpFs, axis=0)), tmpFs))].reshape((3,3))
        print "*****Near Central*****"
        print "target FD"
        print get_deviatoric_defgrad(defgrad)
        print "optmized FD"
        print get_deviatoric_defgrad(defgrad_opt)
        print "quality:"
        print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))

        # use the average 
        defgrad_opt = np.average(tmpFs, axis=0).reshape((3,3))
        print "*****AVERAGE*****"
        print "target FD"
        print get_deviatoric_defgrad(defgrad)
        print "optmized FD"
        print get_deviatoric_defgrad(defgrad_opt)
        print "quality:"
        print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))

        # use the average without the outlier
        defgrad_opt = np.average([tmpFs[i] for i in range(len(tmpFs)) 
                                           if i != np.argmax(map(lambda x: np.linalg.norm(x - np.average(tmpFs, axis=0)), tmpFs))
                                 ], axis=0).reshape(3,3)
        print "*****AVERAGE WITHOUT OUTLIER*****"
        print "target FD"
        print get_deviatoric_defgrad(defgrad)
        print "optmized FD"
        print get_deviatoric_defgrad(defgrad_opt)
        print "quality:"
        print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))

        # hy


        # use the correct central
        cnt = 0
        tmpdeviations_tmpFs = map(lambda x: np.linalg.norm(x - np.average(tmpFs, axis=0)), tmpFs)
        mystd = np.std(tmpdeviations_tmpFs)
        tor = mystd/4.0
        while mystd > tor:
            tmpFs = [tmpFs[i] for i in range(len(tmpFs)) 
                              if i != np.argmax(tmpdeviations_tmpFs)
                    ]  # kick out the outlier
            tmpdeviations_tmpFs = map(lambda x: np.linalg.norm(x - np.average(tmpFs, axis=0)), tmpFs)

            mystd = np.std(tmpdeviations_tmpFs)

            cnt += 1
            if cnt > 10:
                break

        defgrad_opt = np.average(tmpFs, axis=0).reshape(3,3)
        print "*****force halfing std*****"
        print "target FD"
        print get_deviatoric_defgrad(defgrad)
        print "optmized FD"
        print get_deviatoric_defgrad(defgrad_opt)
        print "quality:"
        print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))
                             

        # print "test {}".format(i)
        # print defgrad
        # print defgrad_L2
        # print defgrad_opt
        # print np.linalg.norm(defgrad - defgrad_opt)
        # print "*"*20
        # print get_deviatoric_defgrad(defgrad)
        # print get_deviatoric_defgrad(defgrad_L2)
        # print get_deviatoric_defgrad(defgrad_opt)
        # print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_L2))
        # print np.linalg.norm(get_deviatoric_defgrad(defgrad) - get_deviatoric_defgrad(defgrad_opt))

        # export data to file for analysis
        data = [thisVoxel.name, 
                _DAXMConfig["angR"], _DAXMConfig["magU"],
                n_indexedPeaks, n_fullQ, _DAXMConfig["whiteNoiseLevel"],
                ]
        data += list(defgrad.flatten())      # in XTAL frame!!!
        data += list(defgrad_L2.flatten())   # in XTAL frame!!!
        data += list(defgrad_opt.flatten())  # in XTAL frame!!!

        data += [int(thisVoxel.opt_rst.success),  # is optimization successful
                 thisVoxel.opt_rst.fun,           # fitness function final val
                 thisVoxel.opt_rst.nfev,          # number of iteration used
                ]

        with open(jobConfig["dataFileName"], 'a') as f:
            f.write("\t".join(map(str, data)) + "\n")

        # interactive monitoring
        if jobConfig["monitor"] and i%1000 == 0:
            sys.stderr.write("|")
        elif jobConfig["monitor"] and i%100 == 0:
            sys.stderr.write("*")
    
    return None
""" 

if __name__ == "__main__":
    # parse interface
    ARGS = docopt(__doc__, version="4.0.0")

    if ARGS["run"]:
        jobConfig = read_config(ARGS["JOBFILE"])
        grad_student(jobConfig)
