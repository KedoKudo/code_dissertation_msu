#!/usr/bin/env python

##
# This script is an improved version of the previous auto_cali.py.
# In this version, a convergence criterion is given to stop the calibration
# once the simplex has converged into a single point (used to cause a lot
# of time in previous version).
# The nelder-mead implementation is from the ezxtal package.

import numpy as np
import random as rd
import sys
import os
import subprocess
from ezxtal import nelder_mead


#########
# Macro #
max_iter = 10000  # maximum number of run allowed for optimization
max_try = 100  # maximum number of run allowed for simplex initialization
torlerance = 1e-2  # threshold for optimization (GPa)
EXP_DATA = "./data/exp_load.txt"
SIM_DATA = "./workbench/ascii_table.txt"
OPT_FILE = "./optimizer.txt"
LOG_FILE = "./calibration.log"  # log file store the output of all data
SLIP_RESISTANCE_LOC = 26  # location of slip resistance in the post results
VON_MISES_LOC = 62  # location of von Mises stress in the post results
ETA_EXP = 0.81  # experimental ratio of CRSS_prism to CRSS_basal @Hongmei


def obj_func(vtx):
    ''' call the spectral solver to pull the model and extract the
        accumulated difference of stress between experiment and simulation.
    '''
    # put vtx in the optimizer
    update_config(vtx)
    # call the spectral solver and perform post processing
    # extract stress--strain data, effective CRSS for each slip system
    cmd = './call_spectral.sh'
    subprocess.call([cmd])
    # wait for simulation results
    # open results file and read in the data
    sim_raw = open(SIM_DATA).readlines()
    for i in range(int(sim_raw.pop(0).split()[0])):
        sim_raw.pop(0)  # remove header
    # get the stress difference
    sim_data = [float(line.split()[VON_MISES_LOC])/1e9 for line in sim_raw]
    # compute stress difference
    delta_sigma = abs(float(sum(np.array(sim_data) - np.array(exp_data))))
    # now calculate the averaged effective CRSS
    tmp_basal = np.mean(sim_data[SLIP_RESISTANCE_LOC+0 : SLIP_RESISTANCE_LOC+3])
    tmp_prism = np.mean(sim_data[SLIP_RESISTANCE_LOC+3 : SLIP_RESISTANCE_LOC+6])
    eta_sim = tmp_prism/tmp_basal
    return delta_sigma * np.exp(abs(eta_sim - ETA_EXP))


def vtx_check(vtx):
    ''' check whether given vertex is a physically meaningful one. '''
    # rules for vertex checking
    #   1. no negative values for CRSS
    #   2. no softening, i.e. crss_0 < crss_s
    if any([item < 0 for item in vtx]):
        vtx = [abs(item) for item in vtx]
    if any([item > 0 for item in np.array(vtx[0:5]) - np.array(vtx[5:10])]):
        for i in range(5):
            if vtx[i] > vtx[i+5]:
                vtx[i], vtx[i+5] = vtx[i+5], vtx[i]  # quick swap
    return vtx


def update_config(vtx_new):
    ''' write the new vertex to the ${OPT_FILE} '''
    crss_0 = vtx_new[0:5]
    crss_s = vtx_new[5:10]
    a_slip = vtx_new[10]
    h0_slipslip = vtx_new[11]
    outstr = "gdot0_slip\t0.001\t# reference shear rate\n"
    outstr += "n_slip\t50\t# might need some calibration\n"
    outstr += "tau0_slip\t{} {} {} {} {}\n".format(crss_0[0],
                                                   crss_0[1],
                                                   crss_0[2],
                                                   crss_0[3],
                                                   crss_0[4])
    outstr += "tausat_slip\t{} {} {} {} {}\n".format(crss_s[0],
                                                     crss_s[1],
                                                     crss_s[2],
                                                     crss_s[3],
                                                     crss_s[4])
    outstr += "a_slip\t{}\n".format(a_slip)
    outstr += "gdot0_twin\t0.001\t# no effect\n"
    outstr += "n_twin\t50\n"
    outstr += "tau0_twin\t1.06e9  1.06e9  1.06e9  1.06e9\n"
    outstr += "s_pr\t5.88e7\ntwin_b\t2\ntwin_c\t25\ntwin_d\t0.1\ntwin_e\t0.1\n"
    outstr += "h0_slipslip\t{}".format(h0_slipslip)
    # overwrite old OPT_FILE
    outfile = open(OPT_FILE, "w")
    outfile.write(outstr)
    outfile.close()


def update_log_vtx(vtx, f_val):
    ''' update the most recent vtx in the log file '''
    log_file = open(LOG_FILE, 'a')
    tmp_str = "\t".join([str(item) for item in vtx]) + "\t{}\n".format(f_val)
    log_file.write(tmp_str)
    log_file.close()


def update_log_simplex(simplex):
    ''' write entire simplex to the log file '''
    log_file = open(LOG_FILE, 'a')
    tmp_str = "*"*40 + "\n"
    for f_val in simplex.keys():
        vtx = simplex[f_val]
        tmp_str += "\t".join([str(item) for item in vtx]) + "\t{}\n".format(f_val)
    tmp_str += "*"*40 + "\n"
    log_file.write(tmp_str)
    log_file.close()


###################
# start of script #
# simplex of parameters
simplex = {}  # {f_value: vtx} for note keeping

# read the experiment data into memory
raw_exp = open(EXP_DATA).readlines()
for i in range(int(raw_exp.pop(0)[0])):
    raw_exp.pop(0)
exp_data = tuple([float(line.split()[-1]) for line in raw_exp])

print "initializing parameter simplex"
para_list = []
# reference from Hongmei+Leyun's work
# the 2nd prism slip is considered as the middle point for prism and <c+a>
sample = [153000000.0, 124000000.0, 186000000.0, 235000000.0,  353000000.0,
          188000000.0, 152000000.0, 228000000.0, 705000000.0, 1060000000.0,
          1.0,         470000000.0, "Nan"]
para_list.append(sample)
# prepare header
outstr = "1\theader\n"
outstr += "tau0_basal\ttau0_prism\ttau0_prism2\ttau0_pyra\ttau0_pyrca"
outstr += "taus_basal\ttaus_prism\ttaus_prism2\ttaus_pyra\ttaus_pyrca"
outstr += "a_slip\th0_slipslip\tdelta\n"
for i in range(len(sample)+1):
    for j in range(max_try):
        vtx_tmp = [item*(0.5+rd.random()) for item in sample[:-1]]
        # some test to make sure it make sense
        test = np.array(vtx_tmp[0:5]) - np.array(vtx_tmp[5:10])
        if all([item < 0 for item in test]):
            vtx_tmp.append('Nan')  # got one hit
            print vtx_tmp
            break
    para_list.append(vtx_tmp)
# write to the log file
for entry in para_list:
    outstr += "\t".join([str(me) for me in entry]) + "\n"
outfile = open(LOG_FILE, "w")
outfile.write(outstr)
outfile.close()

print "start optimization process"
print "process initial simplex"
raw_data = open(LOG_FILE, "r").readlines()
i_header = int(raw_data.pop(0)[0])
outstr = "{}\theader\n".format(i_header)
for i in range(i_header):
    tmp = raw_data.pop(0)
    outstr += tmp  # get header
# write header to log file
log_file = open(LOG_FILE, 'w')
log_file.write(outstr)
log_file.close()
# compute f_val for the initial simplex
for line in raw_data:
    # remove the last Nan holder
    vtx = [float(item) for item in line.split()][:-1]
    # call the spectral solver to evaluate the vertex
    f = obj_func(vtx)
    simplex[f] = vtx  # put initial vertex in the simplex
    update_log_vtx(vtx, f)

# initialization complete, start optimization
for key in simplex:
    # for later easy set operation
    simplex[key] = tuple(simplex[key])
for i in range(max_iter):
    print "@iter: {}\n".format(i)
    simplex = nelder_mead(simplex, obj_func, vtx_check)
    # write out all the vertex for the new simplex (wasteful, but straight forward)
    update_log_simplex(simplex)
    # test if already find the best solution
    best_f = sorted(simplex.keys())[0]
    if best_f < torlerance:
        # got the best hit here
        print "!!Got the best parameter set!!"
        best_vtx = simplex[best_f]
        update_log_vtx(best_vtx, best_f)
        break