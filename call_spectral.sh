#!/bin/bash

# This script consist a few consecutive commands that calls
# the spectral solver to do some processing and wait for it
# to finish, then call the post-processing script to extract
# stress--strain data (volume average) and store in a ASCII
# table, which will be read by the main python script to
# compute the cumulative absolute difference between
# simulation and experiment.

GEOM_FILE='cali.geom'
LOAD_FILE='cali.load'
MATERIAL='material.config'
OPT='optimizer.txt'
ODF='avg.linearODF'

# a little clean up before starting a new round
if [ -d workbench ]; then
    rm -rv workbench  # maybe should keep something?
fi

# make a working directory
mkdir workbench

# move all files in position (geom, load, material, optimizer)
cp ${GEOM_FILE} ./workbench/${GEOM_FILE}
cp ${LOAD_FILE} ./workbench/${LOAD_FILE}
cp ${MATERIAL} ./workbench/${MATERIAL}
cp ${OPT} ./workbench/${OPT}  # something containing the CRSS etc.
cp ${ODF} ./workbench/${ODF}  # ODF file to populate orientations

# call the spectral solver with corresponding options
cd workbench
DAMASK_spectral  --geom ${GEOM_FILE}  --load ${LOAD_FILE}

# post processing
postResults --cr f,p --co resistance_slip cali_cali.spectralOut
cd postProc
addCauchy *.txt  # should have only one text file herea
addMises -s Cauchy *.txt
cp *.txt ../ascii_table.txt

# now pass the ball back to the python script...