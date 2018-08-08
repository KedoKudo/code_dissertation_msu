!/usr/bin/env bash

# vic_bary.geom is the microstructure generated from cleaned 
# seed points using BI method

geom_canvas -g 162 226 152  -f 999 < vic_bary.geom \
|\
geom_grainGrowth -N 1  -d 4 -w -i 999 \
|\
geom_grainGrowth -N 3  -d 1    -i 999 \
|\
geom_canvas -g 142 206 130 \
|\
geom_canvas -g 140 200 120 -o 1 3 10 \
|\
geom_canvas -g 140 200 130 -f 0 \
|\
> vicBary.geom; \
geom_check vicBary.geom