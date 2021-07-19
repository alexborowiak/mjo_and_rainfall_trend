#!/bin/sh

set -eu
#source ~/.bash_profile
module use /g/data3/hh5/public/modules 
module load conda/analysis3-20.10
PATH_ALEX=/g/data/zv2/agcd/v1/precip/calib/r005/01day
filenum=0
which cdo
for f in $PATH_ALEX/*.nc
do
	echo $f
	cdo -sellonlatbox,110,160,-44,-10 -selmonth,1,2,3,10,11,12 -remapcon,awap_grid.grd $f $(printf '%d.nc' $((filenum)))
	((filenum=filenum+1))
done
