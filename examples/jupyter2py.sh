#########################################################################
# File Name: jupyter2py.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue May 30 23:26:40 2023
#########################################################################
#!/bin/bash

for afile in `ls */*.ipynb`
do
	echo $afile
	jupyter nbconvert --to script $afile
done
