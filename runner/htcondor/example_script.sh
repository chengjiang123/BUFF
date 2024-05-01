#!/bin/bash

mkdir -p ~/.local/bin
export PATH=~/.local/bin:"${PATH}"
curl -sL https://raw.githubusercontent.com/matthewfeickert/cvmfs-venv/main/cvmfs-venv.sh -o ~/.local/bin/cvmfs-venv
chmod +x ~/.local/bin/cvmfs-venv

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
. "${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh" -3 --quiet

echo "# lsetup 'views LCG_105 x86_64-el9-gcc11-opt'"
lsetup 'views LCG_105 x86_64-el9-gcc11-opt'

cvmfs-venv lcg-example
 . lcg-example/bin/activate

python -m pip install torchcfm

echo "start"
cd /path/to/current/folder
for i in {0..9}; do
   echo hi ${i}
   python example_flowbdt.py --iter ${i}
   mv /path/to/current/folder/models_${i}.pkl /path/to/destination/folder
done


echo "finished"
