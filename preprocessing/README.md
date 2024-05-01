

# Preprocessing code (WiP)

Please checkout each folder to modify based on users' wills. 

## example usage

download the files somewhere from zenodo.
- JetNet: https://zenodo.org/records/6975118
- Calochallenge: https://calochallenge.github.io/homepage/
- Omnifold: https://zenodo.org/records/3548091
- SB_refinement: please go to https://github.com/SaschaDief/SB_refinement/tree/main for detailed instructions


```bash
# jetnet high-level features (specific files)
python preprocess_jetnet.py -m one -f your/path

# jetnet high-level features (all jet types)
python preprocess_jetnet.py -m all

# calochallenge ds1 photon
python preprocess_calo.py -f your/hdf5/path -d 1 -e 0.5

```
