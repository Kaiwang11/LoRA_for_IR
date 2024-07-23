# run all dataset on Multiple matrix re-weighting (MMRW)
python src/main.py -d fiqa -exp_type multi -p True -exp_name MMRW  
python src/main.py -d nfcorpus -exp_type multi -p True -exp_name MMRW
python src/main.py -d scifact -exp_type multi -p True -exp_name MMRW
python src/main.py -d nq -exp_type multi -p True -exp_name MMRW
