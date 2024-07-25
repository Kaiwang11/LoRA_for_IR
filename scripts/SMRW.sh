# run single matrix reweighting( SMRW) every datasets

python src/main.py -d nfcorpus -exp_type single -p True -exp_name SMRW 
python src/main.py -d fiqa -exp_type single -p True  -exp_name SMRW
python src/main.py -d scifact -exp_type single -p True  -exp_name SMRW
python src/main.py -d nq -exp_type single -p True  -exp_name SMRW
