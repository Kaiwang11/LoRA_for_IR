# we apply dora and vera . Then combine our method 
python src/main.py -d fiqa -dora True #pure dora
python src/main.py -d fiqa -dora True -exp_type MagSMRW  -p True # DoRA(magnitude vector)+ Single Matrix Re-Wwight

python src/main.py -d fiqa -vera True #pure vera

python src/main.py -d fiqa -vera True -exp_type ScalSMRW  -p True # VeRA(scale vector)+ Single Matrix Re-Wwight

