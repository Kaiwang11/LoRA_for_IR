directory="./datasets"
subdir_count=$(find "$directory" -mindepth 1 -type d | wc -l)
if [ -z "$(ls -A "$directory")" ] || [ "$subdir_count" -lt 4 ]; then

	for dataset in 'fiqa' 'nfcorpus' 'scifact' 'nq'
	do
		python data_load.py -d $dataset
	done
else
	echo "All data is ready to run "
fi

data='scifact'
echo "Run simple LoRA train  on scifact in 1  epcoh  "
python src/main.py -d scifact -epoch 1 


echo "Run pure LoRA training  experiments on" $data
python src/main.py -d $data  
echo "Run SMRW experiments on" $data
python src/main.py -d $data -exp_type single -p True -exp_name SMRW

echo "Run MMRW experiments on" $data
python src/main.py -d $data -exp_type multi -p True -exp_name MMRW


