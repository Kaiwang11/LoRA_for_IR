directory="./datasets"
if [ -z "$(ls -A "$directory")" ]; then

	for dataset in 'fiqa' 'nfcorpus' 'scifact' 'nq'
	do
		python data_load.py -d $dataset
	done
else
	echo "All data is ready to run "
fi
echo "Run simple LoRA train  on scifact in 1  epcoh  "
python src/main.py -d scifact -epoch 1 


