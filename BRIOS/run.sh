chmod +x /mnt/data1tb/BRIOS/BRIOS/BRIOS/preprocess/preprocess_data_demo.py /mnt/data1tb/BRIOS/BRIOS/BRIOS/preprocess/split_data.py /mnt/data1tb/BRIOS/BRIOS/BRIOS/preprocess/prepare_train_data.py /mnt/data1tb/BRIOS/BRIOS/BRIOS/model_train.py

echo "preprocessing"
python3 /mnt/data1tb/BRIOS/BRIOS/BRIOS/preprocess/preprocess_data_demo.py

echo "split"
python3 /mnt/data1tb/BRIOS/BRIOS/BRIOS/preprocess/split_data.py

echo "prepare train"
python3 /mnt/data1tb/BRIOS/BRIOS/BRIOS/preprocess/prepare_train_data.py

echo "train"
python3 /mnt/data1tb/BRIOS/BRIOS/BRIOS/model_train.py