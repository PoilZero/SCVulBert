#!/usr/bin/env bash

for i in $(seq 1 5);
do
python E65_train.py dataset/delegatecall_196_directlly_from_dataset.txt | tee logs/delegatecall_196_directlly_from_dataset.txt_"$i".log;
python E65_train.py dataset/integeroverflow_275_directlly_from_dataset.txt | tee logs/integeroverflow_275_directlly_from_dataset.txt_"$i".log;
#python E65_train.py dataset/reentrancy_273_directlly_from_dataset.txt | tee logs/reentrancy_273_directlly_from_dataset.txt_"$i".log;
#python E65_train.py dataset/timestamp_349_directlly_from_dataset.txt | tee logs/timestamp_349_directlly_from_dataset.txt_"$i".log;
done

