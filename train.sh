#!/usr/bin/env bash

for i in $(seq 1 10);
do
python E65_train.py dataset/reentrancy_1671.txt | tee logs/reentrancy_1671.txt_"$i".log;
done

for i in $(seq 1 10);
do
python E65_train.py dataset/infinite_loop_1317.txt | tee logs/infinite_loop_1317.txt_"$i".log;
done
