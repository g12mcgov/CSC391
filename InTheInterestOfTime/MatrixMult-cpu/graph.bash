#!/bin/bash

# compile program
make clean
make

for i in {0..2000..2}
do
	./MatrixMult "$i" 2>&1 | tee -a output.txt
done