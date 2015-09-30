#!/bin/bash

# compile program
make clean
make

./MatrixMult 100 2>&1 | tee -a output.txt
./MatrixMult 500 2>&1 | tee -a output.txt
./MatrixMult 1000 2>&1 | tee -a output.txt
./MatrixMult 2000 2>&1 | tee -a output.txt
./MatrixMult 3000 2>&1 | tee -a output.txt