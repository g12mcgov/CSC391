#!/bin/bash

# compile program
make clean
make

for i in {0..500..2}
do
	./MatrixMult "$i" 2>&1 | tee -a output.txt
done

for i in {500..1000..8}
do
	./MatrixMult "$i" 2>&1 | tee -a output.txt
done

for i in {1000..1500..50}
do
	./MatrixMult "$i" 2>&1 | tee -a output.txt
done

for i in {1500..2000..100}
do
	./MatrixMult "$i" 2>&1 | tee -a output.txt
done

for i in {2000..2500..200}
do
	./MatrixMult "$i" 2>&1 | tee -a output.txt
done

for i in {2500..3000..500}
do
	./MatrixMult "$i" 2>&1 | tee -a output.txt
done