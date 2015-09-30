#!/bin/bash

# compile program
make clean
make

for i in {0..10..2}
do
	./MatrixMult "$i" >> output.txt 2>&1
done