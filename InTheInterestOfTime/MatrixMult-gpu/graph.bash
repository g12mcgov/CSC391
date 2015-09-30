#!/bin/bash

make clean >/dev/null

MAX1=32
for j in {0..3000..100}
	do
	for (( i=2; i <= MAX1; i++))
		do
		sed -i "/define TILE_WIDTH/c\#define TILE_WIDTH $i" MatrixMult.cu
		make >/dev/null
		./MatrixMult "$j" | tee -a output.txt
	done
done