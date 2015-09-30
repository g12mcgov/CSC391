#!/bin/bash

# make clean >/dev/null

# Matrix = 100
for i in {0..50..10}
	do
	sed -i "/define TILE_WIDTH/c\#define TILE_WIDTH $i" MatrixMult.cu
	make >/dev/null
	./MatrixMult 100 | tee -a output.txt
done

# Matrix = 500
for i in {0..50..10}
	do
	sed -i "/define TILE_WIDTH/c\#define TILE_WIDTH $i" MatrixMult.cu
	make >/dev/null
	./MatrixMult 500 | tee -a output.txt
done

# Matrix = 1000
for i in {0..50..10}
	do
	sed -i "/define TILE_WIDTH/c\#define TILE_WIDTH $i" MatrixMult.cu
	make >/dev/null
	./MatrixMult 1000 | tee -a output.txt
done

# Matrix = 2000
for i in {0..50..10}
	do
	sed -i "/define TILE_WIDTH/c\#define TILE_WIDTH $i" MatrixMult.cu
	make >/dev/null
	./MatrixMult 2000 | tee -a output.txt
done

# Matrix = 3000
for i in {0..50..10}
	do
	sed -i "/define TILE_WIDTH/c\#define TILE_WIDTH $i" MatrixMult.cu
	make >/dev/null
	./MatrixMult 3000 | tee -a output.txt
done