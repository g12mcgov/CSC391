/* 
* @Author: grantmcgovern
* @Date:   2015-09-28 12:06:08
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-09-29 12:50:07
*/


#define TILE_WIDTH 16

#include <math.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// GPU Methods
__global__ void matrixMult (float *A, float *B, float *C, int width);
__global__ void wakeGPU(void);

/*
* random_populate(float *, int)
*
* Populates a matrix with random floats
*/
void random_populate(float *matrix, int N) {
   	for (int i = 0; i < N; i++) {
   		for(int j = 0; j < N; j++) {
   			matrix[i * N + j] = drand48() * 2;
   		}
   	}
}

/*
* print_matrix(int, int, float *)
*
* Prints a matrix in user friendly format
*/
void print_matrix(int N, float *matrix) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			printf("%lf\t", matrix[i * N + j]);
		}
		printf("\n");
	}
}

/*
* check_command_line_args(int, char *)
*
* Checks to see whether used used proper
* number of command line arguments.
*/
void check_command_line_args(int argc, char *argv[]) {
	/* 
	* Ensure command line args are limited to only 3
	* (Excluding program name)
	*/
	if(argc != 2) {
		printf("Invalid Number of Arguments\n");
		exit(1);
	}
	// Check command line args are positive
	int i = 1;
	for( ; i < argc; i++) {
		// We have a negative argument
		if(atoi(&argv[i][0]) < 0) {
			printf("Invalid Argument: Negative Number given for Matrix Dimensions \n");
			exit(1);
		}
	}
}

/*
* write_to_file(int, int, float *)
*
* Writes a matrix to a '.dat' file
*/
void write_to_file(int N, float *C) {
	FILE *fp;
	fp = fopen("./product.dat", "w");

	// Error opening file
	if(fp == NULL) {
		printf("Error opening file for writing");
		exit(1);
	}
	fprintf(fp, "\nMatrix Product:\n");
	// Write Matrix to file
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			fprintf(fp, "%lf\t", C[i * N + j]);
		}
		fprintf(fp, "\n");
	}
	// Close file
	fclose(fp);
}

/*
* MAIN
*/
int main(int argc, char *argv[]) {
	// Seed random time agaist sys time
	srand48(time(NULL));

	// Check command line arguments
	check_command_line_args(argc, argv);
	
	// Get Matrix dimensions
    int N = atoi(argv[1]);

    // Get size of memory needed
    int size = sizeof(float) * N * N;

    // Declare Host Matrices
	float *A = (float*)malloc(size);
	float *B = (float*)malloc(size);

	// Populate matrices with random numbers
	random_populate(A, N);
	random_populate(B, N);

	// Compute + Allocate size for product matrix
	float* C = (float*)malloc(size);

	// Print Matrices
	printf("\nMatrix A:\n");
	print_matrix(N, A);
	printf("\nMatrix B:\n");
	print_matrix(N, B);

	// Allocate Device Memory
	float* dev_A;
	float* dev_B;
	float* dev_C;

	// Allocate that memory on the GPU
	cudaMalloc((void**) &dev_A, size);
	cudaMalloc((void**) &dev_B, size);
	cudaMalloc((void**) &dev_C, size);

	// Copy Host to Device
	cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

	// setup execution parameters
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid((int)ceil(N/dimBlock.x), (int) ceil(N/dimBlock.y));

	
	// Timing program execution
	clock_t start;
	clock_t stop;
	
	// Wake up GPU
	wakeGPU<<<1, 1>>>();

	start = clock();

	// execute the kernel
	matrixMult<<< dimGrid, dimBlock >>>(dev_A, dev_B, dev_C, N);

	// Synchronize threads, make sure all finished before copying memory back
	cudaThreadSynchronize();

	stop = clock();

	// Copy device memory back to host
	cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

	// Print product matrix
	printf("\nMatrix C:\n");
	print_matrix(N, C);

	// Display time taken
	double time_taken = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("\nExecuted in: %lf seconds\n", time_taken);

	// Write to file
	write_to_file(N, C);

	// Free memory on host + GPU
	free(A);
	free(B);
	free(C);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C); 
}

/*
* matrixMult(float *, float *, float *, int)
*
* Performs matrix multiplication for matrices of floats
*/
__global__ void matrixMult(float *A, float *B, float *C, int width) {
 	float sum = 0;
 	int k = 0;

	int col = blockIdx.x*TILE_WIDTH + threadIdx.x;
 	int row = blockIdx.y*TILE_WIDTH + threadIdx.y; 

	if(col < width && row < width) {
		for (k = 0; k < width; k++) {
			sum += A[row * width + k] * B[k * width + col];
		}
		C[row * width + col] = sum;
	}
}

/*
* wakeGPU(void)
*
* Assures the GPU is alive when we want to use it. 
*/
__global__ void wakeGPU(void) {
	printf("\nGPU Alive!\n");
}