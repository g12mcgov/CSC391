#define TILE_WIDTH 2

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

__global__ void matrixMult (float *A, float *B, float *C, int width);

// Allocates a matrix with random float entries.
void randomInit(float *matrix, int size) {
   	for (int i = 0; i < size; i++) {
   		float random = rand() % 20;
   		matrix[i] = random;
   	}
}

void print_matrix(int rows, int columns, float *matrix) {
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
			printf("%lf\t", matrix[i * columns + j]);
		}
		printf("\n");
	}
}

/*
* check_command_line_args(int)
*
* Checks to see whether used used proper
* number of command line arguments.
*/
void check_command_line_args(int argc, char *argv[]) {
	/* 
	* Ensure command line args are limited to only 3
	* (Excluding program name)
	*/
	if(argc != 5) {
		printf("Invalid Number of Arguments\n");
		exit(1);
	}
	// Check matrices are valid for multiplication
	if(atoi(argv[2]) != atoi(argv[3])) {
		printf("Columns of Matrix A do not match rows of Matrix B\n");
		exit(1);
	}
	// Check command line args are positive
	int i = 0;
	for(i = 0; i < argc; i++) {
		// We have a negative argument
		if(atoi(&argv[i][0]) < 0) {
			printf("Invalid Argument: Negative Number given for Matrix Dimensions \n");
			exit(1);
		}
	}
}

void write_to_file(int matrixC_rows, int matrixC_columns, int *C) {
	FILE *fp;
	fp = fopen("./product.dat", "w");

	// Error opening file
	if(fp == NULL) {
		printf("Error opening file for writing");
		exit(1);
	}
	fprintf(fp, "\nMatrix Product:\n");
	// Write Matrix to file
	for(int i = 0; i < matrixC_rows; i++) {
		for(int j = 0; j < matrixC_columns; j++) {
			fprintf(fp, "%d\t", C[i * matrixC_columns + j]);
		}
		fprintf(fp, "\n");
	}
	// Close file
	fclose(fp);
}

int * populate_matrix(int *matrix, int matrix_rows, int matrix_columns) {
	// Seed random against sys clock
	srand(time(NULL));
	// Indeces
	int i = 0;
	int j = 0;

	// Populate Matrix A
	for(i = 0; i < matrix_rows; i++) {
		for(j = 0; j < matrix_columns; j++) {
			matrix[i * matrix_columns + j] = rand() % 20;
		}
	}
}

int main(int argc, char *argv[]) {
	srand(time(NULL));

	// Check command line arguments
	check_command_line_args(argc, argv);
	
	// Get Matrix A dimensions
    int matrixA_rows = atoi(argv[1]);
    int matrixA_columns = atoi(argv[2]);
    
    // Get Matrix B dimensions
    int matrixB_rows = atoi(argv[3]);
    int matrixB_columns = atoi(argv[4]);

    // Product Matrix dimensions
    int matrixC_rows = matrixA_columns;
	int matrixC_columns = matrixB_rows;

    // Get size of memory needed
    int size_A = sizeof(float) * matrixA_rows * matrixA_columns;
    int size_B = sizeof(float) * matrixB_rows * matrixB_columns;

    // Declare Host Matrices
	float *A = (float*)malloc(size_A);
	float *B = (float*)malloc(size_B);

	// Initialize host memory
	randomInit(A, matrixA_rows * matrixA_columns);
	randomInit(B, matrixB_rows * matrixB_columns);

	print_matrix(matrixA_rows, matrixA_columns, A);
	print_matrix(matrixB_rows, matrixB_columns, B);

	// Allocate Device Memory
	float* dev_A;
	float* dev_B;
	cudaMalloc((void**) &dev_A, size_A);
	cudaMalloc((void**) &dev_B, size_B);

	// Copy Host to Device
	cudaMemcpy(dev_A, A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, size_B, cudaMemcpyHostToDevice);

	unsigned int size_C = sizeof(float) * matrixC_rows * matrixC_columns;
	float* C = (float*) malloc(size_C);

	float* dev_C;
	cudaMalloc((void**) &dev_C, size_C);

	// setup execution parameters
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); 
	dim3 dimGrid((int)ceil(matrixC_columns/dimBlock.x), (int)ceil(matrixC_columns/dimBlock.y));

	// execute the kernel
	matrixMult<<< dimGrid, dimBlock >>>(dev_A, dev_B, dev_C, matrixC_columns);

	cudaMemcpy(C, dev_C, size_C, cudaMemcpyDeviceToHost);

	print_matrix(matrixC_rows, matrixC_columns, C);

	free(A);
	free(B);
	free(C);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C); 
}

__global__ void matrixMult(float *A, float *B, float *C, int width) {
 	float sum = 0;
 
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int k = 0;

	if(col < width && row < width) {
		for (k = 0; k < width; k++) {
			sum += A[row * width + k] * B[k * width + col];
		}
		C[row * width + col] = sum;
	}
}