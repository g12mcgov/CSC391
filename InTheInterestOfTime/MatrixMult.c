/* 
* @Author: grantmcgovern
* @Date:   2015-09-28 12:06:08
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-09-28 23:55:56
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_matrix(int rows, int columns, int *matrix) {
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
			printf("%d\t", matrix[i * columns + j]);
		}
		printf("\n");
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

void matrix_mult(int matrixA_rows, int matrixA_columns, int matrixB_rows, 	\
				 int matrixB_columns, int *A, int *B) {

	// Seed random against sys clock
	srand(time(NULL));
	// Indeces
	int i = 0;
	int j = 0;

	// Populate Matrix A
	for(i = 0; i < matrixA_rows; i++) {
		for(j = 0; j < matrixA_columns; j++) {
			A[i * matrixA_columns + j] = rand() % 20;
		}
	}

	// Populate Matrix B
	for(i = 0; i < matrixB_rows; i++) {
		for(j = 0; j < matrixB_columns; j++) {
			B[i * matrixB_columns + j] = rand() % 20;
		}
	}

	printf("\nMatrix A:\n");
	print_matrix(matrixA_rows, matrixA_columns, A);
	printf("\nMatrix B: \n");
	print_matrix(matrixB_rows, matrixB_columns, B);

	// Dimensions for product matrix
	int matrixC_rows = matrixA_columns;
	int matrixC_columns = matrixB_rows;

	/*
	* Declare new Matrix to recieve our answer
	* (In the form m x n)
	*/
	printf("\nMatrix C Dimensions: %d x %d \n", matrixC_rows, matrixC_columns);

	int *C = (int*)calloc(matrixC_columns * matrixC_columns, sizeof(int));
	for(i = 0; i < matrixC_rows; i++) {
		for(j = 0; j < matrixC_columns; j++) {
			C[i * matrixC_columns + j] = 0;
		}
	}

	print_matrix(matrixC_columns, matrixC_columns, C);

	int sum = 0;
	int k = 0;
	int width = matrixC_columns;

	for(int row = 0; row < matrixC_columns; row++) {
		for(int col = 0; col < matrixC_columns; col++) {
			for(int k = 0; k < width; k++) {
				C[row * width + col] += A[row * width + k] * B[k * width + col];
			}
		}
	}
	// Print Product Matrix
	printf("\nMatrix C:\n");
	print_matrix(matrixC_rows, matrixC_columns, C);
	// Write Product Matrix
	write_to_file(matrixC_rows, matrixC_columns, C);
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
	for(int i = 0; i < argc; i++) {
		// We have a negative argument
		if(atoi(&argv[i][0]) < 0) {
			printf("Invalid Argument: Negative Number given for Matrix Dimensions \n");
			exit(1);
		}
	}
}

int main(int argc, char *argv[]) {
	// Check the command line arguments
	check_command_line_args(argc, argv);
    
    // Get Matrix A dimensions
    int matrixA_rows = atoi(argv[1]);
    int matrixA_columns = atoi(argv[2]);
    
    // Get Matrix B dimensions
    int matrixB_rows = atoi(argv[3]);
    int matrixB_columns = atoi(argv[4]);
    
    // Declare Matrices
    int *A;
    int *B;

    // Index
    int i = 0;

    A = (int *)malloc(matrixA_rows * matrixA_columns * sizeof(int));
    B = (int *)malloc(matrixB_rows * matrixB_columns * sizeof(int));
    
    // Perform Multiplication
    matrix_mult(matrixA_rows, matrixA_columns, matrixB_rows, matrixB_columns, A, B);

    return 0;
}