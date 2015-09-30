/* 
* @Author: grantmcgovern
* @Date:   2015-09-28 12:06:08
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-09-30 02:00:34
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_matrix(int N, float *matrix) {
	int i = 0;
	int j = 0;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			printf("%lf\t", matrix[i * N + j]);
		}
		printf("\n");
	}
}

void write_to_file(int N, float *C) {
	FILE *fp;
	fp = fopen("./product.dat", "w");

	// Error opening file
	if(fp == NULL) {
		printf("Error opening file for writing");
		exit(1);
	}
	fprintf(fp, "\nMatrix Product:\n");
	
	int i = 0;
	int j = 0;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf\t", C[i * N + j]);
		}
		fprintf(fp, "\n");
	}
	// Close file
	fclose(fp);
}

void matrix_mult(int N, float *A, float *B) {

	// Seed random against sys clock
	srand48(time(NULL));
	// Indeces
	int i = 0;
	int j = 0;

	// Populate Matrix A
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			A[i * N + j] = drand48();
		}
	}

	// Populate Matrix B
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			B[i * N + j] = drand48();
		}
	}

	// Print matrices
	printf("\nMatrix A:\n");
	print_matrix(N, A);
	printf("\nMatrix B: \n");
	print_matrix(N, B);

	/*
	* Declare new Matrix [C] to recieve our answer
	* (In the form m x n)
	*/
	float *C = (float*)calloc(N * N, sizeof(float));

	float sum = 0;
	int k = 0;

	// Actually perform the multiplication
	int row = 0;
	int col = 0;
	for(row = 0; row < N; row++) {
		for(col = 0; col < N; col++) {
			for(k = 0; k < N; k++) {
				C[row * N + col] += A[row * N + k] * B[k * N + col];
			}
		}
	}
	// Print Product Matrix
	printf("\nMatrix C:\n");
	print_matrix(N, C);
	// Write Product Matrix to file
	write_to_file(N, C);
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

int main(int argc, char *argv[]) {
	// Check the command line arguments
	check_command_line_args(argc, argv);
    
    // Get matrix
    int N = atoi(argv[1]);
    
    // Declare Matrices
    float *A;
    float *B;

    // Index
    int i = 0;

    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));

    // Timing program execution
	clock_t start;
	clock_t stop;

	start = clock();

    // Perform Multiplication
    matrix_mult(N, A, B);

    stop = clock();

    // Display time taken
	float time_taken = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("\nExecuted in: %lf seconds\n", time_taken);
	//printf("%lf\t%d\n", time_taken, N);

    return 0;
}