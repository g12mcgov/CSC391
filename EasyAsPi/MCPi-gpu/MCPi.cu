/* 
* @Author: grantmcgovern
* @Date:   2015-10-19 14:45:45
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-10-19 21:58:04
*/

#include <stdio.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>

#define THREADS 1024

/**
 * For printing certain statements.
 */
const unsigned int DEBUG = 0;

/*
* check_command_line_args(int)
*
* Checks to see whether used used proper
* number of command line arguments.
*/
void check_command_line_args(int argc) {
	// Ensure command line args are limited to only 1
	if(argc > 2 || argc == 1) {
		printf("Invalid Number of Arguments\n");
		exit(1);
	}
}

/**
 * init()
 * @param seed   [description]
 * @param states [description]
 */
__global__ void init(unsigned int seed, curandState_t* states) {
	curand_init(seed, threadIdx.x, 0, &states[threadIdx.x]);
}

/**
 * randoms()
 * @param states  [description]
 * @param numbers [description]
 */
__global__ void randoms(curandState_t *states, int *num_points, \
						int *x_array, int *y_array, int *counts) {
	// Only care about X dimension
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalId > *num_points) {
		return; 
	}
	
	double x = curand_uniform(&states[globalId]);
	double y = curand_uniform(&states[globalId]);
	double z = 0.0;

	if(DEBUG) {
		printf("X: %f, ", x);
		printf("Y: %f, ", y);
		printf("\n\n");
	}

	float rounded_x = (int)(floor(x * 10.0));
	float rounded_y = (int)(floor(y * 10.0));

	// Add to the array containing rounded values
	x_array[globalId] = rounded_x;
	y_array[globalId] = rounded_y;

	z = x*x + y*y;
	
	// Mark it as either true or false for ease in later computation
	counts[globalId] = (z <= 1.0) ? 1 : 0;
}

void print_histogram(int num_points, int *x_array, int *y_array) {
	printf("\nFrequencies:\n");
	/**
	 * Histogram array
	 *
	 * Each value corresponds to the matching index.
	 * (i.e.):
	 *
	 * 0.0x = [0]
	 * 0.1x = [1]
	 * 0.2x = [2]
	 * 
	 * etc...
	 */
	int histogram_array[10] = {0};

	// Open file for writing
	FILE *file = fopen("./freq.dat", "w+");
	// As long as we can write to the file
	if(file != NULL) {
		int i = 0;
		for(i = 0; i < num_points; i++) {
			int x_value = x_array[i];
			int y_value = y_array[i];
			// Increment histogram counts
			histogram_array[x_value]++;
			histogram_array[y_value]++;
		}
		// Loop through and print histogram
		for(i = 0; i < 10; i++) {
			printf("0.%dx: Frequency: %d\n", i, histogram_array[i]);	
			fprintf(file, "0.%dx: Frequency: %d\n", i, histogram_array[i]);
		}
		exit(0);
	}
	else {
		printf("Unable to open file.\n");
		exit(1);
	}
	fclose(file);
}

/**
 * print_pi_estimation(int , int *)
 * @param num_points
 * @param counts
 */
void print_pi_estimation(int num_points, int *counts) {
	int i = 0;
	int sum = 0;
	for(i = 0; i < num_points; i++) {
		sum = sum + counts[i];
	}
	// Compute estimated pi
	double pi = (double)(4.0 * ((double)sum / (double)num_points));
	// Print Pi value
	printf("Estimate of pi: %lf\n", pi);
}

/**
 * MAIN
 * @param  argc
 * @param  argv
 * @return
 */
int main(int argc, char *argv[]) {
	// Check command line arguments
	check_command_line_args(argc);

	// If we make it to here we know command line args are ok
	int num_points = atoi(argv[1]);

	// keep track of seed value for every thread
	curandState_t *dev_states;

	// Allocate memory on the device
	cudaMalloc((void**) &dev_states, num_points * sizeof(curandState_t));

	// Compute block size
	int block_size = (int)(ceil(num_points / THREADS) + 1);

	// initialize all of the random states on the GPU
	init<<< block_size, THREADS >>>(time(NULL), dev_states);
	
	// Stores the counts for each thread
	int *counts = (int *)malloc(num_points * sizeof(int));

	// Device variables
	int *dev_counts = NULL;
	int *dev_num_points = NULL;
	int *dev_x_array = NULL;
	int *dev_y_array = NULL;

	// Store an array for x values
	int *x_array = (int *)malloc(num_points * sizeof(int));
	// Store an array for y values
	int *y_array = (int *)malloc(num_points * sizeof(int));

	// Allocate on device
	cudaMalloc((void**) &dev_counts, num_points * sizeof(int));
	cudaMalloc((void**) &dev_num_points, sizeof(int));
	cudaMalloc((void**) &dev_x_array, num_points * sizeof(int));
	cudaMalloc((void**) &dev_y_array, num_points * sizeof(int));

	// Copy to device the # of random numbers to generate
	cudaMemcpy(dev_num_points, &num_points, sizeof(int), cudaMemcpyHostToDevice);

	// Generate randoms
	randoms<<< block_size, THREADS >>>(
		dev_states, 
		dev_num_points, 
		dev_x_array,
		dev_y_array, 
		dev_counts
		);

	// Sync threads
	cudaThreadSynchronize();

	// Copy back to device
	cudaMemcpy(counts, dev_counts, num_points * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x_array, dev_x_array, num_points * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y_array, dev_y_array, num_points * sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory on device
	cudaFree(dev_states);
	cudaFree(dev_counts);
	cudaFree(dev_num_points);
	cudaFree(dev_x_array);
	cudaFree(dev_y_array);

	print_pi_estimation(num_points, counts);
	print_histogram(num_points, x_array, y_array);
}