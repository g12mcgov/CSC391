/* 
* @Author: grantmcgovern
* @Date:   2015-10-19 14:45:45
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-10-21 00:55:20
*/

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * For printing certain statements.
 */
const unsigned int DEBUG = 0;

/**
 * Frequencies
 *
 * Holds the values for frequencies in the following
 * format:
 *
 * _0x = 0.0
 * _1x = 1.0
 * _2x = 2.0
 * 
 * etc..
 */
struct Frequencies {
	unsigned int _0x;
	unsigned int _1x;
	unsigned int _2x;
	unsigned int _3x;
	unsigned int _4x;
	unsigned int _5x;
	unsigned int _6x;
	unsigned int _7x;
	unsigned int _8x;
	unsigned int _9x;
};

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
 * add_frequency(struct Frequencies *, float)
 * @param frequencies 
 * @param value
 *
 * Increments the frequency for a specific value. Uses
 * pass-by-reference to keep adding to the same frequency
 * counts.
 */
void add_frequency(struct Frequencies *frequencies, int value) {
	/*
	 * Convert to int since we can't do switch 
	 * statements with floats/doubles in C.
	 *
	 * i.e:
	 *
	 * 0.1 -> 1
	 * 0.2 -> 2
	 * 0.3 -> 3
	 *
	 * etc...
	 */
	switch(value) {
		case 0:
			frequencies->_0x++;
			break;
		case 1:
			frequencies->_1x++;
			break;
		case 2:
			frequencies->_2x++;
			break;
		case 3:
			frequencies->_3x++;
			break;
		case 4: 
			frequencies->_4x++;
			break;
		case 5:
			frequencies->_5x++;
			break;
		case 6:
			frequencies->_6x++;
			break;
		case 7:
			frequencies->_7x++;
			break;
		case 8:
			frequencies->_8x++;
			break;
		case 9:
			frequencies->_9x++;
			break;
	}
}

/**
 * compute_frequency(int, float)
 * @param  frequency 
 * @param  num_points
 * @return (float) frequency value
 *
 * Computes frequency for a decimal place value (0.1, 0.2, etc...)
 * by dividing it by total # of random points.
 */
float compute_frequency(int frequency, float num_points) {
	return (frequency / num_points);
}

/**
 * print_frequency(struct Frequencies *, int)
 * @param frequencies 
 * @param num_points
 *
 * Prints out the count and frequencies (computed)
 */
void print_frequency(struct Frequencies *frequencies, int num_points) {
	printf("\nFrequencies:\n");
	if(DEBUG) {
		printf("0.0x: Count: %d, Frequency: %f\n", 
			frequencies->_0x, compute_frequency(frequencies->_0x, num_points));
		printf("0.1x: Count: %d, Frequency: %f\n", 
			frequencies->_1x, compute_frequency(frequencies->_1x, num_points));
		printf("0.2x: Count: %d, Frequency: %f\n", 
			frequencies->_2x, compute_frequency(frequencies->_2x, num_points));
		printf("0.3x: Count: %d, Frequency: %f\n", 
			frequencies->_3x, compute_frequency(frequencies->_3x, num_points));
		printf("0.4x: Count: %d, Frequency: %f\n", 
			frequencies->_4x, compute_frequency(frequencies->_4x, num_points));
		printf("0.5x: Count: %d, Frequency: %f\n", 
			frequencies->_5x, compute_frequency(frequencies->_5x, num_points));
		printf("0.6x: Count: %d, Frequency: %f\n", 
			frequencies->_6x, compute_frequency(frequencies->_6x, num_points));
		printf("0.7x: Count: %d, Frequency: %f\n", 
			frequencies->_7x, compute_frequency(frequencies->_7x, num_points));
		printf("0.8x: Count: %d, Frequency: %f\n", 
			frequencies->_8x, compute_frequency(frequencies->_8x, num_points));
		printf("0.9x: Count: %d, Frequency: %f\n", 
			frequencies->_9x, compute_frequency(frequencies->_9x, num_points));
	}
	else {
		printf("%d\n", frequencies->_0x);
		printf("%d\n", frequencies->_1x);
		printf("%d\n", frequencies->_2x);
		printf("%d\n", frequencies->_3x);
		printf("%d\n", frequencies->_4x);
		printf("%d\n", frequencies->_5x);
		printf("%d\n", frequencies->_6x);
		printf("%d\n", frequencies->_7x);
		printf("%d\n", frequencies->_8x);
		printf("%d\n", frequencies->_9x);
	}
}

/**
 * write_to_file(struct Frequencies *)
 * @param frequencies [description]
 */
void write_to_file(struct Frequencies *frequencies) {
	// Open file for writing
	FILE *file = fopen("./freq.dat", "w+");
	// As long as we can write to the file
	if(file != NULL) {
		fprintf(file, "0\t%d\n", frequencies->_0x);
		fprintf(file, "1\t%d\n", frequencies->_1x);
		fprintf(file, "2\t%d\n", frequencies->_2x);
		fprintf(file, "3\t%d\n", frequencies->_3x);
		fprintf(file, "4\t%d\n", frequencies->_4x);
		fprintf(file, "5\t%d\n", frequencies->_5x);
		fprintf(file, "6\t%d\n", frequencies->_6x);
		fprintf(file, "7\t%d\n", frequencies->_7x);
		fprintf(file, "8\t%d\n", frequencies->_8x);
		fprintf(file, "9\t%d\n", frequencies->_9x);
	}
	else {
		printf("Unable to open file.\n");
		exit(1);
	}
	fclose(file);
}

/**
 * monte_carlo(int)
 * @param num_points
 *
 * Estimates the value of Pi via the Monte Carlo algorithm.
 */
void monte_carlo(int num_points) {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
	int count = 0;

	// Seed random nums against sys clock
	srand48(time(NULL));

	/**
	 * Vars to store frequency. Ideally, we'd 
	 * use a hash table to do this, but program
	 * is small and is unecessary for this situation.
	 */
	struct Frequencies frequencies;

	/**
	 * Initialize all vals to 0 to prevent
	 * weird/unexpected values.
	 */
	frequencies._0x = 0;
	frequencies._1x = 0;
	frequencies._2x = 0;
	frequencies._3x = 0;
	frequencies._4x = 0;
	frequencies._5x = 0;
	frequencies._6x = 0;
	frequencies._7x = 0;
	frequencies._8x = 0;
	frequencies._9x = 0;

	// Loop through and compute randoms
	int i = 0;
	for(; i < num_points; i++) {
		x = drand48();
		y = drand48();
		
		// X
		if(DEBUG) {
			printf("X: %f, ", x);
		}

		float rounded_x = (int)(floor(x * 10.0));
		add_frequency(&frequencies, rounded_x);

		if(DEBUG) {
			printf("\n");
		}

		// Y
		if(DEBUG) {
			printf("Y: %f, ", y);
		}

		float rounded_y = (int)(floor(y * 10.0));
		add_frequency(&frequencies, rounded_y);

		if(DEBUG) {
			printf("\n\n");
		}

		z = x*x + y*y;

		if(z <= 1) {
			count++;
		}
	}
	// Print program params
	printf("Count: %d\n", count);
	printf("# Points: %d\n", num_points);
	
	// Compute estimated pi
	double pi = (double)(4.0 * (double)count / (double)num_points);
	
	// Print Pi value
	printf("Estimate of pi: %f\n", pi); 

	// Print Frequencies
	print_frequency(&frequencies, num_points);
	write_to_file(&frequencies);
}

/**
 * MAIN
 */
int main(int argc, char *argv[]) {
	check_command_line_args(argc);
	monte_carlo(atoi(argv[1]));
    return 0;
}