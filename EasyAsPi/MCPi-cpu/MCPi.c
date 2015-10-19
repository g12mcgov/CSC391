/* 
* @Author: grantmcgovern
* @Date:   2015-10-19 14:45:45
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-10-19 18:34:00
*/

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

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
		printf("Invalid Number of Arguments");
		exit(1);
	}
}

void add_frequency(struct Frequencies *frequencies, float value) {
	/*
	 * Convert to int since we can't do switch 
	 * statements with floats/doubles in C.
	 */
	printf("Value: %d, ", (int)(value * 10.0));
	switch((int)(value * 10.0)) {
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

float compute_frequency(int frequency, float num_points) {
	return (frequency / num_points);
}

/**
 * print_frequency(struct Frequencies *, int)
 * @param frequencies 
 * @param num_points
 */
void print_frequency(struct Frequencies *frequencies, int num_points) {
	printf("\nFrequencies:\n");
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
	for(int i = 0; i < num_points; i++) {
		x = drand48();
		y = drand48();
		
		// X
		printf("X: %f, ", x);
		float rounded_x = floorf(x * 10.0) / 10.0;
		add_frequency(&frequencies, rounded_x);

		printf("\n");

		// Y
		printf("Y: %f, ", y);
		float rounded_y = floorf(y * 10.0) / 10.0;
		add_frequency(&frequencies, rounded_y);

		printf("\n\n");

		z = x*x + y*y;

		if(z <= 1) {
			count++;
		}
	}
	printf("Count: %d\n", count);
	printf("# Points: %d\n", num_points);
	double pi = (double)(count * 1.0 / num_points * 4.0);
	
	printf("Estimate of pi: %f\n", pi); 

	// Print Frequencies
	print_frequency(&frequencies, num_points);
}

int main(int argc, char *argv[]) {
	check_command_line_args(argc);
	monte_carlo(atoi(argv[1]));
    return 0;
}