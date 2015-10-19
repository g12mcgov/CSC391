/* 
* @Author: grantmcgovern
* @Date:   2015-10-19 14:45:45
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-10-19 15:45:24
*/

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

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

void monte_carlo(int num_points) {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
	int count = 0;

	// Seed random nums against sys clock
	srand48(time(NULL));

	// Loop through and compute randoms
	for(int i = 0; i < num_points; i++) {
		x = drand48();
		y = drand48();
		z = x*x + y*y;

		if(z <= 1) {
			count++;
		}
	}
	printf("Count: %d\n", count);
	printf("# Points: %d\n", num_points);
	double pi = (double)(count * 1.0 / num_points * 4.0);
	printf("Estimate of pi: %f\n", pi); 
}

int main(int argc, char *argv[]) {
	check_command_line_args(argc);
	monte_carlo(atoi(argv[1]));
    return 0;
}