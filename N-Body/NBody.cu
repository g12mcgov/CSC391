/* 
* @Author: grantmcgovern
* @Date:   2015-10-28 12:52:26
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-11-04 16:31:11
*/

#include <string.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>


#define N 9999     // number of bodies
#define WIDTH 7
#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity
#define G 10       // "gravitational constant" (not really)
#define MU 0.001   // "frictional coefficient" 
#define BOXL 100.0 // periodic boundary box length
#define dt 0.05
#define THREADS 768

/**
 * [cuda_memory_check description]
 *
 *  Useful when debugging memory issues.
 */
void cuda_memory_check() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA -error %s (%d)\n", cudaGetErrorString(error), error);
    }
}

/**
 * [check_command_line_args description]
 * @param argc [description]
 * @param argv [description]
 */
void check_command_line_args(int argc, char *argv[] ){
    if (argc != 2){
        fprintf(stderr, "Format: %s { number of timesteps }\n", argv[0]);
        exit(1);
    }
    // Check command line args are positive
    int i = 1;
    for( ; i < argc; i++) {
        // We have a negative argument
        if(atoi(&argv[i][0]) < 0) {
            printf("Invalid Argument: Negative Number given\n");
            exit(1);
        }
    }
}

/**
 * [n_body description]
 * @param dev_body         [description]
 * @param states           [description]
 * @param time_random_seed [description]
 */
__global__ void n_body(float *dev_body, curandState_t *states, int time_random_seed) {
    // Get the global thread Id
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    // Force arrays
    float Fx_dir = 0.0;
    float Fy_dir = 0.0;
    float Fz_dir = 0.0;

    // Differences
    float x_diff = 0.0;
    float y_diff = 0.0;
    float z_diff = 0.0;

    // Initialize curand()
    curand_init(time_random_seed, globalId, 0, &states[globalId]);

    if(globalId <= N) {
        // all other bodies
        int i = 0;
        for(i = 0; i < N; i++) {
            if(i != globalId) {

                // calculate position difference between body i and x in x-directions, y-directions, and z-directions
                x_diff = dev_body[i * WIDTH + X_POS] - dev_body[globalId * WIDTH + X_POS];
                y_diff = dev_body[i * WIDTH + Y_POS] - dev_body[globalId * WIDTH + Y_POS];
                z_diff = dev_body[i * WIDTH + Z_POS] - dev_body[globalId * WIDTH + Z_POS];
            
                 // periodic boundary conditions
                if (x_diff <  -BOXL * 0.5) x_diff += BOXL;
                if (x_diff >=  BOXL * 0.5) x_diff -= BOXL;
                if (y_diff <  -BOXL * 0.5) y_diff += BOXL;
                if (y_diff >=  BOXL * 0.5) y_diff -= BOXL;
                if (z_diff <  -BOXL * 0.5) z_diff += BOXL;
                if (z_diff >=  BOXL * 0.5) z_diff -= BOXL;
            
                // calculate distance (r)
                float rr = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
                float r = sqrt(rr);

                // force between bodies i and x
                float F = 0.0;
                float Fg = 0.0;
                float Fr = 0.0;

                if (r > 2.0) {
                    // Compute gravitational force between body i and x
                    Fg = G * dev_body[i * WIDTH + MASS] * dev_body[globalId * WIDTH + MASS] / rr;

                    Fr = MU * (curand_uniform(&states[globalId]) - 0.5);
                    // Maybe Fr = MU * (drand48() - 0.5); (forces friction to be either positive or negative -- range -0.5, 0.5)

                    F = Fg + Fr;

                    // Compute frictional force
                    Fx_dir += F * x_diff / r;  // resolve forces in x and y directions
                    Fy_dir += F * y_diff / r;  // and accumulate forces
                    Fz_dir += F * z_diff / r;  // 
                } 
                else {
                    // if too close, weak anti-gravitational force
                    F = G * 0.01 * 0.01 / rr;
                    
                    Fx_dir -= F * x_diff / r;  // resolve forces in x and y directions
                    Fy_dir -= F * y_diff / r;  // and accumulate forces
                    Fz_dir -= F * z_diff / r;  // 
                }
            }
        }
    }
    // update velocities
    dev_body[globalId * WIDTH + X_VEL] += Fx_dir * dt / dev_body[globalId * WIDTH + MASS]; 
    dev_body[globalId * WIDTH + Y_VEL] += Fy_dir * dt / dev_body[globalId * WIDTH + MASS];
    dev_body[globalId * WIDTH + Z_VEL] += Fz_dir * dt / dev_body[globalId * WIDTH + MASS];

    if (dev_body[globalId * WIDTH + X_VEL] <  -BOXL * 0.5) dev_body[globalId * WIDTH + X_VEL] += BOXL;
    if (dev_body[globalId * WIDTH + X_VEL] >=  BOXL * 0.5) dev_body[globalId * WIDTH + X_VEL] -= BOXL;
    if (dev_body[globalId * WIDTH + Y_VEL] <  -BOXL * 0.5) dev_body[globalId * WIDTH + Y_VEL] += BOXL;
    if (dev_body[globalId * WIDTH + Y_VEL] >=  BOXL * 0.5) dev_body[globalId * WIDTH + Y_VEL] -= BOXL;
    if (dev_body[globalId * WIDTH + Z_VEL] <  -BOXL * 0.5) dev_body[globalId * WIDTH + Z_VEL] += BOXL;
    if (dev_body[globalId * WIDTH + Z_VEL] >=  BOXL * 0.5) dev_body[globalId * WIDTH + Z_VEL] -= BOXL;
    
    // update positions
    dev_body[globalId * WIDTH + X_POS] += dev_body[globalId * WIDTH + X_VEL] * dt;
    dev_body[globalId * WIDTH + Y_POS] += dev_body[globalId * WIDTH + Y_VEL] * dt;
    dev_body[globalId * WIDTH + Z_POS] += dev_body[globalId * WIDTH + Z_VEL] * dt;
 
    // periodic boundary conditions
    if (dev_body[globalId * WIDTH + X_VEL] <  -BOXL * 0.5) dev_body[globalId * WIDTH + X_VEL] += BOXL;
    if (dev_body[globalId * WIDTH + X_VEL] >=  BOXL * 0.5) dev_body[globalId * WIDTH + X_VEL] -= BOXL;
    if (dev_body[globalId * WIDTH + Y_VEL] <  -BOXL * 0.5) dev_body[globalId * WIDTH + Y_VEL] += BOXL;
    if (dev_body[globalId * WIDTH + Y_VEL] >=  BOXL * 0.5) dev_body[globalId * WIDTH + Y_VEL] -= BOXL;
    if (dev_body[globalId * WIDTH + Z_VEL] <  -BOXL * 0.5) dev_body[globalId * WIDTH + Z_VEL] += BOXL;
    if (dev_body[globalId * WIDTH + Z_VEL] >=  BOXL * 0.5) dev_body[globalId * WIDTH + Z_VEL] -= BOXL;
}

// void initialize_body_array(float *body) {
//     int i = 0;
//     for( ; i < N; i++) {
//         body[i][MASS] = 0.001;
//         body[i][X_VEL] = drand48();
//         body[i][Y_VEL] = drand48();
//         body[i][Z_VEL] = drand48();

//         body[i][X_POS] = drand48();
//         body[i][Y_POS] = drand48();
//         body[i][Z_POS] = drand48();  
//     }
// }

/**
 * MAIN
 * @param  argc [description]
 * @param  argv [description]
 * @return      [description]
 */
int main(int argc, char **argv) {
    // Check if arguments are good
    check_command_line_args(argc, argv);

    // Total number of times
    int tmax = atoi(argv[1]);

    // Size
    int size = N * 7 * sizeof(float);

    // Seed time against system clock
    srand48(time(NULL));

 
    
    // Body array 
    float *body = (float *)malloc(10000 * 7 * sizeof(float));
    float *dev_body;

    // Allocate memory on GPU
    cudaMalloc((void**) &dev_body, 10000 * 7 * sizeof(float));

    // Initialize body array
    int i = 0;
    for(i = 0; i < N; i++) {
        body[i*WIDTH+MASS] = 0.001;
        body[i*WIDTH+X_VEL] = drand48();
        body[i*WIDTH+Y_VEL] = drand48();
        body[i*WIDTH+Z_VEL] = drand48();

        body[i*WIDTH+X_POS] = drand48();
        body[i*WIDTH+Y_POS] = drand48();
        body[i*WIDTH+Z_POS] = drand48();  
    }

    printf("MODEL %8d\n", 0);
    for (i = 0; i < N; i++) {
        printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
            "ATOM", i + 1, "CA ", "GLY", "A", i + 1, body[i*WIDTH+X_POS], body[i*WIDTH+Y_POS], body[i*WIDTH+Z_POS], 1.00, 0.00);
    }
    printf("TER\nENDMDL\n");

    for (int t = 0; t < tmax; t++) {
        // Device memory
        curandState_t *states;
        cudaMalloc((void**) &states, 9999 * sizeof(curandState_t));

        // Copy variables to kernel
        cudaMemcpy(dev_body, body, size, cudaMemcpyHostToDevice);
        
        // Call kernel
        n_body<<< (int)ceil(N / THREADS) + 1, THREADS >>>(dev_body, states, time(NULL));

        // Syncthreads
        cudaThreadSynchronize();

        // Copy memory back
        cudaMemcpy(body, dev_body, size, cudaMemcpyDeviceToHost);

        // positions in PDB format
        printf("MODEL %8d\n", t + 1);
        for (i = 0; i < N; i++) {
            printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
                "ATOM", i + 1, "CA ", "GLY", "A", i + 1, body[i*WIDTH+X_POS], body[i*WIDTH+Y_POS], body[i*WIDTH+Z_POS], 1.00, 0.00);
        }
        printf("TER\nENDMDL\n");
    }
    return 0;
}