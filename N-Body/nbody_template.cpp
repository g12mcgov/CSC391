/* 
* @Author: grantmcgovern
* @Date:   2015-10-28 12:52:26
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-11-01 16:31:11
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define N 9999     // number of bodies
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

float dt = 0.05; // time interval

float body[10000][7]; // data array of bodies

int main(int argc, char **argv) {

    int tmax = 0;
    float Fx_dir[N];
    float Fy_dir[N];
    float Fz_dir[N];

    if (argc != 2) {
        fprintf(stderr, "Format: %s { number of timesteps }\n", argv[0]);
        exit (-1);
    }

    tmax = atoi(argv[1]);

    srand48(time(NULL));

    // TODO: assign each body a random initial positions and velocities
    for (int i = 0; i < N; i++) {
        body[i][MASS] = 0.001;
        body[i][X_VEL] = drand48();
        body[i][Y_VEL] = drand48();
        body[i][Z_VEL] = drand48();

        body[i][X_POS] = drand48();
        body[i][Y_POS] = drand48();
        body[i][Z_POS] = drand48();  
    }

    // print out initial positions in PDB format
    printf("MODEL %8d\n", 0);
    for (int i = 0; i < N; i++) {
        printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
               "ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
    }
    printf("TER\nENDMDL\n");

    // CANNOT BE PARALLEL
    // step through each time step
    for (int t = 0; t < tmax; t++) {
        // force calculation

        // TODO: initialize forces to zero
        for (int i = 0; i < N; i++) {
            Fx_dir[i] = 0.0;
            Fy_dir[i] = 0.0;
            Fz_dir[i] = 0.0;
        }

        // PARALLEL
        for (int x = 0; x < N; x++) {  // force on body x due to
            for (int i = 0; i < N; i++) {   // all other bodies 
                // position differences in x-, y-, and z-directions
                float x_diff, y_diff, z_diff;

                if (i != x) {
                    // TODO: calculate position difference between body i and x in x-directions, y-directions, and z-directions
                    x_diff = body[i][X_POS] - body[x][X_POS];
                    y_diff = body[i][Y_POS] - body[x][Y_POS];
                    z_diff = body[i][Z_POS] - body[x][Z_POS];

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

                    // if sufficiently far away, gravitation force
                    if (r > 2.0) {
                        // TODO: compute gravitational force between body i and x
                        Fg = G * body[i][MASS] * body[x][MASS] / rr;

                        Fr = MU * drand48();
                        // Maybe Fr = MU * (drand48() - 0.5); (forces friction to be either positive or negative -- range -0.5, 0.5)

                        F = Fg + Fr;

                        // TODO: compute frictional force
                        Fx_dir[x] += F * x_diff / r;  // resolve forces in x and y directions
                        Fy_dir[x] += F * y_diff / r;  // and accumulate forces
                        Fz_dir[x] += F * z_diff / r;  // 
                    } 
                    else {
                        // if too close, weak anti-gravitational force
                        float F = G * 0.01 * 0.01 / r;
                        Fx_dir[x] -= F * x_diff / r;  // resolve forces in x and y directions
                        Fy_dir[x] -= F * y_diff / r;  // and accumulate forces
                        Fz_dir[x] -= F * z_diff / r;  // 
                    }
                }
            }
        }

        // update postions and velocity in array
        for (int i = 0; i < N; i++) {
            // TODO: update velocities
            body[i][X_VEL] += Fx_dir[i] * dt / body[i][MASS];
            body[i][Y_VEL] += Fy_dir[i] * dt / body[i][MASS];
            body[i][Z_VEL] += Fz_dir[i] * dt / body[i][MASS];
            
            // periodic boundary conditions
            if (body[i][X_VEL] <  -BOXL * 0.5) body[i][X_VEL] += BOXL;
            if (body[i][X_VEL] >=  BOXL * 0.5) body[i][X_VEL] -= BOXL;
            if (body[i][Y_VEL] <  -BOXL * 0.5) body[i][Y_VEL] += BOXL;
            if (body[i][Y_VEL] >=  BOXL * 0.5) body[i][Y_VEL] -= BOXL;
            if (body[i][Z_VEL] <  -BOXL * 0.5) body[i][Z_VEL] += BOXL;
            if (body[i][Z_VEL] >=  BOXL * 0.5) body[i][Z_VEL] -= BOXL;

            // TODO: update positions
            body[i][X_POS] += body[i][X_VEL] * dt;
            body[i][Y_POS] += body[i][Y_VEL] * dt;
            body[i][Z_POS] += body[i][Z_VEL] * dt;

            // periodic boundary conditions
            if (body[i][X_POS] <  -BOXL * 0.5) body[i][X_POS] += BOXL;
            if (body[i][X_POS] >=  BOXL * 0.5) body[i][X_POS] -= BOXL;
            if (body[i][Y_POS] <  -BOXL * 0.5) body[i][Y_POS] += BOXL;
            if (body[i][Y_POS] >=  BOXL * 0.5) body[i][Y_POS] -= BOXL;
            if (body[i][Z_POS] <  -BOXL * 0.5) body[i][Z_POS] += BOXL;
            if (body[i][Z_POS] >=  BOXL * 0.5) body[i][Z_POS] -= BOXL;
        }

        // print out positions in PDB format
        printf("MODEL %8d\n", t+1);
        for (int i = 0; i < N; i++) {
            printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
                       "ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
        }
        printf("TER\nENDMDL\n");
    }  // end of time period loop
}
