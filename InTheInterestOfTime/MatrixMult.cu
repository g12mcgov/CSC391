#define N 2
#define TILE_WIDTH 4

#include <stdio.h>

__global__ void matrixMult (int *a, int *b, int *c, int width);

int main() {
 int a[N][N] = {{2, 2}, {2, 2}};
 int b[N][N] = {{4, 3}, {2, 5}};
 int c[N][N] = {{0, 0}, {0, 0}};
 
 int *dev_a, *dev_b, *dev_c;
 int size = N * N * sizeof(int);
 
 // initialize a and b matrices here
 cudaMalloc((void **) &dev_a, size);
 cudaMalloc((void **) &dev_b, size);
 cudaMalloc((void **) &dev_c, size);
 
 cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
 
 dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
 dim3 dimGrid((int)ceil(N/dimBlock.x), (int)ceil(N/dimBlock.y));
 
 matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
 
 cudaThreadSynchronize();

 cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

 //printf(c);
 for(int i = 0; i < N; i++) {
 	for(int j = 0; j < N; j++) {
 		printf("%d", c[i][j]);
 		printf("\t");
 	}
 	printf("\n");
 }

 cudaFree(dev_a);
 cudaFree(dev_b);
 cudaFree(dev_c);

}
__global__ void matrixMult(int* A, int* B, int* C, int width) {
 	int k, sum = 0;
 
	int col = blockIdx.x*TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y*TILE_WIDTH + threadIdx.y;

	if(col < width && row < width) {
		for (int k = 0; k < width; k++) {
			sum += A[row * width + k] * B[k * width + col];
			C[row * width + col] = sum;
		}
	}
}