# Parallelized Matrix Multiplication

![preview_1](http://i.imgur.com/k30gCYE.png)
![preview_2](http://i.imgur.com/MuNjcF2.png)

Full assignment PDF can be found [here](https://www.dropbox.com/s/xbuqdqo3w49hybd/CSC%2B391%2BProject%2B2.pdf?dl=0).

Background
=======

During class, we discussed how to implement a matrix multiplication subroutine. We also discussed several metrics for measuring parallel algorithm performance.


Directions
=======

Design and implement C and CUDA programs that called MatrixMult.c and MatrixMult.cu, respectively.

The MatrixMult programs will use a command line argument that corresponds to the row/column length to multiply two matrices with random single precision numbers (i.e., floats). They will then output 1) the entire execution time in seconds from the beginning to the end of the function/kernel call only, and 2) the product of the two matrices with each element printed out to a file called “product.dat” in a tab- delimited, row/column format. The C program must implement a serial code via a function call, and the CUDA program must implement a parallel code via a kernel call. It is completely up to you to implement the serial and parallel algorithms, including any parameters associated the implementation except that the parallel algorithm must use tiles.

You must check for 1) the appropriate number of command line arguments and 2) whether the command line argument corresponds to a positive number (you may round floating point inputs). Appropriate error messages must be issued, followed by a graceful exit.

To analyze the relative performances of the CPU vs. GPU implementations of your matrix multiplication algorithms, you will compute the time it takes to complete a matrix multiplication function/kernel call. For the GPU implementation, you will determine the optimal tile size that you will use for your analysis. You will then construct a graph of the speedup vs. matrix size (i.e., row/column length) for several tile sizes to see how their performances vary with system size and tile size. It is up to you to choose appropriate values for the matrix size to construct your graph, but a matrix size of 2,718 will be used for testing during grading.

Hint: In the execution of your GPU implementation, for accurate timing information, you must call the kernel twice because the GPU initialization takes time that is unrelated to the algorithm.

Running
=======

To run the CUDA code you must be `ssh`'d into the `greenflash` GPU cluster at WFU. So yeah, you have to be a WFU Computer Science student. This has the `NVCC` compiler installed and access to the GPUs.

Then run:

`$ make`

And finally, 

`$ ./MatrixMult 100`

For a NxN matrix of size **100**.

You should then see the output of the random matrix multiplication. It will also be written out to `product.dat`.

If you want to benchmark CPU vs. GPU performance, run the following bash script:

`$ bash graph.bash`

You'll then get execution time for different Matrix sizes AND Tile sizes (if running the CUDA program on a GPU).

Results
=======

Full lab data can be found [here](https://www.dropbox.com/s/s9t5anjtr61d1jo/results.xlsx?dl=0).
