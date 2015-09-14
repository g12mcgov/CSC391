# DecodeEmail2FULP

Full assignment PDF can be found [here](https://github.com/g12mcgov/CSC391/blob/master/DecodeEmail2FULP/CSC%2B391%2BProject%2B1.pdf).

Background
=======

During class, we discussed how to write simple CUDA programs that allocate memory, copies data to the GPU, and execute a single instruction across multiple threads via a kernel call.

Directions
=======

Design and implement a short CUDA program that called `DecodeEmail2FULP.cu`.

The DecodeEmail2FULP program will use a command line argument to open a text file and decode its contents using a kernel call.

The decoding algorithm is as follows: each character must be decremented by the value of 1. That is all.

You must check for 1) the appropriate number of command line arguments and 2) whether the file exists. Appropriate error messages must be issued, followed by a graceful exit.

Running
=======

To run the CUDA code you must be `ssh`'d into the `greenflash` GPU cluster at WFU. So yeah, you have to be a WFU Computer Science student. This has the `NVCC` compiler installed and access to the GPUs.

Then run:

`$ make all`

And finally, 

`./DecodeEmail2FULP encoded09.dat`

You should then see the output of the decrypted message.
