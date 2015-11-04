GCC=nvcc

all: NBody

NBody:      NBody.cu
	$(GCC) -o NBody NBody.cu -lm
clean:
	$(RM)  NBody NBody.o