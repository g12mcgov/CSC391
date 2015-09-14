CC=nvcc

all: DecodeEmail2FULP
DecodeEmail2FULP: DecodeEmail2FULP.o
DecodeEmail2FULP.o: DecodeEmail2FULP.cu

clean:
	rm -f DecodeEmail2FULP DecodeEmail2FULP.o
run: DecodeEmail2FULP
	./DecodeEmail2FULP