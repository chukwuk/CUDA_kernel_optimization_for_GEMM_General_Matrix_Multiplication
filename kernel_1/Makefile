CC=nvcc -ftz=true -use_fast_math 

all: main

main: main.cu 
	$(CC) matrixmul.o main.cu -o main

clean:
	rm -f main matrixmul.o
