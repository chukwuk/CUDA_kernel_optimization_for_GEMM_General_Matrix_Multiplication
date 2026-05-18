CC=nvcc --maxrregcount=210 -ftz=true -use_fast_math 

all: main

main: main.cu 
	$(CC) main.cu -o main

clean:
	rm -f main
