all: main

main: main.cu 
	nvcc --std=c++11 -Werror cross-execution-space-call -lm main.cu -o lab2

clean:
	rm -rf lab2