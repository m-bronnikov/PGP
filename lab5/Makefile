NCC = nvcc
GCC = g++ 
CFLAGS = -std=c++11 -Werror 
EXEC = cross-execution-space-call
NSOURSE = cuda_sort.cu
GSOURCE = testgen.cpp
CSOURCE = check_input.cpp
BIN = run
GEN = generator
CHECK = checker

all:
	$(NCC) $(CFLAGS) $(EXEC) -lm -o $(BIN) $(NSOURSE)
	$(GCC) $(CFLAGS) -o $(GEN) $(GSOURCE)
	$(GCC) $(CFLAGS) -o $(CHECK) $(CSOURCE)
