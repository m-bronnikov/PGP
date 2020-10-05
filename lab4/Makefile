CC = nvcc
CFLAGS =  -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -Werror cross-execution-space-call -lm 
SOURSES = main.cu
BIN = lab4
all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)
