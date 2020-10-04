#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int i) {
	int jdx = blockIdx.x;			// Абсолютный номер потока
    int idx = threadIdx.x;					// Общее кол-во потоков
    if(i == 0){
        for(int j = 0; j < 10000000; ++j){
            continue;
        }
    }
    printf("[%d, %d] = %d\n", idx, jdx, i);
}

int main() {

	for(int i = 0; i < 3; i++){
        printf("start itter: %d \n", i);
        kernel<<<2, 2>>>(i);
    }
    cudaThreadSynchronize();
        
	printf("\n");
	return 0;
}