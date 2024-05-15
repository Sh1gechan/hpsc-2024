#include <cstdio>
#include <cstdlib>

__global__ void initializeBucket(int *bucket, int value) {
  bucket[threadIdx.x] = value;
}

__global__ void countKeysInBuckets(int *keys, int *bucket) {
  atomicAdd(&bucket[keys[threadIdx.x]], 1);
}

__global__ void scanBuckets(int *bucket, int *offset, int range) {
  int i = threadIdx.x;
  offset[i+1] = bucket[i];
  for(int j=1; j<range; j<<=1) {
    __syncthreads();
    int temp = (i >= j) ? offset[i] + offset[i-j] : offset[i];
    __syncthreads();
    offset[i] = temp;
  }
}

__global__ void sortKeysWithBuckets(int *keys, int *bucket, int *offset) {
  int i = threadIdx.x;
  int j = blockIdx.x;
  if (i < bucket[j]) keys[i + offset[j]] = j;
}

int main() {
  int n = 50, range = 5;
  int *keys, *bucket, *offset;

  cudaMallocManaged(&keys, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&offset, (range + 1) * sizeof(int));

  for (int i = 0; i < n; i++) {
    keys[i] = rand() % range;
    printf("%d ", keys[i]);
  }
  printf("\n");

  initializeBucket<<<1, range>>>(bucket, 0);
  cudaDeviceSynchronize();
  
  countKeysInBuckets<<<1, n>>>(keys, bucket);
  cudaDeviceSynchronize();
  
  scanBuckets<<<1, range>>>(bucket, offset, range);
  cudaDeviceSynchronize();
  
  sortKeysWithBuckets<<<range, n>>>(keys, bucket, offset);
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {
    printf("%d ", keys[i]);
  }
  printf("\n");

  cudaFree(keys);
  cudaFree(bucket);
  cudaFree(offset);

  return 0;
}

