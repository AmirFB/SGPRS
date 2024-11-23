#include <iostream>
#include <cstdlib>
__global__ void k(unsigned long long dt){

  unsigned long long start = clock64();
  while (clock64() < (start+dt));
}

int main(int argc, char *argv[]){

  cudaStream_t h, l;
  int hp, lp;
  cudaDeviceGetStreamPriorityRange(&lp, &hp);
  std::cout << "lowest priority: " << lp << " highest priority: " << hp << std::endl;
  cudaStreamCreateWithPriority(&h, cudaStreamDefault, hp);
  cudaStreamCreateWithPriority(&l, cudaStreamDefault, lp);
  unsigned long long dt = 100000000ULL;
  int blocks = 26*5;
  if (argc > 1) dt *= atoi(argv[1]);
  if (argc > 2) blocks = 1;
  for (int i = 0; i < 5; i++) k<<<blocks, 1024,0,h>>>(dt);
  for (int i = 0; i < 5; i++) k<<<blocks, 1024, 0, l>>>(dt);
  cudaDeviceSynchronize();
}