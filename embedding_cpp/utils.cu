#include "utils.h"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace utils {

void sen_test(){

  const size_t alloc_size = 300000000;
  double* big_alloc = (double*)malloc(alloc_size*sizeof(double));


  const auto sen_time = timer::start_clock();
  double* cuda_big_alloc;
  cudaMalloc(&cuda_big_alloc, alloc_size);

  cudaMemcpy(cuda_big_alloc, big_alloc, alloc_size,
             cudaMemcpyHostToDevice);
  timer::stop_clock("SEN TIME", sen_time);
}


}
