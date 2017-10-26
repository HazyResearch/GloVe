#include "cusparse.h"
#include "solver.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "timer.h"

namespace solver {
std::unique_ptr<double> power_iteration(const CSR<double> &cooccurrence,
                                        const size_t n_iterations,
                                        const size_t n_dimensions) {
  auto cuda_init = timer::start_clock();
  cusparseHandle_t handle = 0;
  cusparseStatus_t status;
  cusparseMatDescr_t descr = 0;
  /* initialize cusparselibrary */
  status = cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    // TODO Throw error: CLEANUP("CUSPARSE Library initialization failed");
    return std::unique_ptr<double>();
  }
  /* create and setup matrix descriptor */
  status = cusparseCreateMatDescr(&descr);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    // TODO Throw error: CLEANUP("Matrix descriptor initialization failed");
    return std::unique_ptr<double>();
  }
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  double *cuda_val;
  int *cuda_row_ptr;
  int *cuda_col_ind;
  cudaMalloc(&cuda_val, cooccurrence.nnz * sizeof(double));
  cudaMalloc(&cuda_row_ptr, (cooccurrence.n + 1) * sizeof(int));
  cudaMalloc(&cuda_col_ind, cooccurrence.nnz * sizeof(int));

  cudaMemcpy(cuda_val, cooccurrence.val, cooccurrence.nnz * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_row_ptr, cooccurrence.rowPtr, (cooccurrence.n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_col_ind, cooccurrence.colInd, cooccurrence.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  double *embedding;
  cudaMalloc(&embedding, cooccurrence.n * n_dimensions * sizeof(double));
  double *x;
  cudaMalloc(&x, cooccurrence.n * n_dimensions * sizeof(double));

  timer::stop_clock("CUDA INIT",cuda_init);

  std::cout << cooccurrence.n << " " << cooccurrence.nnz << std::endl;
  std::cout << n_iterations << " " << n_dimensions << std::endl;

  for (size_t i = 0; i < n_iterations; i++) {
    const auto itr_timer = timer::start_clock();
    // C = α ∗ op(A) ∗ B + β ∗ C
    const double alpha = 1.0;
    const double beta = -1.0;
    status = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, cooccurrence.n,
                   n_dimensions, cooccurrence.n, cooccurrence.nnz,
                   &alpha, descr, cuda_val, cuda_row_ptr,
                   cuda_col_ind, embedding, cooccurrence.n,
                   &beta, x, cooccurrence.n);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      std::cout << "ERROR: Matrix-matrix multiplication failed" << std::endl;
      std::cout << "status = " << status << std::endl;
      return std::unique_ptr<double>();
    }  
    cudaDeviceSynchronize();
    timer::stop_clock("ITERATION " + std::to_string(i), itr_timer);
  }

  const auto cuda_free = timer::start_clock();
  cudaFree(cuda_val);
  cudaFree(cuda_row_ptr);
  cudaFree(cuda_col_ind);
  timer::stop_clock("CUDA FREE",cuda_free);

  const auto cpu_xfr = timer::start_clock();
  double *_x = (double *)malloc(sizeof(double) * cooccurrence.n * n_dimensions);
  cudaMemcpy(_x, x, cooccurrence.n * n_dimensions * sizeof(double),
             cudaMemcpyDeviceToHost);
  timer::stop_clock("CPU XFR", cpu_xfr);
  return std::unique_ptr<double>(_x);
}
} // en