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


  for (size_t i = 0; i < n_iterations; i++) {
    auto itr_timer = timer::start_clock();
    // C = α ∗ op(A) ∗ B + β ∗ C
    const double alpha = 1.0;
    const double beta = -1.0;
    cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, cooccurrence.n,
                   n_dimensions, cooccurrence.n, cooccurrence.nnz,
                   &alpha, descr, cuda_val, cuda_row_ptr,
                   cuda_col_ind, embedding, n_dimensions,
                   &beta, x, n_dimensions);
    timer::stop_clock("ITERATION",itr_timer);
  }

  cudaFree(cuda_val);
  cudaFree(cuda_row_ptr);
  cudaFree(cuda_col_ind);

  double *_x = (double *)malloc(sizeof(double) * cooccurrence.n * n_dimensions);
  cudaMemcpy(_x, x, cooccurrence.n * n_dimensions * sizeof(double),
             cudaMemcpyDeviceToHost);
  return std::unique_ptr<double>(_x);
}
} // en