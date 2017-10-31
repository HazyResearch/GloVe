#include <stdio.h>
#include "sparse.h"
#include "mkl.h"

namespace utils {
  void preprocess_cpu(CSR<double>& csr_cooccurrence);
  void normalize_cpu(double *x, const size_t num_rows, const size_t num_cols);
  void normalize_gpu(double *x, const size_t num_rows, const size_t num_cols);
  void sen_test();
}

