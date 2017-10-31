#include "cusparse.h"
#include "solver.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "timer.h"
#include "utils.h"
#include <memory>
#include <string.h>

namespace solver {
std::unique_ptr<double> cpu_power_iteration(const CSR<double> &cooccurrence,
                                        const size_t n_iterations,
                                        const size_t n_dimensions) {
  double *embedding =
  (double*)malloc(cooccurrence.n * n_dimensions * sizeof(double));
  for (size_t i = 0; i < cooccurrence.n * n_dimensions; i++) {
    embedding[i] = ((int)rand()) % 2;
  }
  const auto normalization = timer::start_clock();
  utils::normalize_cpu(embedding, cooccurrence.n, n_dimensions);
  timer::stop_clock("NORMALIZING", normalization);

  double *tmp_embedding =
      (double *)malloc(cooccurrence.n * n_dimensions * sizeof(double));

  std::cout << cooccurrence.n << " " << cooccurrence.nnz << std::endl;
  std::cout << n_iterations << " " << n_dimensions << std::endl;

  const auto iterations = timer::start_clock();
  for (size_t i = 0; i < n_iterations; i++) {
    const auto itr_timer = timer::start_clock();

    auto tmp = tmp_embedding;
    tmp_embedding = embedding;
    embedding = tmp;

    memset(embedding, 0, cooccurrence.n * n_dimensions * sizeof(double));
    //#pragma omp parallel for schedule(dynamic) num_threads(48)
    for (size_t row = 0; row < cooccurrence.n; ++row) {
      for (size_t colidx = cooccurrence.rowPtr[row];
           colidx < cooccurrence.rowPtr[row + 1]; ++colidx) {
        cblas_daxpy(
            n_dimensions, cooccurrence.val[colidx],
            &tmp_embedding[cooccurrence.colInd[colidx] * n_dimensions], 1,
            &embedding[row * n_dimensions], 1);
      }
    }

    timer::stop_clock("ITERATION " + std::to_string(i), itr_timer);
    const auto normalization = timer::start_clock();
    utils::normalize_cpu(embedding, cooccurrence.n, n_dimensions);
    timer::stop_clock("NORMALIZING",normalization);
  }

  timer::stop_clock("ITERATIONS",iterations);

  free(tmp_embedding);

  return std::unique_ptr<double>(embedding);
}
} // en