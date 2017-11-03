#include <stdio.h>
#include <vector>
#include "mkl.h"
#include "sparse.h"

namespace utils {
void preprocess_cpu(CSR<double>& csr_cooccurrence);
void normalize_cpu(double* x, const size_t num_rows, const size_t num_cols);
void normalize_gpu(double* x, const size_t num_rows, const size_t num_cols);
void sen_test();
std::vector<std::string> load_vocab(const std::string& filepath);

void save_to_file(const double* const matrix, const size_t m, const size_t n,
                  const std::vector<std::string>& vocab,
                  const std::string& filename);
}
