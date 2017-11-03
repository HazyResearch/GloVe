#include "utils.h"
#include <cstring>
#include <memory>
#include <fstream>

namespace {
void qr(double* const _Q, double* const _A, const size_t _m, const size_t _n) {
  // Maximal rank is used by Lapacke
  const size_t rank = std::min(_m, _n);

  // Tmp Array for Lapacke
  const std::unique_ptr<double[]> tau(new double[rank]);

  // Calculate QR factorisations
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int)_m, (int)_n, _A, (int)_n, tau.get());

  // Copy the upper triangular Matrix R (rank x _n) into position
  /*
  for (size_t row = 0; row < rank; ++row) {
    memset(_R + row * _n, 0, row * sizeof(double));  // Set starting zeros
    memcpy(
        _R + row * _n + row, _A + row * _n + row,
        (_n - row) *
            sizeof(double));  // Copy upper triangular part from Lapack result.
  }*/

  // Create orthogonal matrix Q (in tmpA)
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int)_m, (int)rank, (int)rank, _A, (int)_n,
                 tau.get());

  // Copy Q (_m x rank) into position
  if (_m == _n) {
    std::memcpy(_Q, _A, sizeof(double) * (_m * _n));
  } else {
    for (size_t row = 0; row < _m; ++row) {
      std::memcpy(_Q + row * rank, _A + row * _n, sizeof(double) * (rank));
    }
  }
}
}

namespace utils {

void preprocess_cpu(CSR<double>& csr_cooccurrence) {
  double D = 0.0;
  double* wc = (double*)malloc(csr_cooccurrence.n * sizeof(double));
  double* wc0 = (double*)malloc(csr_cooccurrence.nnz * sizeof(double));
#pragma omp parallel for reduction(+ : D)
  for (size_t row = 0; row < csr_cooccurrence.n; ++row) {
    double sum = 0.0;
    for (size_t colidx = csr_cooccurrence.rowPtr[row];
         colidx < csr_cooccurrence.rowPtr[row + 1]; ++colidx) {
      sum += csr_cooccurrence.val[colidx];
    }
    D += sum;
    wc[row] = sum;
  }

  double* wc1 = (double*)malloc(csr_cooccurrence.nnz * sizeof(double));
  double* D_vals = (double*)malloc(csr_cooccurrence.nnz * sizeof(double));
  size_t i = 0;
  for (size_t row = 0; row < csr_cooccurrence.n; ++row) {
    for (size_t colidx = csr_cooccurrence.rowPtr[row];
         colidx < csr_cooccurrence.rowPtr[row + 1]; ++colidx) {
      wc0[i] = wc[row];
      wc1[i] = wc[csr_cooccurrence.colInd[colidx]];
      D_vals[i++] = D;
    }
  }

  /*
  for(size_t row = 0; row < csr_cooccurrence.n; ++row){
  for(size_t colidx = csr_cooccurrence.rowPtr[row]; colidx <
  csr_cooccurrence.rowPtr[row+1]; ++colidx){
    std::cout << "WC0: " << row << " " << csr_cooccurrence.colInd[colidx] << " "
  << wc0[colidx] << std::endl;
    std::cout << "WC1: "  << row << " " << csr_cooccurrence.colInd[colidx] << "
  " << wc1[colidx] << std::endl;
  }
  }
  */

  // v = torch.log(v) + torch.log(torch.DoubleTensor(nnz).fill_(D)) -
  // torch.log(wc0) - torch.log(wc1)
  vdLn(csr_cooccurrence.nnz, csr_cooccurrence.val, csr_cooccurrence.val);
  vdLn(csr_cooccurrence.nnz, D_vals, D_vals);
  vdLn(csr_cooccurrence.nnz, wc0, wc0);
  vdLn(csr_cooccurrence.nnz, wc1, wc1);
  vdAdd(csr_cooccurrence.nnz, csr_cooccurrence.val, D_vals,
        csr_cooccurrence.val);
  vdSub(csr_cooccurrence.nnz, csr_cooccurrence.val, wc0, csr_cooccurrence.val);
  vdSub(csr_cooccurrence.nnz, csr_cooccurrence.val, wc1, csr_cooccurrence.val);

// clamp(min=0)
#pragma omp parallel for
  for (size_t row = 0; row < csr_cooccurrence.n; ++row) {
    double sum = 0.0;
    for (size_t colidx = csr_cooccurrence.rowPtr[row];
         colidx < csr_cooccurrence.rowPtr[row + 1]; ++colidx) {
      if (csr_cooccurrence.val[colidx] < 0) csr_cooccurrence.val[colidx] = 0;
    }
  }

  /*
  for(size_t row = 0; row < csr_cooccurrence.n; ++row){
  for(size_t colidx = csr_cooccurrence.rowPtr[row]; colidx <
  csr_cooccurrence.rowPtr[row+1]; ++colidx){
    std::cout << "PREPRCESSED: "  << row << " " <<
  csr_cooccurrence.colInd[colidx] << " " << csr_cooccurrence.val[colidx] <<
  std::endl;
  }
  }
  */
}

void normalize_cpu(double* x, const size_t num_rows, const size_t num_cols) {
  for (size_t i = 0; i < num_cols; i++) {
    std::cout << cblas_dnrm2(num_rows, &x[i], num_cols) << " ";
  }
  std::cout << std::endl;
  qr(x, x, num_rows, num_cols);
  /*
  for(size_t i = 0; i < 5; i++){
    std::cout << x[i] << std::endl;
  }*/
}

std::vector<std::string> load_vocab(const std::string& filepath) {
  FILE* pFile;
  long lSize;
  char* buffer;
  size_t result;

  pFile = fopen(filepath.c_str(), "rb");
  if (pFile == NULL) {
    fputs("File error", stderr);
    exit(1);
  }

  // obtain file size:
  fseek(pFile, 0, SEEK_END);
  lSize = ftell(pFile);
  rewind(pFile);

  // allocate memory to contain the whole file:
  buffer = (char*)malloc(sizeof(char) * lSize);
  if (buffer == NULL) {
    fputs("Memory error", stderr);
    exit(2);
  }

  // copy the file into the buffer:
  result = fread(buffer, 1, lSize, pFile);
  if (result != lSize) {
    fputs("Reading error", stderr);
    exit(3);
  }

  std::vector<std::string> vocab;
  char* pch = strtok(buffer, " \n");
  while (pch != NULL) {
    vocab.push_back(pch);
    pch = strtok(NULL, " \n");

    pch = strtok(NULL, " \n");
  }

  // terminate
  fclose(pFile);
  free(buffer);
  return vocab;
}

void save_to_file(const double * const matrix, const size_t m, const size_t n,
                  const std::vector<std::string>& vocab,
                  const std::string& filename) {
  std::ofstream myfile;
  myfile.open(filename);
  for (size_t i = 0; i < m; ++i) {
    myfile << vocab.at(i) << " ";
    for (size_t j = 0; j < n; ++j) {
      myfile << matrix[i * n + j] << " ";
    }
    myfile << "\n";
  }
}
}
