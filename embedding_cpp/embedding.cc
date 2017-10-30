#include "cusparse.h"
#include "sparse.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "solver.h"
#include "timer.h"
#include <omp.h>

int main(void) {
  mkl_set_num_threads(48);  // set num threads
  omp_set_num_threads(48);  // set num threads
  const auto loading = timer::start_clock();
  size_t n_dimensions = 1000;
  size_t n_iterations = 5;
  const std::string coocurrence_file = "/dfs/scratch0/caberger/datasets/coocurrence/wikipedia/cooccurrence.shuf.bin";
  //const std::string coocurrence_file = "/dfs/scratch0/caberger/datasets/coocurrence/pubmed/cooccurrence.shuf.bin";

  COO<double> coo_cooccurrence = COO<double>::from_file(coocurrence_file);
  timer::stop_clock("LOADING",loading);

  
  std::cout << "nnz: " << coo_cooccurrence.nnz << std::endl;
  std::cout << "n: " << coo_cooccurrence.n << std::endl;

  const auto csr_build = timer::start_clock();
  CSR<double> csr_cooccurrence = CSR<double>::from_coo(coo_cooccurrence);
  /*debug*/ //csr_cooccurrence.print();
  timer::stop_clock("CSR BUILD",csr_build);

  const auto preprocessing = timer::start_clock();

  /*
  double D  = 0.0;
  #pragma omp parallel for reduction(+ : D)
  for (size_t i = 0; i < csr_cooccurrence.nnz; ++i) {
    D += csr_cooccurrence.val[i];
  }
  */
  double D  = 0.0;
  double* wc = (double*)malloc(csr_cooccurrence.n*sizeof(double));
  double* wc0 = (double*)malloc(csr_cooccurrence.nnz*sizeof(double));
  #pragma omp parallel for reduction(+ : D)
  for(size_t row = 0; row < csr_cooccurrence.n; ++row){
    double sum = 0.0;
    for(size_t colidx = csr_cooccurrence.rowPtr[row]; colidx < csr_cooccurrence.rowPtr[row+1]; ++colidx){
      //std::cout << row << " " << csr_cooccurrence.colInd[colidx] << " " << csr_cooccurrence.val[colidx] << std::endl;
      sum += csr_cooccurrence.val[colidx];
    }
    D += sum;
    wc[row] = sum;
  }

  double* wc1 = (double*)malloc(csr_cooccurrence.nnz*sizeof(double));
  double* D_vals = (double*)malloc(csr_cooccurrence.nnz*sizeof(double));
  size_t i = 0;
  for(size_t row = 0; row < csr_cooccurrence.n; ++row){
    for(size_t colidx = csr_cooccurrence.rowPtr[row]; colidx < csr_cooccurrence.rowPtr[row+1]; ++colidx){
      wc0[i]= wc[row];
      wc1[i] = wc[csr_cooccurrence.colInd[colidx]]; 
      D_vals[i++] = D;
    }
  }

  /*
  for(size_t row = 0; row < csr_cooccurrence.n; ++row){
    for(size_t colidx = csr_cooccurrence.rowPtr[row]; colidx < csr_cooccurrence.rowPtr[row+1]; ++colidx){
      std::cout << "WC0: " << row << " " << csr_cooccurrence.colInd[colidx] << " " << wc0[colidx] << std::endl;
      std::cout << "WC1: "  << row << " " << csr_cooccurrence.colInd[colidx] << " " << wc1[colidx] << std::endl;
    }
  }
  */

  // v = torch.log(v) + torch.log(torch.DoubleTensor(nnz).fill_(D)) - torch.log(wc0) - torch.log(wc1)
  vdLn(csr_cooccurrence.nnz, csr_cooccurrence.val, csr_cooccurrence.val);
  vdLn(csr_cooccurrence.nnz, D_vals, D_vals);
  vdLn(csr_cooccurrence.nnz, wc0, wc0);
  vdLn(csr_cooccurrence.nnz, wc1, wc1);
  vdAdd(csr_cooccurrence.nnz, csr_cooccurrence.val, D_vals, csr_cooccurrence.val);
  vdSub(csr_cooccurrence.nnz, csr_cooccurrence.val, wc0, csr_cooccurrence.val);
  vdSub(csr_cooccurrence.nnz, csr_cooccurrence.val, wc1, csr_cooccurrence.val);

  // clamp(min=0)
  #pragma omp parallel for
  for(size_t row = 0; row < csr_cooccurrence.n; ++row){
    double sum = 0.0;
    for(size_t colidx = csr_cooccurrence.rowPtr[row]; colidx < csr_cooccurrence.rowPtr[row+1]; ++colidx){
      if(csr_cooccurrence.val[colidx] < 0)
        csr_cooccurrence.val[colidx] = 0;
    }
  }

  /*
  for(size_t row = 0; row < csr_cooccurrence.n; ++row){
    for(size_t colidx = csr_cooccurrence.rowPtr[row]; colidx < csr_cooccurrence.rowPtr[row+1]; ++colidx){
      std::cout << "PREPRCESSED: "  << row << " " << csr_cooccurrence.colInd[colidx] << " " << csr_cooccurrence.val[colidx] << std::endl;
    }
  }
  */


  timer::stop_clock("PREPROCESSING TIME", preprocessing);

  const auto solver = timer::start_clock();
  std::unique_ptr<double> embedding =
      solver::power_iteration(csr_cooccurrence, n_iterations, n_dimensions);
  timer::stop_clock("SOLVER TIME", solver);

}
