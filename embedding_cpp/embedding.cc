#include "cusparse.h"
#include "sparse.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "solver.h"
#include "timer.h"
#include "utils.h"
#include <omp.h>

int main(void) {
  size_t n_dimensions = 50;
  size_t n_iterations = 5;
  bool gpu = false;


  mkl_set_num_threads(48);  // set num threads
  omp_set_num_threads(48);  // set num threads
  const auto loading = timer::start_clock();
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
  if(!gpu)
    utils::preprocess_cpu(csr_cooccurrence);
  timer::stop_clock("PREPROCESSING TIME", preprocessing);

  const auto solver = timer::start_clock();
  if(!gpu)
    std::unique_ptr<double> embedding = solver::cpu_power_iteration(
        csr_cooccurrence, n_iterations, n_dimensions);
  timer::stop_clock("SOLVER TIME", solver);

}
