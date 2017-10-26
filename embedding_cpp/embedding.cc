#include "cusparse.h"
#include "sparse.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "solver.h"
#include "timer.h"

int main(void) {
  size_t n_dimensions = 1000;
  size_t n_iterations = 50;
  const std::string coocurrence_file = "../../embedding/data/cooccurrence/"
                                       "wikipedia_sample/cooccurrence.shuf.bin";

  COO<double> coo_cooccurrence = COO<double>::from_file(coocurrence_file);

  std::cout << "nnz: " << coo_cooccurrence.nnz << std::endl;
  std::cout << "n: " << coo_cooccurrence.n << std::endl;
  CSR<double> csr_cooccurrence = CSR<double>::from_coo(coo_cooccurrence);
  /*debug*/ //csr_cooccurrence.print();

  const auto solver = timer::start_clock();
  std::unique_ptr<double> embedding =
      solver::power_iteration(csr_cooccurrence, n_iterations);
  timer::stop_clock("SOLVER TIME", solver);

}
