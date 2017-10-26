#ifndef SOLVER_H
#define SOLVER_H

#include "sparse.h"
#include <memory>

namespace solver {
std::unique_ptr<double> power_iteration(const CSR<double> &embedding,
                                        const size_t n_iterations = 50,
                                        const size_t n_dimensions = 1000);
} // end namespace solver

#endif