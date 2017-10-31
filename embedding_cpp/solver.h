#ifndef SOLVER_H
#define SOLVER_H

#include "sparse.h"
#include <memory>

namespace solver {
std::unique_ptr<double> gpu_power_iteration(const CSR<double> &embedding,
                                        const size_t n_iterations,
                                        const size_t n_dimensions);

std::unique_ptr<double> cpu_power_iteration(const CSR<double> &embedding,
                                        const size_t n_iterations,
                                        const size_t n_dimensions);

} // end namespace solver

#endif