#include "cxxopts.h"
#include "cusparse.h"
#include "sparse.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "solver.h"
#include "timer.h"
#include "utils.h"
#include <omp.h>

int main(int argc, char *argv[], char *envp[]) {
  size_t n_dimensions = 50;
  size_t n_iterations = 10;
  size_t n_threads = 48;
  bool gpu = true;

  std::string cooccurrence_file = "cooccurrence.shuf.bin";
  std::string vocab_file = "vocab.txt";
  std::string output_file = "vectors.txt";

  try {
    cxxopts::Options parser(argv[0], "Compute embeddings");

    parser.add_options()
      ("h,help", "print help")
      ("d,dim", "number of dimensions in embedding", cxxopts::value<size_t>(n_dimensions)->default_value("50"))
      ("n,iter", "number of iterations", cxxopts::value<size_t>(n_iterations)->default_value("10"))
      ("t,threads", "number of CPU threads", cxxopts::value<size_t>(n_threads)->default_value("48"))
      // ("g,gpu", "toogle GPU use", cxxopts::value<bool>(gpu)->default_value(true))
      ("c,cooccurrence", "name of cooccurrence file", cxxopts::value<std::string>(cooccurrence_file)->default_value("cooccurrence.shuf.bin"))
      ("v,vocab", "name of vocab file", cxxopts::value<std::string>(vocab_file)->default_value("vocab.txt"))
      ("o,vectors", "name of vectors output file", cxxopts::value<std::string>(output_file)->default_value("vectors.txt"))
    ;
    parser.parse(argc, argv);

    if (parser.count("help") != 0) {
      std:: cout << parser.help() << std::endl;
      exit(0);
    }
  }
  catch (const cxxopts::OptionException& e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }

  std::cout << "dimension: " << n_dimensions << std::endl;
  std::cout << "iterations: " << n_iterations << std::endl;
  std::cout << "threads: " << n_threads << std::endl;

  mkl_set_num_threads(n_threads);  // set num threads
  omp_set_num_threads(n_threads);  // set num threads
  const auto loading = timer::start_clock();

  COO<double> coo_cooccurrence = COO<double>::from_file(cooccurrence_file);
  std::vector<std::string> vocab = utils::load_vocab(vocab_file);
  timer::stop_clock("LOADING", loading);
  
  std::cout << "nnz: " << coo_cooccurrence.nnz << std::endl;
  std::cout << "n: " << coo_cooccurrence.n << std::endl;

  const auto csr_build = timer::start_clock();
  CSR<double> csr_cooccurrence = CSR<double>::from_coo(coo_cooccurrence);
  /*debug*/ //csr_cooccurrence.print();
  timer::stop_clock("CSR BUILD", csr_build);

  const auto preprocessing = timer::start_clock();
  if(!gpu)
    utils::preprocess_cpu(csr_cooccurrence);
  timer::stop_clock("PREPROCESSING TIME", preprocessing);

  const auto solver = timer::start_clock();
  std::unique_ptr<double> embedding;
  if(!gpu)
    embedding = solver::cpu_power_iteration(
        csr_cooccurrence, n_iterations, n_dimensions);
  timer::stop_clock("SOLVER TIME", solver);

  const auto save_to_file = timer::start_clock();
  utils::save_to_file(embedding.get(), csr_cooccurrence.n, n_dimensions, vocab,
                      output_file);
  timer::stop_clock("WRITING FILE", save_to_file);

}
