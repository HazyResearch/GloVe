#ifndef SPARSE_H
#define SPARSE_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <climits>
#include <mkl.h>
#include <assert.h> 

// Elements from the binary COO file read on disk.
template<class T>
struct COOElem {
  int row;
  int col;
  T val;
};

// Coordinate matrix.
template <class T>
struct COO {
  size_t nnz;
  size_t n; // number of rows
  int* rowind;
  int* colind;
  T* val;

  // Constructs a COO from an array of COOElem.
  COO(const size_t num_bytes, COOElem<T>* buffer)
      : nnz(num_bytes / sizeof(COOElem<T>)) {
    rowind = (int*)malloc(nnz * sizeof(int));
    colind = (int*)malloc(nnz * sizeof(int));
    val = (T*)malloc(nnz * sizeof(T));

    size_t _n = 0;
    #pragma omp parallel for reduction(max : _n)
    for (size_t i = 0; i < nnz; ++i) {
      rowind[i] = buffer[i].row-1; // Convert 1-based to 0-based index.
      _n = (rowind[i] >= _n) ? rowind[i]:_n;
      colind[i] = buffer[i].col-1; // Convert 1-based to 0-based index.
      val[i] = buffer[i].val;
    }
    n = _n+1;
  }

  ~COO(){
    free(rowind);
    free(colind);
    free(val);
  }

  // Prints out the first N elements of the COO matrix (debug).
  // Note: only prints exact amount for even numbers (otherwise off by 1).
  void print(size_t N = ULONG_MAX) {
    bool all_elems = (N == ULONG_MAX);
    N = (all_elems) ? nnz:N;
    for (size_t i = 0; i < N/2; ++i) {
      std::cout << rowind[i] << " " << colind[i] << " " << val[i]
                << std::endl;
    }
    if(!all_elems)
      std::cout << "..." << std::endl;
    for (size_t i = nnz-(N/2); i < nnz; ++i) {
      std::cout << rowind[i] << " " << colind[i] << " " << val[i]
        << std::endl;
    }
  }

  static COO<double> from_file(const std::string& filepath) {
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

    COO<double> embedding = COO<double>(lSize, (COOElem<double>*)buffer);
    
    // terminate
    fclose(pFile);
    free(buffer);
    return embedding;
  }
};

// Compressed Sparse Row
template <class T>
struct CSR {
  int* rowPtr;
  int* colInd;
  T* val;
  int n;
  int nnz;

  ~CSR(){
    free(rowPtr);
    free(colInd);
    free(val);
  }

  static CSR<double> from_coo(const COO<double>& coo) {
    CSR<double> csr = CSR<double>();
    csr.rowPtr = (int*)malloc((coo.n + 1) * sizeof(int));
    csr.colInd = (int*)malloc(coo.nnz * sizeof(int));
    csr.val = (double*)malloc(coo.nnz * sizeof(double));
    csr.n = coo.n;
    csr.nnz = coo.nnz;

    int job[] = {
        1,    // job(1)=1 (coo->csr with no sorting)
        0,    // job(2)=0 (zero-based indexing for csr matrix)
        0,    // job(3)=0 (zero-based indexing for coo matrix)
        0,    // empty
        csr.nnz,  // job(5)=nnz (sets nnz for csr matrix)
        0     // job(6)=0 (all output arrays filled)
    };
    int info;

    mkl_dcsrcoo(job, &csr.n, csr.val, csr.colInd, csr.rowPtr, &csr.nnz, coo.val,
                coo.rowind, coo.colind, &info);

    return csr;
  }

  void print(){
    std::cout << nnz << std::endl;
    std::cout << n << std::endl;
    for(size_t row = 0; row < n; ++row){
      for(size_t colidx = rowPtr[row]; colidx < rowPtr[row+1]; ++colidx){
        assert(colidx <= nnz);
        std::cout << row << " " << colInd[colidx] << " " << val[colidx] << std::endl;
      }
    }
  }

};

#endif
