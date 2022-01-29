#pragma once
#include "mkl.h"
#include "mklhelper.h"
#include "mklutil.h"
#include "stopwatch.h"

namespace mkl_freigs {

template <class Graph,typename FP>
class MKLfreigs {
public:
  MKLfreigs(FP alpha,MKL_INT rank,MKL_INT q,MKL_INT s,bool convex_projection,bool gbbs,bool upper,bool analyze);
  ~MKLfreigs();
  void run(Graph& GA);
  mkl_util::mat<FP> *S;
  mkl_util::mat<FP> *matU;
  mkl_util::mat<FP> *degree_rt;
  mkl_util::mat<FP> *degree_alpha;
private:
  sparse_matrix_t csrA;
  struct matrix_descr descrA;
  MKL_INT rank, q, s;
  FP alpha;
  bool convex_projection;
  bool gbbs;
  bool upper;
  bool analyze;
};

} // namespace mkl_freigs
#include "mklfreigs.cpp"
