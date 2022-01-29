// Usage:
// numactl -i all ./sketchne -s -m -rounds 1 other_parameters
// flags:
//   required: //   optional:
//     -rounds : the number of times to run the algorithm //     -c : indicate that the graph is compressed
//     -m : indicate that the graph should be mmap'd
//     -s : indicate that the graph is symmetric

#include <cstdio>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <typeinfo>
#include <vector>
#include <iomanip> 
#include "ssr_appro.hpp"
#include "spectral_propagation.hpp"

using namespace std;
using namespace mkl_freigs;
using namespace mkl_util;

template<typename FP>
void save_emb(mat<FP> *M, string filename)
{
    MKL_INT m = M->nrows, n = M->ncols;
    FILE *fid = fopen(filename.c_str(), "wb");
    fwrite(M->d,sizeof(FP),m*n,fid);
    fclose(fid);
    return ;
}

template <class Graph>
double sketchne_mkl_runner(Graph& GA, commandLine P) {
  std::string emb_out = P.getOptionValue("-emb_out", "");
  std::string spec_out = P.getOptionValue("-spec_out", "");
  size_t window_size = static_cast<size_t>(P.getOptionLongValue("-window_size", 10));
  size_t negative_samples = static_cast<size_t>(P.getOptionLongValue("-negative_samples", 1));
  bool gbbs_sparse_mm = static_cast<bool>(P.getOptionLongValue("-gbbs_sparse_mm", 0));
  float alpha = static_cast<float>(P.getOptionDoubleValue("-alpha", 0.5));
  size_t eig_rank = static_cast<size_t>(P.getOptionLongValue("-eig_rank", 256));
  size_t power_iteration = static_cast<size_t>(P.getOptionLongValue("-power_iteration", 10));
  size_t oversampling = static_cast<size_t>(P.getOptionLongValue("-oversampling", 50));
  bool convex_projection = static_cast<bool>(P.getOptionLongValue("-convex_projection", 0));
  size_t emb_dim = static_cast<size_t>(P.getOptionLongValue("-emb_dim", 128));
  size_t eta1 = static_cast<size_t>(P.getOptionLongValue("-eta1", 8));
  size_t eta2 = static_cast<size_t>(P.getOptionLongValue("-eta2", 8));
  size_t s1 = static_cast<size_t>(P.getOptionLongValue("-s1", 50));
  size_t s2 = static_cast<size_t>(P.getOptionLongValue("-s2", 300));
  bool normalize = static_cast<bool>(P.getOptionLongValue("-normalize", 0));
  size_t order = static_cast<size_t>(P.getOptionLongValue("-order", 10));
  float theta = static_cast<float>(P.getOptionDoubleValue("-theta", 0.5));
  float mu = static_cast<float>(P.getOptionDoubleValue("-mu", 0.2));
  bool upper = static_cast<bool>(P.getOptionLongValue("-upper", 0));
  bool analyze = static_cast<bool>(P.getOptionLongValue("-analyze", 0));

  std::cout << "### Application: ssrne_mkl" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << num_workers() << std::endl;
  std::cout << "### n: " << GA.n << std::endl;
  std::cout << "### m: " << GA.m << std::endl;
  std::cout << "### Params: " << std::endl;
  std::cout << "###  -emb_out = " << emb_out       << std::endl;
  std::cout << "###  -spec_out = " << spec_out       << std::endl;
  std::cout << "###  -window_size = " << window_size  << std::endl;
  std::cout << "###  -negative_samples = " << negative_samples  << std::endl;
  std::cout << "###  -gbbs_sparse_mm = " << std::boolalpha << gbbs_sparse_mm  << std::endl;
  std::cout << "###  -alpha = " << alpha << std::endl;
  std::cout << "###  -eig_rank = " << eig_rank  << std::endl;
  std::cout << "###  -power_iteration = " << power_iteration  << std::endl;
  std::cout << "###  -oversampling = " << oversampling  << std::endl;
  std::cout << "###  -convex_projection = " << std::boolalpha << convex_projection  << " (The default setting is false)" << std::endl;
  std::cout << "###  -emb_dim = " << emb_dim  << std::endl;
  std::cout << "###  -eta1 = " << eta1  << " (it's the column sparsity parameter)" << std::endl;
  std::cout << "###  -eta2 = " << eta2  << " (it's the column sparsity parameter)" << std::endl;
  std::cout << "###  -s1 = " << s1  << std::endl;
  std::cout << "###  -s2 = " << s2  << std::endl;
  std::cout << "###  -normalize = " << std::boolalpha << normalize    << std::endl;
  std::cout << "###  -order = " << order       << std::endl;
  std::cout << "###  -theta = " << theta       << std::endl;
  std::cout << "###  -mu = " << mu       << std::endl;
  std::cout << "###  -upper = " << std::boolalpha << upper       << std::endl;
  std::cout << "###  -analyze = " << std::boolalpha << analyze << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  Stopwatch total_timer;
  using FP = float;
  std::cout << "# using float point type " << typeid(FP).name() << " (f for float, d for double)" << std::endl;
  Stopwatch step_timer;
  mat<FP>* emb = ssr_appro::sketchne<Graph,FP>(GA,window_size,negative_samples,gbbs_sparse_mm,alpha,eig_rank,power_iteration,oversampling,convex_projection,emb_dim,eta1,eta2,s1,s2,normalize,order,theta,mu,upper,analyze);
  std::cout<<"time of sketchne(freigs and sparse signed randomized single-pass svd):"<<step_timer.elapsed()<<std::endl;
  if (emb_out.size() > 0) {
    std::cout << "dump network embedding to " << emb_out << std::endl;
    save_emb<FP>(emb,emb_out);
    std::cout<<"time of save emb:"<<step_timer.elapsed()<<std::endl;
  }
  if (spec_out.size() > 0) {
    spectral_propagation::chebyshev_expansion<Graph, FP>(emb, GA, order, theta, mu);
    std::cout<<"time of spectral propagation:"<<step_timer.elapsed()<<std::endl;
    std::cout << "dump spec embedding to " << spec_out << std::endl;
    save_emb<FP>(emb,spec_out);
    std::cout<<"time of save spec:"<<step_timer.elapsed()<<std::endl;
  }
  double tt = total_timer.elapsed();
  std::cout << "### Running Time: " << tt << std::endl;
  return tt;
}

generate_symmetric_main(sketchne_mkl_runner, false);
