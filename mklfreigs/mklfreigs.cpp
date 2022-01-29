
#include <iostream>
#include <cassert>
#include <set>
#include "mklfreigs.h"

namespace mkl_freigs{

template <class Graph,typename FP>
MKLfreigs<Graph,FP>::MKLfreigs(FP alpha,MKL_INT rank,MKL_INT q,MKL_INT s,bool convex_projection,bool gbbs,bool upper,bool analyze) {
  this->rank = rank;
  this->q = q;
  this->s = s;
  this->alpha = alpha;
  this->convex_projection = convex_projection;
  this->gbbs = gbbs;
  this->upper = upper;
  this->analyze = analyze;
  this->S = NULL;
  this->matU = NULL;
  this->degree_rt = NULL;
  this->degree_alpha = NULL;
}

template <class Graph,typename FP>
MKLfreigs<Graph,FP>::~MKLfreigs() {
  if (gbbs){
    mkl_util::util<FP>::matrix_delete(degree_alpha);
  }
  // if (S != NULL) {
  //   mkl_util::util<FP>::matrix_delete(S);
  // }
  // if (matU != NULL) {
  //   mkl_util::util<FP>::matrix_delete(matU);
  // }
  // if (degree_rt != NULL){
  //   mkl_util::util<FP>::matrix_delete(degree_rt);
  // }
}

template <class Graph,typename FP>
void MKLfreigs<Graph,FP>::run(Graph &GA) {
  MKL_INT n = GA.n;
  if (!gbbs){
    Stopwatch step_timer;
    using W = typename Graph::weight_type;
    MKL_INT i,j;
    //get D^(-1/2) and A
    auto offs = pbbs::sequence<MKL_INT>(n+1, [&] (size_t i) { return i==n ? 0 : GA.get_vertex(i).getOutDegree(); });
    size_t m = pbbslib::scan_add_inplace(offs.slice());
    std::cout << offs.size() << std::endl;
    MKL_INT* col_idx = new MKL_INT[m];
    FP* value = new FP[m];
    degree_rt = mkl_util::util<FP>::matrix_new(n,1);

    parallel_for(0, GA.n, [&] (size_t i) {
        size_t k = 0;
        size_t off_i = offs[i];
        FP degree = static_cast<FP>(GA.get_vertex(i).getOutDegree());
        degree_rt->d[i] = 1.0/sqrt(degree);
        if (degree == 0)
            degree_rt->d[i] = 0;
        // put self loop first
        auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
            assert(u != v);//no self loop
            col_idx[off_i + k] = v;
            value[off_i + k] = 1.0;
            k++;
        };
        GA.get_vertex(i).mapOutNgh(i, map_f, false);
    });
    MKL_INT* rows_start = offs.to_array();
    MKL_INT* rows_end = rows_start + 1;
    std::cout<<"fininsh get the mat_csr A and d_rt cost time:"<<step_timer.elapsed()<<std::endl;
    //computing D^{-alpha} A D^{-alpha}
    #pragma omp parallel shared(value,col_idx,rows_start,rows_end,degree_rt, n, alpha) private(i, j)
    {
    #pragma omp for
        for (i = 0; i < n; i++)
        {
          for (j = rows_start[i]; j < rows_end[i]; j++)
            value[j] = value[j] * pow(mkl_util::util<FP>::matrix_get_element(degree_rt, i, 0), alpha * 2) * pow(mkl_util::util<FP>::matrix_get_element(degree_rt, col_idx[j], 0), alpha * 2);
        }
    }
    std::cout<<"time for D^{-alpha}*A*D^{-alpha}:"<<step_timer.elapsed()<<std::endl;
    if (upper){
      std::cout<<"use upper matrix"<<std::endl;
      descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
      descrA.mode = SPARSE_FILL_MODE_UPPER;
      descrA.diag = SPARSE_DIAG_NON_UNIT;
    }
    else{
      descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    }
    sparse_status_t status;
    std::cout << "calling mkl_sparse_create_csr" << std::endl;
    Stopwatch timer_create_csr;
    status = mklhelper<FP>::mkl_sparse_create_csr(&csrA,
      SPARSE_INDEX_BASE_ZERO,
      n,
      n,
      rows_start,
      rows_end,
      col_idx,
      value);
    std::cout << "create sparse csr matrix done, status = " << status << ",n = " << n << std::endl;
    if (status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_s_create_csr error\n");
    }
    if (analyze){
      status = mkl_sparse_set_mm_hint(csrA,SPARSE_OPERATION_NON_TRANSPOSE,descrA,SPARSE_LAYOUT_ROW_MAJOR,rank,2*(q+1));
      std::cout << "set sparse mm hint done, status=" << status << std::endl;
      status = mkl_sparse_set_memory_hint(csrA, SPARSE_MEMORY_NONE);
      std::cout << "set sparse memory hint done, status=" << status << std::endl;
    }
    status = mkl_sparse_optimize(csrA);
    if (status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_optimize error\n");
    }
    std::cout << "Elapsed time: " << timer_create_csr.elapsed() << "s" << std::endl;
  }
  else{
    degree_rt = mkl_util::util<FP>::matrix_new(n,1);
    degree_alpha = mkl_util::util<FP>::matrix_new(n,1);
    parallel_for(0, GA.n, [&] (size_t i) {
        FP degree = static_cast<FP>(GA.get_vertex(i).getOutDegree());
        degree_rt->d[i] = 1.0/sqrt(degree);
        degree_alpha->d[i] = 1.0/pow(degree,alpha);
        if (degree == 0){
          degree_rt->d[i] = 0;
          degree_alpha->d[i] = 0;
        }
    });
  }

  MKL_INT l = rank + s;
  std::cout << "fast randomized eigs have l:"<<l<<" q:"<<q<<std::endl;
  Stopwatch timer_begin;
  mkl_util::mat<FP> *Y = mkl_util::util<FP>::matrix_new(n,l);
  mkl_util::mat<FP> *Omega = mkl_util::util<FP>::matrix_new(n,l);
  mkl_util::util<FP>::initialize_random_gaussian_matrix(Omega);
    
  std::cout<<"time at the begining of the freigs(initialize_random_gaussian_matrix):"<<timer_begin.elapsed()<<std::endl;
  if (!gbbs)
    mkl_util::util<FP>::sparse_csr_mm(csrA,Omega,Y,1.0,0,descrA);
  else{
    parallel_for(0,n*l,[&](MKL_INT i){
        Y->d[i] = 0.0;
    });
    mkl_util::gbbs_util<Graph,FP>::gbbs_sparse_mm(GA,Omega,Y,degree_alpha);
  }
  mkl_util::util<FP>::matrix_delete(Omega);
  mkl_util::mat<FP> *Q = mkl_util::util<FP>::matrix_new(n,l);
  mkl_util::mat<FP> *UU = mkl_util::util<FP>::matrix_new(l,l);
  mkl_util::mat<FP> *SS = mkl_util::util<FP>::matrix_new(l,1);
  mkl_util::mat<FP> *VV = mkl_util::util<FP>::matrix_new(l,l);
  mkl_util::util<FP>::eigSVD(Y,Q,SS,VV);
  mkl_util::util<FP>::matrix_delete(Y);
  
  mkl_util::mat<FP> *Q_tmp = mkl_util::util<FP>::matrix_new(n,l);
  mkl_util::mat<FP> *Q_tmp2 = mkl_util::util<FP>::matrix_new(n,l);
  std::cout<<"time at the begining of the freigs(csrmm,eigSVD):"<<timer_begin.elapsed()<<std::endl;
  MKL_INT i;
  Stopwatch timer_power_iteration;
  for (i=0;i<q;i++){
    if (!gbbs){
      mkl_util::util<FP>::sparse_csr_mm(csrA,Q,Q_tmp,1.0,0,descrA);
      mkl_util::util<FP>::sparse_csr_mm(csrA,Q_tmp,Q_tmp2,1.0,0,descrA);
    }
    else{
      parallel_for(0,n*l,[&](MKL_INT i){
        Q_tmp->d[i] = 0.0;
        Q_tmp2->d[i] = 0.0;
      });
      mkl_util::gbbs_util<Graph,FP>::gbbs_sparse_mm(GA,Q,Q_tmp,degree_alpha);
      mkl_util::gbbs_util<Graph,FP>::gbbs_sparse_mm(GA,Q_tmp,Q_tmp2,degree_alpha);
    }
    //std::cout<<"csr mm time:"<<timer_power_iteration.elapsed()<<std::endl;
    mkl_util::util<FP>::eigSVD(Q_tmp2,Q,SS,VV);
    //std::cout<<"eigSVD time:"<<timer_power_iteration.elapsed()<<std::endl;
  }
  mkl_util::util<FP>::matrix_delete(Q_tmp);
  mkl_util::util<FP>::matrix_delete(Q_tmp2);
  
  Stopwatch timer_eig_small_matrix;
  if (!convex_projection){
    std::cout<<"no convex projection"<<std::endl;
    mkl_util::mat<FP> *B = mkl_util::util<FP>::matrix_new(n,l);
    mkl_util::mat<FP> *M = mkl_util::util<FP>::matrix_new(l,l);
    if (!gbbs){
      mkl_util::util<FP>::sparse_csr_mm(csrA,Q,B,1.0,0.0,descrA);
      sparse_status_t mkl_status = mkl_sparse_destroy(csrA);
      if (mkl_status != SPARSE_STATUS_SUCCESS){
        std::cout<<"mkl_sparse_destroy error status:"<<mkl_status<<std::endl;
      }
    }
    else{
      parallel_for(0,n*l,[&](MKL_INT i){
        B->d[i] = 0.0;
      });
      mkl_util::gbbs_util<Graph,FP>::gbbs_sparse_mm(GA,Q,B,degree_alpha);
    }
    mkl_util::util<FP>::matrix_transpose_matrix_mult(Q,B,M,1.0,0);
    mkl_util::util<FP>::matrix_delete(B);
    mkl_free_buffers();
    MKL_INT info = mklhelper<FP>::LAPACKE_syev(LAPACK_ROW_MAJOR, 'V', 'U', M->ncols, M->d, M->ncols, SS->d);
    if (info!=0)
    {
      std::cout<<"some Error happen in the eig,info:"<<info<<std::endl;
      exit(1);
    }
    MKL_INT inds[rank]; 
    for(i=s;i<rank+s;i++)
    {
      inds[i-s] = rank+s-(i-s)-1;
    }
    mkl_util::mat<FP> *UU2 = mkl_util::util<FP>::matrix_new(l, rank);
    mkl_util::util<FP>::matrix_get_selected_columns(M, inds, UU2);
    S = mkl_util::util<FP>::matrix_new(rank,1);
    mkl_util::util<FP>::matrix_get_selected_rows(SS, inds, S);
    matU = mkl_util::util<FP>::matrix_new(n,rank);
    mkl_util::util<FP>::matrix_matrix_mult(Q, UU2, matU, 1.0, 0);
    mkl_util::util<FP>::matrix_delete(Q);
    mkl_util::util<FP>::matrix_delete(UU);
    mkl_util::util<FP>::matrix_delete(SS);
    mkl_util::util<FP>::matrix_delete(VV);
    mkl_util::util<FP>::matrix_delete(M);
    mkl_util::util<FP>::matrix_delete(UU2);
  }
  else{
    // https://arxiv.org/pdf/1609.00048.pdf (section 5)
    std::cout<<"convex projection"<<std::endl;
    mkl_util::mat<FP> *C = mkl_util::util<FP>::matrix_new(n,2*l);
    mkl_util::mat<FP> *B = mkl_util::util<FP>::matrix_new(n,l);
    if (!gbbs){
      mkl_util::util<FP>::sparse_csr_mm(csrA,Q,B,1.0,0,descrA);
      sparse_status_t mkl_status = mkl_sparse_destroy(csrA);
      if (mkl_status != SPARSE_STATUS_SUCCESS){
        std::cout<<"mkl_sparse_destroy error status:"<<mkl_status<<std::endl;
      }
    }
    else{
      parallel_for(0,n*l,[&](MKL_INT i){
        B->d[i] = 0.0;
      });
      mkl_util::gbbs_util<Graph,FP>::gbbs_sparse_mm(GA,Q,B,degree_alpha);
    }
    MKL_INT inds[l];
    for (MKL_INT j=0;j<l;j++)
      inds[j]=j;
    mkl_util::util<FP>::matrix_set_selected_columns(C,inds,Q);
    mkl_util::util<FP>::matrix_delete(Q);
    for (MKL_INT j=0;j<l;j++)
      inds[j]=j+l;
    mkl_util::util<FP>::matrix_set_selected_columns(C,inds,B);
    mkl_util::util<FP>::matrix_delete(B);
    mkl_util::mat<FP> *Ut = mkl_util::util<FP>::matrix_new(n,2*l);
    mkl_util::mat<FP> *T = mkl_util::util<FP>::matrix_new(2*l,2*l);
    mkl_util::util<FP>::compact_QR_factorization(C,Ut,T);
    mkl_util::util<FP>::matrix_delete(C);
    mkl_util::mat<FP> *T1 = mkl_util::util<FP>::matrix_new(2*l,l);
    mkl_util::mat<FP> *T2 = mkl_util::util<FP>::matrix_new(2*l,l);
    for (MKL_INT j=0;j<l;j++)
      inds[j]=j;
    mkl_util::util<FP>::matrix_get_selected_columns(T,inds,T1);
    for (MKL_INT j=0;j<l;j++)
      inds[j]=j+l;
    mkl_util::util<FP>::matrix_get_selected_columns(T,inds,T2);
    mkl_util::util<FP>::matrix_delete(T);
    mkl_util::mat<FP> *St1 = mkl_util::util<FP>::matrix_new(2*l,2*l);
    mkl_util::mat<FP> *St2 = mkl_util::util<FP>::matrix_new(2*l,2*l);
    mkl_util::util<FP>::matrix_matrix_transpose_mult(T1,T2,St1,1.0,0);
    mkl_util::util<FP>::matrix_matrix_transpose_mult(T2,T1,St2,1.0,0);
    mkl_util::util<FP>::matrix_delete(T1);
    mkl_util::util<FP>::matrix_delete(T2);
    mkl_util::util<FP>::matrix_matrix_add(St1,St2,1.0);
    mkl_util::util<FP>::matrix_delete(St1);
    mkl_util::util<FP>::matrix_scale(St2,0.5);
    mkl_util::mat<FP> *SS2 = mkl_util::util<FP>::matrix_new(2*l,1);
    MKL_INT info = mklhelper<FP>::LAPACKE_syevd(LAPACK_ROW_MAJOR, 'V', 'U', St2->ncols, St2->d, St2->ncols, SS2->d);//try syevd
    if (info!=0)
    {
      std::cout<<"some Error happen in the eig,info:"<<info<<std::endl;
      exit(1);
    }    
    MKL_INT ind[rank];
    for(i=rank+2*s;i<2*(rank+s);i++)
    {
      //ind[i-(k+2*s)] = i;
      ind[i-(rank+2*s)] = 2*(rank+s)-(i-rank-2*s)-1;
    }
    mkl_util::mat<FP> *UU2 = mkl_util::util<FP>::matrix_new(2*l, rank);
    mkl_util::util<FP>::matrix_get_selected_columns(St2, ind, UU2);
    S = mkl_util::util<FP>::matrix_new(rank,1);
    mkl_util::util<FP>::matrix_get_selected_rows(SS2, ind, S);
    matU = mkl_util::util<FP>::matrix_new(n,rank);
    mkl_util::util<FP>::matrix_matrix_mult(Ut, UU2, matU, 1.0, 0);
    mkl_util::util<FP>::matrix_delete(St2);
    mkl_util::util<FP>::matrix_delete(UU2);
    mkl_util::util<FP>::matrix_delete(Ut);
    mkl_util::util<FP>::matrix_delete(SS2);
  }
  
  std::cout<<"time after the power iteration(in freigs):"<<timer_eig_small_matrix.elapsed()<<std::endl;
  
  
  return ;
}

} // namespace mkl_freigs

