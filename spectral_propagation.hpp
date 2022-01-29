#ifndef SPECTRAL_PROPAGATION_HPP
#define SPECTRAL_PROPAGATION_HPP
#include <cmath>
#include <iostream>
#include <boost/math/special_functions/bessel.hpp>
#include "mklfreigs/mklutil.h"
#include "mklfreigs/mklfreigs.h"
#include "stopwatch.h"

using namespace boost;
using namespace mkl_freigs;
using namespace mkl_util;
using namespace std;

const float EPS = 0.00000000001f;

namespace spectral_propagation{

template<typename FP>
FP bessel(int a, FP b){
    return static_cast<FP>(boost::math::cyl_bessel_i(a, b));
}

template<typename FP>
void compute_u_sigma_root(mat<FP> *U, mat<FP> *S, mat<FP> *emb, bool normalize){
    MKL_INT i;
    MKL_INT n = U->nrows;
    MKL_INT k = U->ncols;
    if (S->d) {
      std::cout << "# computing u*sqrt(sigma), sigma(0)=" << S->d[0] << ", sigma(" << k-1 << ")=" << S->d[k-1] << std::endl;
    } else {
      std::cout << "S is NULL" << std::endl;
    }
    //memset(emb->d,0,sizeof(FP)*n*k);//may very slow when n*k is very large
    #pragma omp parallel shared(emb) private(i)
    {
    #pragma omp for
        for (i = 0; i < ((emb->nrows) * (emb->ncols)); i++)
        {
            emb->d[i] = 0.0;
        }
    }
    #pragma omp parallel shared(S,U,emb) private(i)
    {
    #pragma omp for
        for (i = 0;i < U->ncols;i++){
            FP num = sqrt(util<FP>::matrix_get_element(S,i,0));
            mklhelper<FP>::cblas_axpy(U->nrows, num, U->d+i, U->ncols, emb->d+i, U->ncols);
        }
    }
    if (normalize) {//l2 normalization
        printf("normalize l2 to emb\n");
        #pragma omp parallel shared(n, k, emb) private(i)
        {
        #pragma omp for
            for (i = 0;i < n;i++){
                FP norm = mklhelper<FP>::cblas_nrm2(k,emb->d+i*k,1);
                if (norm>EPS)
                    mklhelper<FP>::cblas_scal(k,1.0/norm,emb->d+i*k,1);
            }
        }
    }
    return ;
}


template<typename FP>
void get_embedding_dense(mat<FP> *emb,lapack_int n,lapack_int rank){
    mat<FP> *U = util<FP>::matrix_new(n,rank);
    mat<FP> *S = util<FP>::matrix_new(rank,1);
    mat<FP> *V = util<FP>::matrix_new(rank,rank);
    util<FP>::eigSVD(emb,U,S,V);
    // vec<FP> *work = util<FP>::vector_new(rank);
    // lapack_int lapack_status = mklhelper<FP>::LAPACKE_gesvd(LAPACK_ROW_MAJOR,'S', 'S', n, rank, emb->d, rank, S->d, U->d, rank, V->d, rank, work->d);//may exist some bug
    // util<FP>::vector_delete(work);
    //lapack_int lapack_status = mklhelper<FP>::LAPACKE_gesdd(LAPACK_ROW_MAJOR, 'S', n, rank,emb->d, rank, S->d, U->d, rank, V->d, rank);//gesdd slower than gesvd,gesvd get bug
    //if (lapack_status != 0)
    //    std::cout<<"some error in LAPACKE_gesvd(gesdd) info:"<<lapack_status<<std::endl;

    compute_u_sigma_root(U, S, emb, true);
    util<FP>::matrix_delete(U);
    util<FP>::matrix_delete(S);
    util<FP>::matrix_delete(V);
    return ;
}


template <class Graph, typename FP>
void chebyshev_expansion(mat<FP> *emb, Graph& GA,  size_t order=10, FP theta=0.5, FP mu=0.2){
    using W = typename Graph::weight_type;
    Stopwatch step_timer;
    std::cout<<"---------chebyshev_expansion---------"<<std::endl;
    if (order == 1)
        return ;
    MKL_INT i;
    // I will assume GA has no self loops
    MKL_INT n = emb->nrows;
    MKL_INT rank = emb->ncols;
    std::cout << n << std::endl;
    mat<FP>* degree_array;
    sparse_status_t mkl_status;
    sparse_matrix_t csrM;
    struct matrix_descr descrM;
    //compute (1-mu)*I-l1normalize(A+I)
    auto offs = pbbs::sequence<MKL_INT>(n+1, [&] (size_t i) { return i==n ? 0 : GA.get_vertex(i).getOutDegree() + 1; });
    size_t m = pbbslib::scan_add_inplace(offs.slice());
    assert(GA.m + GA.n == m);
    std::cout << offs.size() << std::endl;

    // MKL_INT* row_idx = new MKL_INT[m];
    MKL_INT* col_idx = new MKL_INT[m];
    FP* value = new FP[m];
    parallel_for(0, GA.n, [&] (size_t i) {
        size_t k = 0;
        size_t off_i = offs[i];
        FP degree = static_cast<FP>(GA.get_vertex(i).getOutDegree() + 1);
        // put self loop first
        // row_idx[off_i] = i
        col_idx[off_i] = i;
        value[off_i] = 1.0 - mu - 1.0 / degree;
        auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
            assert(u != v);
            ++k;
            // row_idx[off_i + k] = u;
            col_idx[off_i + k] = v;
            value[off_i + k] = -1.0 / degree;
        };
        GA.get_vertex(i).mapOutNgh(i, map_f, false);
    });
    
    
    descrM.type = SPARSE_MATRIX_TYPE_GENERAL;
    MKL_INT* rows_start = offs.to_array();
    MKL_INT* rows_end = rows_start + 1;
    mkl_status = mklhelper<FP>::mkl_sparse_create_csr(&csrM,
        SPARSE_INDEX_BASE_ZERO,
        n,
        n,
        rows_start,
        rows_end,
        col_idx,
        value);
    if (mkl_status != SPARSE_STATUS_SUCCESS){
        std::cout<<"mkl_sparse_create_csr error status:"<<mkl_status<<std::endl;
    }

    mkl_status = mkl_sparse_optimize(csrM);
    if (mkl_status != SPARSE_STATUS_SUCCESS){
        std::cout<<"mkl_sparse_optimize error status:"<<mkl_status<<std::endl;
    }
    std::cout<<"the begining of spectral propagation, mkl_sparse init and optimize cost time:"<<step_timer.elapsed()<<" s"<<std::endl;


    mat<FP>* conv = util<FP>::matrix_new(n,rank);
    mat<FP>* tmp = util<FP>::matrix_new(n,rank);
    mat<FP>* L0 = util<FP>::matrix_new(n,rank);
    mat<FP>* L1 = util<FP>::matrix_new(n,rank);
    mat<FP>* L2 = util<FP>::matrix_new(n,rank);
    util<FP>::sparse_csr_mm(csrM,emb,tmp,1.0,0,descrM);//tmp = M*emb
    util<FP>::matrix_copy(emb,L0);//L0 = emb
    util<FP>::matrix_copy(emb,L1);//L1 = emb
    util<FP>::sparse_csr_mm(csrM,tmp,L1,0.5,-1.0,descrM);//L1 = 0.5*M*M*emb - emb
    parallel_for(0,n*rank,[&](MKL_INT i){
        conv->d[i] = 0.0;
    });
    util<FP>::matrix_matrix_add(L0,conv,bessel(0,theta));//conv = iv(0, theta) * L0
    util<FP>::matrix_matrix_add(L1,conv,-2*bessel(1,theta));//conv = conv - 2 * iv(1, theta) * Lx1
    std::cout<<"conv = conv - 2 * iv(1, theta) * Lx1 finished,time:"<<step_timer.elapsed()<<" s"<<std::endl;

    for (i = 2;i < static_cast<MKL_INT>(order);i++){
        util<FP>::sparse_csr_mm(csrM,L1,tmp,1.0,0,descrM);//tmp = M*L1
        util<FP>::matrix_copy(L0,L2);//L2 = L0
        util<FP>::sparse_csr_mm(csrM,tmp,L2,1.0,-1.0,descrM);//L2 = M*M*L1 - L2 = M*M*L1 - L0
        util<FP>::matrix_matrix_add(L1,L2,-2.0);//L2 = M*M*L1 - L0 - 2*L1
        if (i % 2 == 0)
            util<FP>::matrix_matrix_add(L2,conv,2*bessel(i,theta));
        else
            util<FP>::matrix_matrix_add(L2,conv,-2*bessel(i,theta));
        std::swap(L1->d,L0->d);
        std::swap(L2->d,L1->d);
        std::cout<<"bessel once cost time:"<<step_timer.elapsed()<<" s"<<std::endl;
    }
    mkl_status = mkl_sparse_destroy(csrM);
    if (mkl_status != SPARSE_STATUS_SUCCESS){
        std::cout<<"mkl_sparse_destroy error status:"<<mkl_status<<std::endl;
    }
    util<FP>::matrix_delete(L0);
    util<FP>::matrix_delete(L1);
    util<FP>::matrix_delete(L2);
    util<FP>::matrix_copy(emb,tmp);
    util<FP>::matrix_matrix_add(conv,tmp,-1.0);//tmp = emb - conv
    util<FP>::matrix_delete(conv);

    parallel_for(0, GA.n, [&] (size_t i) {
        // put self loop first
        for (MKL_INT j = 0;j<rank;j++)
            emb->d[i*rank+j] = tmp->d[i*rank+j];
        auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
            mklhelper<FP>::cblas_axpy(rank,1.0,tmp->d+v*rank,1,emb->d+u*rank,1);
        };
        GA.get_vertex(i).mapOutNgh(i, map_f, false);
    });
    util<FP>::matrix_delete(tmp);
    get_embedding_dense(emb, n, rank); //or L2 normalize to emb
    std::cout<<"last period and get embedding dense cost time:"<<step_timer.elapsed()<<" s"<<std::endl;
    return ;
}

}

#endif



