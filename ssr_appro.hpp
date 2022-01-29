#pragma once
#include "mklfreigs/mklutil.h"
#include "mklfreigs/mklfreigs.h"
#include "spectral_propagation.hpp"
#include "sp_sign.hpp"
#include "stopwatch.h"

namespace ssr_appro{

template<typename FP>
void deepwalk_filter(mat<FP> *evals, MKL_INT h,MKL_INT window_size)
{
    MKL_INT i;
    for (i = 0; i < h; i++)
    {
        float x = evals->d[i];
        if (x >= 1)
            evals->d[i] = 1;
        else
            evals->d[i] = x * (1 - pow(x, window_size)) / (1 - x) /window_size;
        evals->d[i] = std::max(evals->d[i], (FP)0.0);
    }
    return;
}


template<typename FP>
mat<FP> *sparse_sign_randomized_single_pass_svd(mat<FP> *F,mat<FP> *CF,MKL_INT dim,MKL_INT s1,MKL_INT s2,MKL_INT eta1,MKL_INT eta2,bool normalize){
    MKL_INT n = F->nrows,eig_rank = F->ncols;
    MKL_INT l1 = dim+s1;
    MKL_INT l2 = dim+s2;
    std::cout<<"set l1:"<<l1<<" and l2:"<<l2<<std::endl;
    mat_csc<FP> *spMat1 = util<FP>::csc_matrix_new(n, l1, eta1 * l1);
    MKL_INT *unique_pos = new MKL_INT[l1*eta1]();
    MKL_INT unique_num = sp_sign_gen_csc<FP>(n,l1,eta1,unique_pos,spMat1);
    MKL_INT *num2index = new MKL_INT[n]();
    for (MKL_INT i = 0; i < unique_num; ++i)
        num2index[unique_pos[i]] = i;
    std::cout<<"total nnz is l1*eta1:"<<l1*eta1<<" and unique num = "<<unique_num<<std::endl;
    mat<FP> *CF_compressed = util<FP>::matrix_new(eig_rank, unique_num);
    util<FP>::matrix_get_selected_columns(CF, unique_pos, CF_compressed);
    std::cout<<"compute get selected columns finished"<<std::endl;
    
    Stopwatch timer;
    float ratio = 1.0;
    std::cout<<"caculate Q = M_compressed * spMat1"<<std::endl;
    mat<FP> *Y = util<FP>::matrix_new(n, l1);
    MKL_INT i,j,k;
    MKL_INT batch_size = static_cast<MKL_INT>(std::min(1e6,n*1.0));// batch_size parameter is used for batch matrix matrix multiplication
    MKL_INT step = n / batch_size;
    if (n % batch_size != 0)
        ++step;
    std::cout<<"batch_size:"<<batch_size<<",step:"<<step<<std::endl;

    for (MKL_INT cnt = 0;cnt<step;cnt++){
        MKL_INT start_pos = cnt * batch_size;
        MKL_INT end_pos = (cnt+1) * batch_size - 1;
        if (end_pos >= n)
            end_pos = n - 1;
        MKL_INT pos_size = end_pos - start_pos + 1;
        mat<FP> *M_compressed = util<FP>::matrix_new(pos_size,unique_num);
        mklhelper<FP>::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pos_size, unique_num, eig_rank, 1.0, F->d+start_pos*F->ncols, eig_rank, CF_compressed->d, CF_compressed->ncols, 0.0, M_compressed->d, M_compressed->ncols);
        #pragma omp parallel shared(start_pos,end_pos,n, l1, M_compressed, spMat1, Y) private(i,j,k)
        {
        #pragma omp for
            for (i = start_pos; i < end_pos; ++i)
            {
                for (j = 0; j < l1; ++j)
                {
                    float sum = 0;
                    for (k = spMat1->pointerB[j]; k < spMat1->pointerE[j]; ++k)
                    {
                        float ans = util<FP>::matrix_get_element(M_compressed, i-start_pos, num2index[spMat1->rows[k]]);
                        if (ans <= 0)
                            ans = 0;
                        else
                            ans = log(ans+1);
                        if (spMat1->values[k] > 0)
                            sum += ans*ratio;
                        else
                            sum -= ans*ratio;
                    }
                    util<FP>::matrix_set_element(Y, i, j, sum);
                }
            }
        }
        util<FP>::matrix_delete(M_compressed);
    }  
    std::cout<<"compute Q cost time: "<<timer.elapsed()<<" s"<<std::endl;

    util<FP>::csc_matrix_delete(spMat1);
    util<FP>::matrix_delete(CF_compressed);
    delete [] unique_pos;
    delete [] num2index;
    mat<FP> *Q = util<FP>::matrix_new(n, l1);
    mat<FP> *SS = util<FP>::matrix_new(l1,1);
    mat<FP> *VV = util<FP>::matrix_new(l1,l1);
    util<FP>::eigSVD(Y,Q,SS,VV);
    //util<FP>::QR_factorization_getQ_inplace(Q);
    util<FP>::matrix_delete(Y);
    util<FP>::matrix_delete(SS);
    util<FP>::matrix_delete(VV);
    std::cout<<"compute QR for Q time: "<<timer.elapsed()<<" s"<<std::endl;

    mat_csc<FP> *spMat2 = util<FP>::csc_matrix_new(n, l2, eta2 * l2);
    unique_pos = new MKL_INT[l2*eta2]();
    unique_num = sp_sign_gen_csc<FP>(n,l2,eta2,unique_pos,spMat2);
    num2index = new MKL_INT[n]();
    for (i = 0; i < unique_num; ++i)
        num2index[unique_pos[i]] = i;
    std::cout<<"total nnz is l2*eta2:"<<l2*eta2<<" and unique num = "<<unique_num<<std::endl;

    std::cout<<"caculate M_core = F_compressed * CF_compressed"<<std::endl;
    mat<FP> *M_core = util<FP>::matrix_new(unique_num, unique_num);
    #pragma omp parallel shared(unique_num,unique_pos,eig_rank, F, CF, M_core) private(i,j,k)
    {
    #pragma omp for
        for (i = 0; i < unique_num; ++i)
        {
            for (j = 0; j < unique_num; ++j)
            {
                float sum = 0;
                for (k = 0; k < eig_rank; ++k)
                    sum += util<FP>::matrix_get_element(F, unique_pos[i], k) * util<FP>::matrix_get_element(CF, k, unique_pos[j]);
                if (sum > 0)
                    util<FP>::matrix_set_element(M_core, i, j, log(sum+1));
                else
                    util<FP>::matrix_set_element(M_core, i, j, 0);
            }
        }
    }
    delete [] unique_pos;
    util<FP>::matrix_delete(F);
    util<FP>::matrix_delete(CF);

    std::cout<<"caculate M_right = M_core * spMat2"<<std::endl;
    mat<FP> *M_right = util<FP>::matrix_new(unique_num, l2);
    #pragma omp parallel shared(l2, unique_num, spMat2, M_core, M_right) private(i,j,k)
    {
    #pragma omp for
        for (i = 0; i < unique_num; ++i)
        {
            for (j = 0; j < l2; ++j)
            {
                float sum = 0;
                for (k = spMat2->pointerB[j]; k < spMat2->pointerE[j]; ++k)
                {
                    float ans = util<FP>::matrix_get_element(M_core, i, num2index[spMat2->rows[k]]);
                    if (spMat2->values[k] > 0)
                        sum += ans*ratio;
                    else
                        sum -= ans*ratio;
                }
                util<FP>::matrix_set_element(M_right, i, j, sum);
            }
        }
    }
    util<FP>::matrix_delete(M_core);

    std::cout<<"caculate Z = spMat2' * M_right"<<std::endl;
    mat<FP> *Z = util<FP>::matrix_new(l2, l2);
    #pragma omp parallel shared(l2, spMat2, Z, M_right) private(i,j,k)
    {
    #pragma omp for
        for (i = 0; i < l2; ++i)
        {
            for (j = 0; j < l2; ++j)
            {
                float sum = 0;
                for (k = spMat2->pointerB[i]; k < spMat2->pointerE[i]; ++k)
                {
                    float ans = util<FP>::matrix_get_element(M_right, num2index[spMat2->rows[k]], j);
                    if (spMat2->values[k] > 0)
                        sum += ans*ratio;
                    else
                        sum -= ans*ratio;
                }
                util<FP>::matrix_set_element(Z, i, j, sum);
            }
        }
    }
    util<FP>::matrix_delete(M_right);
    delete [] num2index;
    std::cout<<"compute Z cost time: "<<timer.elapsed()<<" s"<<std::endl;
    

    std::cout<<"caculate temp = spMat2' * Q"<<std::endl;
    mat<FP> *temp = util<FP>::matrix_new(l2, l1);
    #pragma omp parallel shared(l2, l1, spMat2, Q, temp) private(i,j,k)
    {
    #pragma omp for
        for (i = 0; i < l2; ++i)
        {
            for (j = 0; j < l1; ++j)
            {
                float sum = 0;
                for (k = spMat2->pointerB[i]; k < spMat2->pointerE[i]; ++k)
                {
                    float ans = util<FP>::matrix_get_element(Q, spMat2->rows[k], j);
                    if (spMat2->values[k] > 0)
                        sum += ans*ratio;
                    else
                        sum -= ans*ratio;
                }
                util<FP>::matrix_set_element(temp, i, j, sum);
            }
        }
    }
    util<FP>::csc_matrix_delete(spMat2);

    mat<FP> *Utemp = util<FP>::matrix_new(l2, l1);
    mat<FP> *Rtemp = util<FP>::matrix_new(l1, l1);
    memset(Utemp->d,0,sizeof(FP)*l2*l1);
    memset(Rtemp->d,0,sizeof(FP)*l1*l1);
    util<FP>::compact_QR_factorization(temp, Utemp, Rtemp);
    util<FP>::matrix_delete(temp);
    mat<FP> *Rtemp2 = util<FP>::matrix_new(l1,l1);
    util<FP>::matrix_copy(Rtemp,Rtemp2);

    mat<FP> *UtZ = util<FP>::matrix_new(l1, l2);
    util<FP>::matrix_transpose_matrix_mult(Utemp, Z, UtZ, 1.0, 0);
    util<FP>::matrix_delete(Z);

    util<FP>::linear_solve_Uxb(Rtemp, UtZ);
    mat<FP> *UtTt = util<FP>::matrix_new(l1, l1);
    util<FP>::matrix_transpose_matrix_transpose_mult(Utemp, UtZ, UtTt, 1.0, 0);
    
    util<FP>::linear_solve_Uxb(Rtemp2, UtTt);
    Z = util<FP>::matrix_new(l1, l1);
    util<FP>::matrix_build_transpose(UtTt,Z);
    util<FP>::matrix_delete(Utemp);
    util<FP>::matrix_delete(Rtemp);
    util<FP>::matrix_delete(Rtemp2);
    util<FP>::matrix_delete(UtTt);
    util<FP>::matrix_delete(UtZ);

    mat<FP> *Uc = util<FP>::matrix_new(l1, l1);
    mat<FP> *Sc = util<FP>::matrix_new(l1, l1);
    mat<FP> *Vc = util<FP>::matrix_new(l1, l1);
    util<FP>::singular_value_decomposition(Z, Uc, Sc, Vc);
    util<FP>::matrix_delete(Z);
    mat<FP> *Uk = util<FP>::matrix_new(l1,dim);
    MKL_INT inds[dim];
    for (MKL_INT i = 0;i<dim;i++){
        inds[i] = i;
    }
    util<FP>::matrix_get_selected_columns(Uc,inds,Uk);
    
    mat<FP> *U = util<FP>::matrix_new(n, dim);
    mat<FP> *S = util<FP>::matrix_new(dim, 1);
    util<FP>::matrix_matrix_mult(Q, Uk, U, 1.0, 0);
    util<FP>::matrix_delete(Q);
    util<FP>::matrix_delete(Uk);

    for (MKL_INT i = 0; i < dim; ++i)
    {
        util<FP>::matrix_set_element(S, i, 0, util<FP>::matrix_get_element(Sc, i, i));
    }

    FP maxs = DBL_MIN;
    FP mins = DBL_MAX;
    for (MKL_INT i = 0; i < dim; i++)
    {
        maxs = std::max(maxs, S->d[i]);
        mins = std::min(mins, S->d[i]);
    }
    std::cout<<"singular value max:"<<maxs<<"---singular value min:"<<mins<<std::endl;
    util<FP>::matrix_delete(Uc);
    util<FP>::matrix_delete(Sc);
    util<FP>::matrix_delete(Vc);

    mat<FP> *emb = util<FP>::matrix_new(n,dim);
    spectral_propagation::compute_u_sigma_root<FP>(U,S,emb,normalize);
    util<FP>::matrix_delete(U);
    util<FP>::matrix_delete(S);

    std::cout<<"time cost in computing solve and svd:"<<timer.elapsed()<<"s"<<std::endl;
    
    return emb;
}



template<class Graph,typename FP>
mat<FP>* sketchne(Graph& GA,size_t window_size,size_t negative_samples,bool gbbs,FP alpha,size_t eig_rank,size_t power_iteration,size_t oversampling,bool convex_projection,size_t emb_dim,size_t eta1,size_t eta2,size_t s1,size_t s2,bool normalize,size_t order,FP theta,FP mu,bool upper,bool analyze){
    std::cout<<"start ssrne"<<std::endl;
    MKL_INT n = GA.n;
    MKL_INT m = GA.m;
    MKL_INT dim = emb_dim;
    mkl_freigs::MKLfreigs<Graph,FP> freigsOfNormalizedGraph(alpha,eig_rank,power_iteration,oversampling,convex_projection,gbbs,upper,analyze);
    Stopwatch freigs_timer;
    freigsOfNormalizedGraph.run(GA);
    std::cout<<"time for freigs:"<<freigs_timer.elapsed()<<std::endl;
    std::cout<<"evals max:"<<freigsOfNormalizedGraph.S->d[0]<<"---evals min:"<<freigsOfNormalizedGraph.S->d[eig_rank-1]<<std::endl;
    mat<FP> *matU = freigsOfNormalizedGraph.matU;
    mat<FP> *evals = freigsOfNormalizedGraph.S;
    mat<FP> *d_rt = freigsOfNormalizedGraph.degree_rt;
    mat<FP> *F;
    mat<FP> *CF;
    mkl_free_buffers();
    Stopwatch timer;
    std::cout<<"alpha:"<<alpha<<std::endl;
    if (alpha == 0.5){//it's constant parameter in netmf's matrix
        MKL_INT i;
        deepwalk_filter(evals, eig_rank, window_size);
        FP para = static_cast<FP>(m) / static_cast<FP>(negative_samples);
        util<FP>::matrix_scale(evals,para);
        F = util<FP>::matrix_new(n,eig_rank);
        #pragma omp parallel shared(F) private(i)
        {
        #pragma omp for
            for (i = 0; i < ((F->nrows) * (F->ncols)); i++)
            {
                F->d[i] = 0.0;
            }
        }
        util<FP>::diag_matrix_mult(d_rt,matU,F);//F = D^(-1/2)*matU
        util<FP>::matrix_delete(matU);
        mkl_free_buffers();
        mat<FP> *Ft = util<FP>::matrix_new(eig_rank,n);
        util<FP>::matrix_build_transpose(F,Ft);
        CF = util<FP>::matrix_new(eig_rank,n);
        #pragma omp parallel shared(CF) private(i)
        {
        #pragma omp for
            for (i = 0; i < ((CF->nrows) * (CF->ncols)); i++)
            {
                CF->d[i] = 0.0;
            }
        }
        util<FP>::diag_matrix_mult(evals,Ft,CF);// CF = evals * F'
        util<FP>::matrix_delete(evals);
        util<FP>::matrix_delete(Ft);
    }
    else{
        mat<FP> *d_rt_inv_plus_alpha = util<FP>::matrix_new(n,1);
        MKL_INT i,j;
        i = 0;
        #pragma omp parallel shared(d_rt_inv_plus_alpha,n) private(i)
        {
        #pragma omp for
            for (i = 0; i < n; i++)
            {
                d_rt_inv_plus_alpha->d[i] = pow(d_rt->d[i], 2 - alpha * 2);
            }
        }
        F = util<FP>::matrix_new(n,eig_rank);
        i = 0;
        #pragma omp parallel shared(F) private(i)
        {
        #pragma omp for
            for (i = 0; i < ((F->nrows) * (F->ncols)); i++)
            {
                F->d[i] = 0.0;
            }
        }
        util<FP>::diag_matrix_mult(d_rt_inv_plus_alpha,matU,F);//F = D^(-1+alpha)*matU
        i = 0;
        #pragma omp parallel shared(d_rt_inv_plus_alpha,n) private(i)
        {
        #pragma omp for
            for (i = 0; i < n; i++)
            {
                d_rt_inv_plus_alpha->d[i] = pow(d_rt->d[i], 2 - alpha * 4);
            }
        }
        mkl_free_buffers();
        mat<FP> *res = util<FP>::matrix_new(n,eig_rank);
        i = 0;
        #pragma omp parallel shared(res) private(i)
        {
        #pragma omp for
            for (i = 0; i < ((res->nrows) * (res->ncols)); i++)
            {
                res->d[i] = 0.0;
            }
        }
        util<FP>::diag_matrix_mult(d_rt_inv_plus_alpha,matU,res);//res = D^(-1+2*alpha)*matU
        util<FP>::matrix_delete(d_rt_inv_plus_alpha);
        mat<FP> *evalsm = util<FP>::matrix_new(eig_rank,eig_rank);
        mat<FP> *K = util<FP>::matrix_new(eig_rank,eig_rank);
        mat<FP> *Ki = util<FP>::matrix_new(eig_rank,eig_rank);
        mat<FP> *Kiter = util<FP>::matrix_new(eig_rank,eig_rank);
        i = 0;
        #pragma omp parallel shared(K,Ki,evalsm,Kiter,eig_rank) private(i)
        {
        #pragma omp for
            for (i = 0; i < (eig_rank*eig_rank); i++)
            {
                K->d[i] = 0.0;
                Ki->d[i] = 0.0;
                evalsm->d[i] = 0.0;
                Kiter->d[i] = 0.0;
            }
        }
        util<FP>::matrix_transpose_matrix_mult(matU,res,Ki,1.0,0.0);
        util<FP>::matrix_diag_mult(Ki,evals,K);//K = matU'*D^(-1+2*alpha)*matU*diag(evals)
        util<FP>::matrix_delete(res);
        util<FP>::matrix_delete(matU);
        util<FP>::initialize_identity_matrix(evalsm);
        util<FP>::initialize_identity_matrix(Kiter);
        for (i = 1;i < static_cast<MKL_INT>(window_size);i++){
            util<FP>::matrix_matrix_mult(Kiter,K,Ki,1.0,0.0);
            util<FP>::matrix_copy(Ki,Kiter);
            util<FP>::matrix_matrix_add(Kiter,evalsm,1.0);
        }
        mkl_free_buffers();
        i = 0;
        j = 0;
        #pragma omp parallel shared(evalsm,evals) private(i,j)
        {
        #pragma omp for
            for (i = 0; i < (evalsm->nrows); i++)
            {
                for (j = 0; j < (evalsm->ncols); j++)
                {
                    util<FP>::matrix_set_element(evalsm, i, j, util<FP>::matrix_get_element(evals,i,0)*util<FP>::matrix_get_element(evalsm, i, j));
                }
            }
        }
        FP para = (static_cast<FP>(m) / static_cast<FP>(negative_samples)) / static_cast<FP>(window_size);
        util<FP>::matrix_scale(evalsm,para);
        CF = util<FP>::matrix_new(eig_rank,n);
        util<FP>::matrix_matrix_transpose_mult(evalsm,F,CF,1.0,0.0);// CF = evalsm*F'
        
        util<FP>::matrix_delete(K);
        util<FP>::matrix_delete(Ki);
        util<FP>::matrix_delete(Kiter);
        util<FP>::matrix_delete(evalsm);
    }
    util<FP>::matrix_delete(d_rt);
    std::cout<<"get the F and CF matrix cost time:"<<timer.elapsed()<<" s"<<std::endl;
    mat<FP> *emb = sparse_sign_randomized_single_pass_svd(F, CF, dim, s1, s2, eta1, eta2, normalize);
    //util<FP>::matrix_delete(F);
    //util<FP>::matrix_delete(CF);    

    std::cout<<"time of sparse sign randomized single pass svd process:"<<timer.elapsed()<<" s"<<std::endl;
    return emb;
}

}
