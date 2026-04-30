#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace axyb {
struct Matrix{int rows=0,cols=0;std::vector<double>a;Matrix()=default;Matrix(int r,int c,double v=0):rows(r),cols(c),a((size_t)r*c,v){}double&operator()(int r,int c){return a[(size_t)c*rows+r];}double operator()(int r,int c)const{return a[(size_t)c*rows+r];}static Matrix eye(int n){Matrix I(n,n);for(int i=0;i<n;++i)I(i,i)=1;return I;}};
struct CMatrix{int rows=0,cols=0;std::vector<std::complex<double>>a;CMatrix()=default;CMatrix(int r,int c):rows(r),cols(c),a((size_t)r*c){}std::complex<double>&operator()(int r,int c){return a[(size_t)c*rows+r];}std::complex<double>operator()(int r,int c)const{return a[(size_t)c*rows+r];}};
struct SparseMatrixCSC{int rows=0,cols=0;std::vector<int>col_ptr,row_idx;std::vector<double>values;SparseMatrixCSC()=default;SparseMatrixCSC(int r,int c):rows(r),cols(c),col_ptr((size_t)c+1){}static SparseMatrixCSC from_linear_indices(int rows,int cols,const std::vector<unsigned long long>&lin,const std::vector<double>&vals);void multiply(const std::vector<double>&x,std::vector<double>&y)const;void transpose_multiply(const std::vector<double>&x,std::vector<double>&y)const;double column_dot(int c0,int c1)const;};
Matrix matmul(const Matrix&A,const Matrix&B);Matrix transpose(const Matrix&A);Matrix add(const Matrix&A,const Matrix&B,double alphaB=1.0);Matrix inverse(Matrix A);Matrix solve_dense(Matrix A,Matrix B,bool prefer_cholesky=false);Matrix orthonormalize3(const Matrix&A);Matrix rod2dcm(const std::vector<double>&r);std::vector<double> flatten_col_major(const Matrix&M);Matrix skew(const std::vector<double>&g);double det3(const Matrix&A);double frobenius_squared(const Matrix&A);CMatrix eig_complex(const Matrix&A,std::vector<std::complex<double>>&evals);
}
