#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "_kernel.hpp"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace softdoe {
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Vec = std::vector<double>;

struct Matrix {
    int rows = 0, cols = 0;
    Vec data;
    Matrix() = default;
    Matrix(int n, int p): rows(n), cols(p), data(size_t(n)*p) {}
    double* row(int i) { return data.data() + size_t(i)*cols; }
    const double* row(int i) const { return data.data() + size_t(i)*cols; }
};
inline Matrix matrix(const Array& x) {
    if (x.ndim()!=2 || x.shape(0)<1 || x.shape(1)<1 ||
        x.shape(0)>std::numeric_limits<int>::max() || x.shape(1)>std::numeric_limits<int>::max())
        throw std::invalid_argument("Expected a nonempty two-dimensional matrix.");
    Matrix out(int(x.shape(0)), int(x.shape(1)));
    std::copy(x.data(),x.data()+x.size(),out.data.begin());
    for (double v: out.data) if (!std::isfinite(v)) throw std::invalid_argument("Matrix must be finite.");
    return out;
}
inline Array array(const Matrix& x) {
    Array out({x.rows,x.cols});
    std::copy(x.data.begin(),x.data.end(),out.mutable_data());
    return out;
}
struct FactorError: std::runtime_error { using std::runtime_error::runtime_error; };

// Use a consistent system BLAS on macOS, independent of which BLAS a SciPy
// wheel bundles. Other platforms use SciPy's checked Cython BLAS/LAPACK ABI.
// Neither path calls Python in the search loop.
#ifdef __APPLE__
// Retain the LP64 API so source builds also support macOS before 13.3.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
class Linalg {
#ifndef __APPLE__
    void *trsm_, *gemm_, *syrk_, *potrf_;
    static void* capsule(const char* module, const char* function, bool& wide) {
        py::object cap=py::module_::import(module).attr("__pyx_capi__")[py::str(function)];
        const char* name=PyCapsule_GetName(cap.ptr());
        if (!name) throw py::error_already_set();
        std::string signature(name);
        wide=signature.find("int64_t *")!=std::string::npos;
        if (!wide && signature.find("int *")==std::string::npos)
            throw std::runtime_error("Unsupported SciPy BLAS integer ABI: "+signature);
        void* ptr=PyCapsule_GetPointer(cap.ptr(),name);
        if (!ptr) throw py::error_already_set();
        return ptr;
    }
#endif
    bool wide_=false;
    template<class I> void chol_impl(double* a,int n,int ld) const {
        // Use LAPACK's fixed recursive panel algorithm for small matrices.
        // Vendor POTRF panels can round differently even with identical input,
        // changing strict SOFT decisions at ties. Larger factorizations retain
        // the vendor's blocked implementation.
        if (n<=64) { chol_panel<I>(a,n,ld); return; }
#ifdef __APPLE__
        char lower='L'; __CLPK_integer size=n,stride=ld,info=0;
        dpotrf_(&lower,&size,a,&stride,&info);
#else
        using Fn=void(*)(char*,I*,double*,I*,I*);
        char lower='L'; I size=n,stride=ld,info=0;
        reinterpret_cast<Fn>(potrf_)(&lower,&size,a,&stride,&info);
#endif
        if (info) throw FactorError("Correlation matrix is not positive definite.");
    }
    template<class I> void chol_panel(double* a,int n,int ld) const {
        if (n==1) {
            if (!(a[0]>0) || !std::isfinite(a[0]))
                throw FactorError("Correlation matrix is not positive definite.");
            a[0]=std::sqrt(a[0]); return;
        }
        const int first=n/2,second=n-first;
        chol_panel<I>(a,first,ld);
        double* lower=a+first;
        double* diagonal=lower+size_t(first)*ld;
#ifdef __APPLE__
        cblas_dtrsm(CblasColMajor,CblasRight,CblasLower,CblasTrans,CblasNonUnit,
                    second,first,1,a,ld,lower,ld);
        cblas_dsyrk(CblasColMajor,CblasLower,CblasNoTrans,second,first,
                    -1,lower,ld,1,diagonal,ld);
#else
        using Solve=void(*)(char*,char*,char*,char*,I*,I*,double*,double*,I*,double*,I*);
        char side='R',uplo='L',trans='T',diag='N';
        I rows=second,columns=first,stride=ld;
        double one=1,minus_one=-1;
        reinterpret_cast<Solve>(trsm_)(&side,&uplo,&trans,&diag,&rows,&columns,&one,
                                       a,&stride,lower,&stride);
        using Update=void(*)(char*,char*,I*,I*,double*,double*,I*,double*,double*,I*);
        trans='N';
        reinterpret_cast<Update>(syrk_)(&uplo,&trans,&rows,&columns,&minus_one,
                                       lower,&stride,&one,diagonal,&stride);
#endif
        chol_panel<I>(diagonal,second,ld);
    }
    template<class I> void solve_impl(const double* a,int ld,double* b,int ldb,int n,int m) const {
#ifdef __APPLE__
        cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,
                    n,m,1,a,ld,b,ldb);
#else
        using Fn=void(*)(char*,char*,char*,char*,I*,I*,double*,double*,I*,double*,I*);
        char side='L',lower='L',trans='N',diag='N';
        I nr=n,nc=m,la=ld,lb=ldb; double one=1;
        reinterpret_cast<Fn>(trsm_)(&side,&lower,&trans,&diag,&nr,&nc,&one,
                                    const_cast<double*>(a),&la,b,&lb);
#endif
    }
    template<class I> void cross_impl(const double* a,int lda,const double* b,int ldb,
                                     double* c,int ldc,int n,int m,int k,double alpha,double beta) const {
#ifdef __APPLE__
        cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,m,k,alpha,a,lda,b,ldb,beta,c,ldc);
#else
        using Fn=void(*)(char*,char*,I*,I*,I*,double*,double*,I*,double*,I*,double*,double*,I*);
        char ta='T',tb='N'; I nr=n,nc=m,inner=k,la=lda,lb=ldb,lc=ldc;
        reinterpret_cast<Fn>(gemm_)(&ta,&tb,&nr,&nc,&inner,&alpha,const_cast<double*>(a),
                                   &la,const_cast<double*>(b),&lb,&beta,c,&lc);
#endif
    }
public:
    Linalg() {
#ifndef __APPLE__
        bool b,c,d;
        trsm_=capsule("scipy.linalg.cython_blas","dtrsm",wide_);
        gemm_=capsule("scipy.linalg.cython_blas","dgemm",b);
        potrf_=capsule("scipy.linalg.cython_lapack","dpotrf",c);
        syrk_=capsule("scipy.linalg.cython_blas","dsyrk",d);
        if (b!=wide_ || c!=wide_ || d!=wide_)
            throw std::runtime_error("Inconsistent SciPy BLAS/LAPACK integer ABI.");
#endif
    }
    void chol(double* a,int n,int ld) const {
        if (wide_) chol_impl<int64_t>(a,n,ld); else chol_impl<int>(a,n,ld);
    }
    void solve(const double* a,int ld,double* b,int ldb,int n,int m) const {
        if (wide_) solve_impl<int64_t>(a,ld,b,ldb,n,m); else solve_impl<int>(a,ld,b,ldb,n,m);
    }
    void cross(const double* a,int lda,const double* b,int ldb,double* c,int ldc,
               int n,int m,int k,double alpha=1,double beta=0) const {
        if (wide_) cross_impl<int64_t>(a,lda,b,ldb,c,ldc,n,m,k,alpha,beta);
        else cross_impl<int>(a,lda,b,ldb,c,ldc,n,m,k,alpha,beta);
    }
};
#ifdef __APPLE__
#pragma clang diagnostic pop
#endif

inline double correlation(const double* x,const double* y,int p,double beta,double alpha=1) {
    return kernel_correlation(x,y,p,alpha,beta);
}
inline double log_correlation(const double* x,const double* y,int p,double beta,double alpha=1) {
    return kernel_log_correlation(x,y,p,alpha,beta);
}
inline bool close_gpu(double a,double b) {
    return std::abs(a-b)<=std::sqrt(double(std::numeric_limits<float>::epsilon()))*
                             std::max({1.0,std::abs(a),std::abs(b)});
}

// All SOFT swaps permute a fixed set of coordinates. Cache each one-dimensional
// factor against the QMC points once, retaining the original product order.
class GridCache {
    std::vector<Vec> levels_, tables_;
    int p_=0,m_=0;
public:
    GridCache()=default;
    GridCache(const Matrix& A,const Matrix& points,double delta,double beta):p_(A.cols),m_(points.rows) {
        if (size_t(A.rows)*2*A.cols*points.rows*sizeof(double)>64*1024*1024) return;
        levels_.resize(p_); tables_.resize(p_);
        for (int d=0;d<p_;++d) {
            auto& values=levels_[d];
            for (int i=0;i<A.rows;++i) {
                double v=A.row(i)[d]; values.push_back(v); values.push_back(v<0.5?v+delta:v-delta);
            }
            std::sort(values.begin(),values.end());
            values.erase(std::unique(values.begin(),values.end()),values.end());
            auto& table=tables_[d]; table.resize(values.size()*m_);
            for (size_t j=0;j<values.size();++j) for (int q=0;q<m_;++q) {
                double z=values[j]-points.row(q)[d]; table[j*m_+q]=1.0/(1.0+z*z*(1.0/beta));
            }
        }
    }
    bool fill(const double* row,double* out) const {
        if (levels_.empty()) return false;
        std::fill(out,out+m_,1.0);
        for (int d=0;d<p_;++d) {
            auto it=std::lower_bound(levels_[d].begin(),levels_[d].end(),row[d]);
            if (it==levels_[d].end() || *it!=row[d]) return false;
            const double* values=tables_[d].data()+size_t(it-levels_[d].begin())*m_;
            for (int q=0;q<m_;++q) out[q]*=values[q];
        }
        return true;
    }
};

struct Block {
    Matrix x;
    Vec W,G,z,scales,norms,logs;
    double score=0;
};

class FactorState {
protected:
    const Linalg& blas;
public:
    int n=0,p,capacity;
    double beta,alpha;
    Matrix x;
    Vec L;
    FactorState(const Linalg& b,int dimensions,int max_rows,double kernel_beta,double kernel_alpha=1):
        blas(b),p(dimensions),capacity(max_rows),beta(kernel_beta),alpha(kernel_alpha),x(max_rows,dimensions),L(size_t(max_rows)*max_rows) {}
    virtual ~FactorState()=default;
    void factor_initialize(const Matrix& design) {
        n=design.rows;
        if (n>capacity || design.cols!=p) throw std::invalid_argument("Incompatible design shape.");
        std::copy(design.data.begin(),design.data.end(),x.data.begin());
        std::fill(L.begin(),L.end(),0);
        for (int j=0;j<n;++j) for (int i=j;i<n;++i)
            L[i+size_t(j)*capacity]=i==j?1.0+1e-6:correlation(x.row(i),x.row(j),p,beta,alpha);
        blas.chol(L.data(),n,capacity);
    }
    void factor_candidate(const Matrix& block,Block& slot) {
        int k=block.rows;
        if (block.cols!=p || n+k>capacity) throw std::invalid_argument("Incompatible append shape.");
        slot.x=block; slot.W.resize(size_t(n)*k); slot.G.resize(size_t(k)*k);
        for (int j=0;j<k;++j) for (int i=0;i<n;++i)
            slot.W[i+size_t(j)*n]=correlation(x.row(i),block.row(j),p,beta,alpha);
        blas.solve(L.data(),capacity,slot.W.data(),n,n,k);
        for (int j=0;j<k;++j) for (int i=0;i<k;++i)
            slot.G[i+size_t(j)*k]=i==j?1.0+1e-6:correlation(block.row(i),block.row(j),p,beta,alpha);
        blas.cross(slot.W.data(),n,slot.W.data(),n,slot.G.data(),k,k,k,n,-1,1);
        blas.chol(slot.G.data(),k,k);
    }
    void factor_accept(const Block& slot) {
        int k=slot.x.rows;
        for (int r=0;r<k;++r) {
            std::copy(slot.x.row(r),slot.x.row(r)+p,x.row(n+r));
            for (int c=0;c<n;++c) L[n+r+size_t(c)*capacity]=slot.W[c+size_t(r)*n];
            for (int c=0;c<=r;++c) L[n+r+size_t(n+c)*capacity]=slot.G[r+size_t(c)*k];
        }
        n+=k;
    }
    Matrix design() const {
        Matrix out(n,p); std::copy(x.data.begin(),x.data.begin()+size_t(n)*p,out.data.begin()); return out;
    }
};

class CpuState: public FactorState {
    const Matrix& points;
    const GridCache* cache;
    Vec rhs, scaled;
    void log_rhs(const Matrix& design) {
        rhs.resize(size_t(design.rows)*m);
        for (int r=0;r<design.rows;++r) {
            double* row=rhs.data()+size_t(r)*m;
            bool cached=alpha==1 && cache && cache->fill(design.row(r),row);
            for (int q=0;q<m;++q) {
                if (cached && row[q]>0) row[q]=std::log(row[q]);
                else row[q]=log_correlation(design.row(r),points.row(q),p,beta,alpha);
            }
        }
    }
    static double norm(const double* v,int count) {
        // MOFAT/Armadillo uses two accumulation chains for small squared
        // norms. Preserve that ordering rather than a single running sum.
        double even=0,odd=0;
        int i=0;
        for (;i+1<count;i+=2) { even+=v[i]*v[i]; odd+=v[i+1]*v[i+1]; }
        if (i<count) even+=v[i]*v[i];
        return even+odd;
    }
public:
    int m;
    Vec y,scales,norms,logs;
    Block slots[2];
    double score=0;
    bool gpu_policy=false;
    CpuState(const Linalg& b,const Matrix& q,double kernel_beta,int max_rows,const GridCache* grid=nullptr,double kernel_alpha=1):
        FactorState(b,q.cols,max_rows,kernel_beta,kernel_alpha),points(q),cache(grid),
        scaled(size_t(max_rows)*q.rows),m(q.rows),y(size_t(max_rows)*q.rows),
        scales(q.rows),norms(q.rows),logs(q.rows) {}
    void initialize(const Matrix& design) {
        factor_initialize(design);
        if (alpha==1) {
            rhs.resize(size_t(n)*m);
            for (int r=0;r<n;++r) {
                double* row=rhs.data()+size_t(r)*m;
                if (cache && cache->fill(design.row(r),row)) continue;
                for (int q=0;q<m;++q)
                    row[q]=correlation(design.row(r),points.row(q),p,beta);
            }
        } else {
            log_rhs(design);
        }
        for (int q=0;q<m;++q) {
            bool direct=alpha==1;
            if (direct) for (int r=0;r<n;++r) {
                double value=rhs[size_t(r)*m+q];
                if (value<std::numeric_limits<double>::min() || !std::isfinite(value)) {
                    direct=false; break;
                }
            }
            // Preserve MOFAT's direct correlation/maximum scaling whenever
            // products are representable. A log/exp round trip adds rounding
            // that can change a strict search decision near a tie.
            // Recompute the entire column in log space after underflow.
            if (alpha==1 && !direct) for (int r=0;r<n;++r)
                rhs[size_t(r)*m+q]=log_correlation(design.row(r),points.row(q),p,beta);
            double scale=-std::numeric_limits<double>::infinity();
            for (int r=0;r<n;++r) scale=std::max(scale,rhs[size_t(r)*m+q]);
            scales[q]=direct?std::log(scale):scale;
            for (int r=0;r<n;++r)
                y[r+size_t(q)*capacity]=direct?rhs[size_t(r)*m+q]/scale:
                    std::exp(rhs[size_t(r)*m+q]-(std::isfinite(scale)?scale:0));
        }
        blas.solve(L.data(),capacity,y.data(),capacity,n,m);
        score=std::numeric_limits<double>::infinity();
        for (int q=0;q<m;++q) {
            norms[q]=norm(y.data()+size_t(q)*capacity,n);
            logs[q]=2*scales[q]+std::log(std::max(norms[q],std::numeric_limits<double>::min()));
            if (std::isnan(logs[q]) || logs[q]==std::numeric_limits<double>::infinity())
                throw std::runtime_error("Nonfinite Q evaluation.");
            score=std::min(score,logs[q]);
        }
    }
    double evaluate(const Matrix& block,int index) {
        Block& slot=slots[index]; int k=block.rows;
        factor_candidate(block,slot); log_rhs(block);
        slot.scales.resize(m); slot.norms.resize(m); slot.logs.resize(m); slot.z.resize(size_t(k)*m);
        for (int q=0;q<m;++q) {
            double scale=scales[q];
            for (int r=0;r<k;++r) scale=std::max(scale,rhs[size_t(r)*m+q]);
            slot.scales[q]=scale;
            double f=std::exp(scales[q]-(std::isfinite(scale)?scale:0));
            for (int r=0;r<n;++r) scaled[r+size_t(q)*capacity]=y[r+size_t(q)*capacity]*f;
        }
        blas.cross(slot.W.data(),n,scaled.data(),capacity,slot.z.data(),k,k,m,n);
        for (int q=0;q<m;++q) for (int r=0;r<k;++r)
            slot.z[r+size_t(q)*k]=std::exp(rhs[size_t(r)*m+q]-(std::isfinite(slot.scales[q])?slot.scales[q]:0))-slot.z[r+size_t(q)*k];
        blas.solve(slot.G.data(),k,slot.z.data(),k,k,m);
        slot.score=std::numeric_limits<double>::infinity();
        for (int q=0;q<m;++q) {
            double f=std::exp(scales[q]-(std::isfinite(slot.scales[q])?slot.scales[q]:0));
            slot.norms[q]=norms[q]*(f*f)+norm(slot.z.data()+size_t(q)*k,k);
            slot.logs[q]=2*slot.scales[q]+std::log(std::max(slot.norms[q],std::numeric_limits<double>::min()));
            if (std::isnan(slot.logs[q]) || slot.logs[q]==std::numeric_limits<double>::infinity())
                throw std::runtime_error("Nonfinite incremental Q evaluation.");
            slot.score=std::min(slot.score,slot.logs[q]);
        }
        return slot.score;
    }
    std::pair<double,double> evaluate_pair(const Matrix& a,const Matrix& b) {
        double left=evaluate(a,0),right=evaluate(b,1); return {left,right};
    }
    void accept(int index,bool needs_state=true) {
        Block& slot=slots[index]; score=slot.score;
        if (!needs_state) return;
        int k=slot.x.rows;
        for (int q=0;q<m;++q) {
            double f=std::exp(scales[q]-(std::isfinite(slot.scales[q])?slot.scales[q]:0));
            if (f!=1) for (int r=0;r<n;++r) y[r+size_t(q)*capacity]*=f;
            for (int r=0;r<k;++r) y[n+r+size_t(q)*capacity]=slot.z[r+size_t(q)*k];
        }
        scales=slot.scales; norms=slot.norms; logs=slot.logs;
        factor_accept(slot);
    }
};

struct Result { Matrix design; double score; };
inline Matrix shift(Matrix block,int column,double delta) {
    for (int i=0;i<block.rows;++i) {
        double& v=block.row(i)[column]; v=v<0.5?v+delta:v-delta;
    }
    return block;
}
inline Matrix bind_blocks(const std::vector<Matrix>& blocks) {
    Matrix out(blocks[0].rows*int(blocks.size()),blocks[0].cols);
    double* dest=out.data.data();
    for (const auto& block: blocks) dest=std::copy(block.data.begin(),block.data.end(),dest);
    return out;
}
template<class State> Result construct(const Matrix& A,double delta,bool strict,State& state) {
    std::vector<Matrix> blocks; blocks.reserve(A.cols+1); blocks.push_back(A);
    if (!strict) {
        for (int d=0;d<A.cols;++d) blocks.push_back(shift(A,d,delta));
        Matrix design=bind_blocks(blocks); state.initialize(design); return {std::move(design),state.score};
    }
    blocks.push_back(shift(A,0,delta)); state.initialize(bind_blocks(blocks));
    double score=state.score;
    for (int d=1;d<A.cols;++d) {
        Matrix left=shift(blocks.front(),d,delta),right=shift(blocks.back(),d,delta);
        auto values=state.evaluate_pair(left,right);
        int index=(values.first>=values.second || (state.gpu_policy && close_gpu(values.first,values.second)))?0:1;
        score=index==0?values.first:values.second;
        state.accept(index,d+1<A.cols);
        if (index==0) blocks.insert(blocks.begin(),std::move(left)); else blocks.push_back(std::move(right));
    }
    return {bind_blocks(blocks),score};
}
template<class State> Result optimize(Matrix A,double delta,int p0,int p1,bool strict,State& state) {
    Result best=construct(A,delta,strict,state);
    for (int i=0;i<A.rows-1;++i) for (int j=i+1;j<A.rows;++j) for (int k=1;k<p0;++k) {
        Matrix candidate=A; std::swap(candidate.row(i)[p1+k],candidate.row(j)[p1+k]);
        Result next=construct(candidate,delta,strict,state);
        if (next.score>best.score && !(state.gpu_policy && close_gpu(next.score,best.score))) {
            A=std::move(candidate); best=std::move(next);
        }
    }
    return best;
}
inline void validate(const Matrix& x,const Matrix& points,double beta) {
    if (x.cols!=points.cols || !(beta>0) || !std::isfinite(beta))
        throw std::invalid_argument("Incompatible Q inputs or invalid beta.");
}
} // namespace softdoe
