// Runtime-compiled CUDA kernels. All arithmetic and storage are double precision.
// Evaluation columns are contiguous for coalesced accesses across threads.
__device__ double log_correlation(const double* row, const double* points,
                                  int q, int p, int m, double beta) {
    // SOFT uses alpha=1. A product avoids one logarithm per dimension;
    // retain the log sum when a high-dimensional product underflows.
    double product=1;
    for (int d=0;d<p;++d) {
        double diff=row[d]-points[d*m+q];
        product*=1.0/(1.0+diff*diff/beta);
    }
    if (product>=2.2250738585072014e-308) return log(product);
    double value=0;
    for (int d=0;d<p;++d) {
        double diff=row[d]-points[d*m+q];
        value-=log1p(diff*diff/beta);
    }
    return value;
}
extern "C" __global__ void initialize(
    double* y,double* scales,double* norms,double* logs,
    const double* points,const double* x,const double* L,
    int n,int p,int m,double beta) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=m) return;
    double scale=-1.7976931348623157e308;
    for (int r=0;r<n;++r) {
        double value=log_correlation(x+r*p,points,q,p,m,beta);
        y[r*m+q]=value; scale=fmax(scale,value);
    }
    double norm=0;
    for (int r=0;r<n;++r) {
        double value=exp(y[r*m+q]-scale);
        for (int c=0;c<r;++c) value-=L[r*n+c]*y[c*m+q];
        value/=L[r*n+r]; y[r*m+q]=value; norm+=value*value;
    }
    scales[q]=scale; norms[q]=norm;
    logs[q]=2*scale+log(fmax(norm,2.2250738585072014e-308));
}
extern "C" __global__ void evaluate_pair(
    double* logs,double* z,double* new_scales,double* new_norms,
    const double* y,const double* scales,const double* norms,
    const double* points,const double* packed,
    int n,int k,int p,int m,double beta) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=2*m) return;
    int choice=tid/m,q=tid%m;
    const double* block=packed+choice*k*p;
    const double* W=packed+2*k*p+choice*k*n;
    const double* G=packed+2*k*(p+n)+choice*k*k;
    double* candidate=z+choice*k*m;
    double scale=scales[q];
    for (int r=0;r<k;++r) {
        double value=log_correlation(block+r*p,points,q,p,m,beta);
        candidate[r*m+q]=value; scale=fmax(scale,value);
    }
    double rescale=exp(scales[q]-scale),norm=norms[q]*rescale*rescale;
    for (int r=0;r<k;++r) {
        double value=exp(candidate[r*m+q]-scale);
        for (int c=0;c<n;++c) value-=W[r*n+c]*y[c*m+q]*rescale;
        for (int c=0;c<r;++c) value-=G[r*k+c]*candidate[c*m+q];
        value/=G[r*k+r]; candidate[r*m+q]=value; norm+=value*value;
    }
    new_scales[tid]=scale; new_norms[tid]=norm;
    logs[tid]=2*scale+log(fmax(norm,2.2250738585072014e-308));
}
extern "C" __global__ void accept(
    double* y,double* scales,double* norms,const double* z,
    const double* new_scales,const double* new_norms,
    int n,int k,int m,int choice) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=m) return;
    int selected=choice*m+q;
    double rescale=exp(scales[q]-new_scales[selected]);
    for (int r=0;r<n;++r) y[r*m+q]*=rescale;
    for (int r=0;r<k;++r) y[(n+r)*m+q]=z[(choice*k+r)*m+q];
    scales[q]=new_scales[selected]; norms[q]=new_norms[selected];
}
