#include "_native.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <memory>
#include <mutex>

namespace softdoe {
namespace {
std::string error_text(NSError* error) {
    return error ? std::string([[error localizedDescription] UTF8String]) : "unknown Metal error";
}
struct Context {
    id<MTLDevice> device=nil;
    id<MTLCommandQueue> queue=nil;
    id<MTLLibrary> library=nil;
    id<MTLComputePipelineState> init=nil,pair=nil,accept=nil;
    explicit Context(const std::string& source) {
        @autoreleasepool {
            if (@available(macOS 10.13,*)) {
                if (&MTLCreateSystemDefaultDevice) device=MTLCreateSystemDefaultDevice();
            }
            if (!device) throw std::runtime_error("No usable native Metal device is available.");
            NSError* error=nil;
            queue=[device newCommandQueue];
            library=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                            options:nil error:&error];
            if (!queue || !library) { cleanup(); throw std::runtime_error("Metal setup failed: "+error_text(error)); }
            try { init=make("initialize"); pair=make("evaluate_pair"); accept=make("accept"); }
            catch (...) { cleanup(); throw; }
        }
    }
    id<MTLComputePipelineState> make(const char* name) {
        id<MTLFunction> f=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!f) throw std::runtime_error(std::string("Missing Metal kernel: ")+name);
        NSError* error=nil;
        id<MTLComputePipelineState> result=[device newComputePipelineStateWithFunction:f error:&error];
        [f release];
        if (!result) throw std::runtime_error("Metal pipeline failed: "+error_text(error));
        return result;
    }
    void cleanup() {
        [accept release]; [pair release]; [init release]; [library release]; [queue release]; [device release];
        accept=nil; pair=nil; init=nil; library=nil; queue=nil; device=nil;
    }
    ~Context() { cleanup(); }
};
std::shared_ptr<Context> context(const std::string& source) {
    static std::mutex mutex;
    static std::shared_ptr<Context> cached;
    static std::string previous;
    std::lock_guard<std::mutex> lock(mutex);
    if (!cached || previous!=source) { auto value=std::make_shared<Context>(source); cached=value; previous=source; }
    return cached;
}
struct Buffer {
    id<MTLBuffer> value=nil;
    Buffer(id<MTLDevice> device,size_t size) {
        value=[device newBufferWithLength:std::max(size,size_t(1))*sizeof(float)
                                 options:MTLResourceStorageModeShared];
        if (!value) throw std::bad_alloc();
    }
    ~Buffer() { [value release]; }
    Buffer(const Buffer&)=delete;
    float* data() { return static_cast<float*>([value contents]); }
};
inline void bind(id<MTLComputeCommandEncoder> e,const Buffer& b,int i) { [e setBuffer:b.value offset:0 atIndex:i]; }
inline void parameter(id<MTLComputeCommandEncoder> e,int value,int index) {
    uint32_t v=uint32_t(value); [e setBytes:&v length:sizeof(v) atIndex:index];
}
inline void dispatch(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> pipeline,int count) {
    [e setComputePipelineState:pipeline];
    if (@available(macOS 10.13,*)) {
        [e dispatchThreads:MTLSizeMake(count,1,1)
           threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,[pipeline maxTotalThreadsPerThreadgroup]),1,1)];
    }
}
inline void submit(id<MTLCommandBuffer> command,id<MTLComputeCommandEncoder> encoder) {
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if ([command status]==MTLCommandBufferStatusError)
        throw std::runtime_error("Metal evaluation failed: "+error_text([command error]));
}

class MetalState: public FactorState {
    std::shared_ptr<Context> ctx;
    const Matrix& points;
    int m,k,pending=-1,pending_n=0;
    Buffer q_gpu,y_gpu,scale_gpu,norm_gpu,logs_gpu,z_gpu,new_scale_gpu,new_norm_gpu;
    Buffer x_gpu,l_gpu,packed_gpu;
    Block slots[2];
    void encode_accept(id<MTLComputeCommandEncoder> e) {
        bind(e,y_gpu,0); bind(e,scale_gpu,1); bind(e,norm_gpu,2); bind(e,z_gpu,3);
        bind(e,new_scale_gpu,4); bind(e,new_norm_gpu,5);
        parameter(e,pending_n,6); parameter(e,k,7); parameter(e,m,8); parameter(e,pending,9);
        dispatch(e,ctx->accept,m); pending=-1;
    }
    double min_value(int which) {
        float* values=logs_gpu.data()+size_t(which)*m;
        double best=std::numeric_limits<double>::infinity();
        for (int j=0;j<m;++j) {
            if (!std::isfinite(values[j])) throw std::runtime_error("Nonfinite native Metal Q evaluation.");
            best=std::min(best,double(values[j]));
        }
        return best;
    }
    void upload(const Matrix& block,int index) {
        factor_candidate(block,slots[index]);
        const Block& s=slots[index]; float* data=packed_gpu.data();
        for (size_t i=0;i<block.data.size();++i) data[index*k*p+i]=float(block.data[i]);
        for (size_t i=0;i<s.W.size();++i) data[2*k*p+size_t(index)*k*n+i]=float(s.W[i]);
        float* g=data+2*k*(p+n)+index*k*k;
        for (int r=0;r<k;++r) for (int c=0;c<k;++c) g[r*k+c]=float(s.G[r+size_t(c)*k]);
    }
    void evaluate(int choices) {
        @autoreleasepool {
            id<MTLCommandBuffer> command=[ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
            if (!command || !e) throw std::runtime_error("Metal command allocation failed.");
            if (pending>=0) encode_accept(e);
            bind(e,logs_gpu,0); bind(e,z_gpu,1); bind(e,new_scale_gpu,2); bind(e,new_norm_gpu,3);
            bind(e,y_gpu,4); bind(e,scale_gpu,5); bind(e,norm_gpu,6); bind(e,q_gpu,7); bind(e,packed_gpu,8);
            parameter(e,n,9); parameter(e,k,10); parameter(e,p,11); parameter(e,m,12);
            float b=float(beta); [e setBytes:&b length:sizeof(b) atIndex:13];
            dispatch(e,ctx->pair,choices*m); submit(command,e);
        }
    }
public:
    double score=0;
    bool gpu_policy=true;
    MetalState(const Linalg& blas,const Matrix& q,double beta,int max_rows,int block_size,
               std::shared_ptr<Context> context,double alpha=1): FactorState(blas,q.cols,max_rows,beta,alpha),ctx(context),points(q),m(q.rows),k(block_size),
        q_gpu(ctx->device,size_t(q.rows)*q.cols),y_gpu(ctx->device,size_t(max_rows)*m),
        scale_gpu(ctx->device,m),norm_gpu(ctx->device,m),logs_gpu(ctx->device,2*size_t(m)),
        z_gpu(ctx->device,2*size_t(k)*m),new_scale_gpu(ctx->device,2*size_t(m)),new_norm_gpu(ctx->device,2*size_t(m)),
        x_gpu(ctx->device,size_t(max_rows)*p),l_gpu(ctx->device,size_t(max_rows)*max_rows),
        packed_gpu(ctx->device,2*size_t(k)*(p+max_rows+k)) {
        for (int d=0;d<p;++d) for (int j=0;j<m;++j) q_gpu.data()[size_t(d)*m+j]=float(q.row(j)[d]);
    }
    void initialize(const Matrix& design) {
        pending=-1; factor_initialize(design);
        for (size_t i=0;i<design.data.size();++i) x_gpu.data()[i]=float(design.data[i]);
        for (int r=0;r<n;++r) for (int c=0;c<n;++c) l_gpu.data()[size_t(r)*n+c]=float(L[r+size_t(c)*capacity]);
        @autoreleasepool {
            id<MTLCommandBuffer> command=[ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
            if (!command || !e) throw std::runtime_error("Metal command allocation failed.");
            bind(e,y_gpu,0); bind(e,scale_gpu,1); bind(e,norm_gpu,2); bind(e,logs_gpu,3);
            bind(e,q_gpu,4); bind(e,x_gpu,5); bind(e,l_gpu,6);
            parameter(e,n,7); parameter(e,p,8); parameter(e,m,9);
            float b=float(beta); [e setBytes:&b length:sizeof(b) atIndex:10];
            float a=float(alpha); [e setBytes:&a length:sizeof(a) atIndex:11];
            dispatch(e,ctx->init,m); submit(command,e);
        }
        score=min_value(0);
    }
    std::pair<double,double> evaluate_pair(const Matrix& a,const Matrix& b) {
        upload(a,0); upload(b,1); evaluate(2); return {min_value(0),min_value(1)};
    }
    void accept(int index,bool needs_state=true) {
        score=min_value(index);
        if (needs_state) { pending=index; pending_n=n; factor_accept(slots[index]); }
        else pending=-1;
    }
    int selected(const std::vector<bool>& eligible) {
        float* logs=logs_gpu.data(); int best=-1;
        for (int j=0;j<m;++j) if (eligible[j] && (best<0 || logs[j]<logs[best])) best=j;
        if (best<0) throw std::runtime_error("No numerically eligible augmentation candidate remains.");
        // Preserve existing A-SOFT augmentation semantics. This is distinct
        // from SOFT optimization, whose GPU ties never invoke CPU rescoring.
        double tolerance=std::sqrt(double(std::numeric_limits<float>::epsilon()))*std::max(1.0,std::abs(double(logs[best])));
        std::vector<int> close;
        for (int j=0;j<m;++j) if (eligible[j] && std::abs(double(logs[j])-logs[best])<=tolerance) close.push_back(j);
        if (close.size()>1) {
            Matrix subset(int(close.size()),p);
            for (size_t j=0;j<close.size();++j) std::copy(points.row(close[j]),points.row(close[j])+p,subset.row(int(j)));
            CpuState precise(blas,subset,beta,n); precise.initialize(design());
            size_t index=std::min_element(precise.logs.begin(),precise.logs.end())-precise.logs.begin();
            best=close[index];
        }
        return best;
    }
    void append_point(const Matrix& point,bool needs_state) {
        if (needs_state) { upload(point,0); evaluate(1); accept(0); }
        else { Block trial; factor_candidate(point,trial); }
    }
};

py::dict metal_info(const std::string& source) {
    py::dict result; result["compiled"]=true;
    try {
        std::shared_ptr<Context> ctx;
        { py::gil_scoped_release release; ctx=context(source); }
        result["available"]=true; result["device"]=std::string([[ctx->device name] UTF8String]); result["reason"]=py::none();
    } catch (const std::exception& error) {
        result["available"]=false; result["device"]=py::none(); result["reason"]=error.what();
    }
    return result;
}
py::tuple optimize_metal(Array base,Array evaluation,double beta,double delta,int p0,int p1,bool strict,const std::string& source) {
    Matrix A=matrix(base),q=matrix(evaluation); validate(A,q,beta);
    if (A.cols<2 || A.rows>std::numeric_limits<int>::max()/(A.cols+1) ||
        p0<0 || p1<0 || p1>A.cols || (p0>1 && p1+p0-1>=A.cols) || !std::isfinite(delta))
        throw std::invalid_argument("Invalid SOFT search dimensions.");
    Linalg blas; Result result;
    {
        py::gil_scoped_release release;
        MetalState state(blas,q,beta,A.rows*(A.cols+1),A.rows,context(source));
        result=optimize(std::move(A),delta,p0,p1,strict,state);
    }
    return py::make_tuple(array(result.design),result.score);
}
py::object augment_metal(Array initial,Array candidates,int count,double beta,const std::string& source,bool return_score) {
    Matrix x=matrix(initial),points=matrix(candidates); validate(x,points,beta);
    if (count<1 || count>points.rows || x.rows>std::numeric_limits<int>::max()-count)
        throw std::invalid_argument("Invalid augmentation count.");
    Linalg blas; Matrix result(x.rows+count,x.cols); double score=0;
    {
        py::gil_scoped_release release;
        std::copy(x.data.begin(),x.data.end(),result.data.begin());
        MetalState state(blas,points,beta,x.rows+count,1,context(source)); state.initialize(x);
        std::vector<bool> eligible(points.rows,true);
        for (int step=0;step<count;++step) {
            while (true) {
                int selected=state.selected(eligible); eligible[selected]=false;
                Matrix block(1,points.cols); std::copy(points.row(selected),points.row(selected)+points.cols,block.data.begin());
                try { state.append_point(block,step+1<count || return_score); } catch (const FactorError&) { continue; }
                std::copy(block.data.begin(),block.data.end(),result.row(x.rows+step));
                break;
            }
        }
        score=state.score;
    }
    if (return_score) return py::make_tuple(array(result),score);
    return array(result);
}
double evaluate_metal(Array design,Array evaluation,double beta,double alpha,const std::string& source) {
    Matrix x=matrix(design),q=matrix(evaluation); validate(x,q,beta);
    if (!(alpha>0) || !std::isfinite(alpha)) throw std::invalid_argument("alpha must be positive and finite.");
    // Reject parameters that cannot represent this kernel on Metal, allowing
    // the public wrapper to repeat the calculation in CPU double precision.
    if (!(float(beta)>0) || !std::isfinite(float(beta)) ||
        !(float(alpha)>0) || !std::isfinite(float(alpha)))
        throw std::runtime_error("Kernel parameters exceed Metal float32 range.");
    Linalg blas; double score;
    {
        py::gil_scoped_release release;
        MetalState state(blas,q,beta,x.rows,1,context(source),alpha);
        state.initialize(x); score=state.score;
    }
    return score;
}
} // namespace
void register_metal(py::module_& m) {
    m.def("metal_info",&metal_info);
    m.def("optimize_metal",&optimize_metal);
    m.def("augment_metal",&augment_metal,py::arg("design"),py::arg("candidates"),
          py::arg("count"),py::arg("beta"),py::arg("source"),py::arg("return_score")=false);
    m.def("evaluate_metal",&evaluate_metal);
}
} // namespace softdoe
