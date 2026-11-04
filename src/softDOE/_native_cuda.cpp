// CUDA driver/NVRTC are loaded at runtime; CPU builds need no CUDA toolkit.
#include "_native.hpp"
#include <dlfcn.h>
#include <memory>
#include <mutex>

namespace softdoe {
namespace {
using Handle=void*;
using DevicePtr=unsigned long long;
class Library {
    Handle handle;
public:
    explicit Library(const std::string& path):handle(dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL)) {
        if (!handle) { const char* error=dlerror(); throw std::runtime_error("Cannot load "+path+": "+(error?error:"unknown error")); }
    }
    ~Library() { dlclose(handle); }
    Library(const Library&)=delete;
    template<class T> T symbol(const char* name) {
        void* result=dlsym(handle,name);
        if (!result) throw std::runtime_error(std::string("Missing CUDA symbol: ")+name);
        return reinterpret_cast<T>(result);
    }
};
struct Driver {
    Library lib{"libcuda.so.1"};
    int (*init)(unsigned)=lib.symbol<decltype(init)>("cuInit");
    int (*device_get)(int*,int)=lib.symbol<decltype(device_get)>("cuDeviceGet");
    int (*retain)(Handle*,int)=lib.symbol<decltype(retain)>("cuDevicePrimaryCtxRetain");
    int (*release)(int)=lib.symbol<decltype(release)>("cuDevicePrimaryCtxRelease_v2");
    int (*get_current)(Handle*)=lib.symbol<decltype(get_current)>("cuCtxGetCurrent");
    int (*set_current)(Handle)=lib.symbol<decltype(set_current)>("cuCtxSetCurrent");
    int (*module_load)(Handle*,const void*,unsigned,int*,void**)=lib.symbol<decltype(module_load)>("cuModuleLoadDataEx");
    int (*module_unload)(Handle)=lib.symbol<decltype(module_unload)>("cuModuleUnload");
    int (*function)(Handle*,Handle,const char*)=lib.symbol<decltype(function)>("cuModuleGetFunction");
    int (*alloc)(DevicePtr*,size_t)=lib.symbol<decltype(alloc)>("cuMemAlloc_v2");
    int (*free)(DevicePtr)=lib.symbol<decltype(free)>("cuMemFree_v2");
    int (*upload)(DevicePtr,const void*,size_t)=lib.symbol<decltype(upload)>("cuMemcpyHtoD_v2");
    int (*download)(void*,DevicePtr,size_t)=lib.symbol<decltype(download)>("cuMemcpyDtoH_v2");
    int (*launch)(Handle,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,Handle,void**,void**)=lib.symbol<decltype(launch)>("cuLaunchKernel");
    int (*error_string)(int,const char**)=lib.symbol<decltype(error_string)>("cuGetErrorString");
    void check(int status) const {
        if (!status) return;
        const char* message=nullptr; error_string(status,&message);
        throw std::runtime_error(std::string("CUDA: ")+(message?message:std::to_string(status)));
    }
};
std::shared_ptr<Driver> driver() {
    static auto value=std::make_shared<Driver>();
    return value;
}
struct ContextScope {
    Driver& d;
    Handle previous=nullptr;
    ContextScope(Driver& api,Handle context):d(api) { d.check(d.get_current(&previous)); d.check(d.set_current(context)); }
    ~ContextScope() { d.set_current(previous); }
};
std::vector<char> compile(const std::string& source,const std::string& path,int architecture) {
    Library lib(path);
    auto create=lib.symbol<int(*)(Handle*,const char*,const char*,int,const char*const*,const char*const*)>("nvrtcCreateProgram");
    auto run=lib.symbol<int(*)(Handle,int,const char*const*)>("nvrtcCompileProgram");
    auto size=lib.symbol<int(*)(Handle,size_t*)>("nvrtcGetPTXSize");
    auto get=lib.symbol<int(*)(Handle,char*)>("nvrtcGetPTX");
    auto log_size=lib.symbol<int(*)(Handle,size_t*)>("nvrtcGetProgramLogSize");
    auto log_get=lib.symbol<int(*)(Handle,char*)>("nvrtcGetProgramLog");
    auto destroy=lib.symbol<int(*)(Handle*)>("nvrtcDestroyProgram");
    auto arch_count=lib.symbol<int(*)(int*)>("nvrtcGetNumSupportedArchs");
    auto arch_list=lib.symbol<int(*)(int*)>("nvrtcGetSupportedArchs");
    int count=0;
    if (arch_count(&count) || count<=0) throw std::runtime_error("Cannot read NVRTC architectures.");
    std::vector<int> supported(count);
    if (arch_list(supported.data())) throw std::runtime_error("Cannot read NVRTC architectures.");
    int target=0;
    for (int a:supported) if (a<=architecture) target=std::max(a,target);
    if (!target) throw std::runtime_error("NVRTC does not support this GPU architecture.");
    Handle program=nullptr;
    if (create(&program,source.c_str(),"softdoe_strict.cu",0,nullptr,nullptr))
        throw std::runtime_error("Cannot create NVRTC program.");
    try {
        std::string arch="--gpu-architecture=compute_"+std::to_string(target);
        const char* options[]={"--std=c++11","--fmad=false",arch.c_str()};
        int status=run(program,3,options);
        if (status) {
            size_t length=0; log_size(program,&length);
            std::vector<char> log(std::max(length,size_t(1)),0); log_get(program,log.data());
            throw std::runtime_error("CUDA kernel compilation failed: "+std::string(log.data()));
        }
        size_t length=0;
        if (size(program,&length)) throw std::runtime_error("Cannot read compiled CUDA kernel size.");
        std::vector<char> ptx(length);
        if (get(program,ptx.data())) throw std::runtime_error("Cannot read compiled CUDA kernels.");
        destroy(&program); return ptx;
    } catch (...) { destroy(&program); throw; }
}
struct Context {
    // Keep the driver loaded until this context's destructor has released it,
    // independently of the order in which function statics are destroyed.
    std::shared_ptr<Driver> api=driver();
    Driver& d=*api;
    Handle primary=nullptr,module=nullptr,init=nullptr,pair=nullptr,accept=nullptr;
    int device=0;
    Context(const std::string& source,const std::string& nvrtc,int ordinal,int arch) {
        d.check(d.init(0)); d.check(d.device_get(&device,ordinal)); d.check(d.retain(&primary,device));
        try {
            ContextScope scope(d,primary);
            auto ptx=compile(source,nvrtc,arch);
            d.check(d.module_load(&module,ptx.data(),0,nullptr,nullptr));
            d.check(d.function(&init,module,"initialize"));
            d.check(d.function(&pair,module,"evaluate_pair"));
            d.check(d.function(&accept,module,"accept"));
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        if (!primary) return;
        Handle old=nullptr; d.get_current(&old); d.set_current(primary);
        if (module) d.module_unload(module);
        d.set_current(old); d.release(device); primary=nullptr; module=nullptr;
    }
    ~Context() { cleanup(); }
    void launch(Handle fn,int count,void** args) {
        d.check(d.launch(fn,(unsigned(count)+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));
    }
};
std::shared_ptr<Context> context(const std::string& source,const std::string& nvrtc,int ordinal,int arch) {
    static std::mutex mutex;
    static std::shared_ptr<Context> cached;
    static std::string key;
    std::string next=std::to_string(ordinal)+":"+std::to_string(arch)+":"+nvrtc+":"+source;
    std::lock_guard<std::mutex> lock(mutex);
    if (!cached || key!=next) { auto value=std::make_shared<Context>(source,nvrtc,ordinal,arch); cached=value; key=next; }
    return cached;
}
struct Buffer {
    Driver& d;
    DevicePtr ptr=0;
    Buffer(Driver& api,size_t size):d(api) { d.check(d.alloc(&ptr,std::max(size,size_t(1))*sizeof(double))); }
    ~Buffer() { if (ptr) d.free(ptr); }
    Buffer(const Buffer&)=delete;
    void upload(const Vec& data) { d.check(d.upload(ptr,data.data(),data.size()*sizeof(double))); }
    void download(Vec& data) { d.check(d.download(data.data(),ptr,data.size()*sizeof(double))); }
};
class CudaState:public FactorState {
    std::shared_ptr<Context> ctx;
    const Matrix& points;
    int m,k,pending=-1,pending_n=0;
    Buffer q_gpu,y_gpu,scale_gpu,norm_gpu,logs_gpu,z_gpu,new_scale_gpu,new_norm_gpu;
    Buffer x_gpu,l_gpu,packed_gpu;
    Block slots[2];
    Vec packed,logs;
    void flush() {
        if (pending<0) return;
        void* args[]={&y_gpu.ptr,&scale_gpu.ptr,&norm_gpu.ptr,&z_gpu.ptr,&new_scale_gpu.ptr,&new_norm_gpu.ptr,
                      &pending_n,&k,&m,&pending};
        ctx->launch(ctx->accept,m,args); pending=-1;
    }
    double min_value(int which) {
        double best=std::numeric_limits<double>::infinity();
        for (int j=0;j<m;++j) {
            double value=logs[size_t(which)*m+j];
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite native CUDA Q evaluation.");
            best=std::min(best,value);
        }
        return best;
    }
    void pack(const Matrix& block,int index) {
        factor_candidate(block,slots[index]);
        const Block& s=slots[index];
        std::copy(block.data.begin(),block.data.end(),packed.begin()+index*k*p);
        std::copy(s.W.begin(),s.W.end(),packed.begin()+2*k*p+size_t(index)*k*n);
        double* g=packed.data()+2*k*(p+n)+index*k*k;
        for (int r=0;r<k;++r) for (int c=0;c<k;++c) g[r*k+c]=s.G[r+size_t(c)*k];
    }
public:
    double score=0;
    bool gpu_policy=true;
    CudaState(const Linalg& blas,const Matrix& q,double beta,int max_rows,int block_size,
              std::shared_ptr<Context> context):FactorState(blas,q.cols,max_rows,beta),ctx(context),points(q),m(q.rows),k(block_size),
        q_gpu(ctx->d,size_t(m)*p),y_gpu(ctx->d,size_t(max_rows)*m),scale_gpu(ctx->d,m),norm_gpu(ctx->d,m),logs_gpu(ctx->d,2*size_t(m)),
        z_gpu(ctx->d,2*size_t(k)*m),new_scale_gpu(ctx->d,2*size_t(m)),new_norm_gpu(ctx->d,2*size_t(m)),
        x_gpu(ctx->d,size_t(max_rows)*p),l_gpu(ctx->d,size_t(max_rows)*max_rows),packed_gpu(ctx->d,2*size_t(k)*(p+max_rows+k)),
        logs(2*size_t(m)) {
        Vec transposed(size_t(p)*m);
        for (int d=0;d<p;++d) for (int j=0;j<m;++j) transposed[size_t(d)*m+j]=q.row(j)[d];
        q_gpu.upload(transposed);
    }
    void initialize(const Matrix& design) {
        pending=-1; factor_initialize(design); x_gpu.upload(design.data);
        Vec factor(size_t(n)*n);
        for (int r=0;r<n;++r) for (int c=0;c<n;++c) factor[size_t(r)*n+c]=L[r+size_t(c)*capacity];
        l_gpu.upload(factor);
        void* args[]={&y_gpu.ptr,&scale_gpu.ptr,&norm_gpu.ptr,&logs_gpu.ptr,&q_gpu.ptr,&x_gpu.ptr,&l_gpu.ptr,&n,&p,&m,&beta};
        ctx->launch(ctx->init,m,args);
        // Only the first m entries have been initialized.
        ctx->d.check(ctx->d.download(logs.data(),logs_gpu.ptr,size_t(m)*sizeof(double)));
        score=min_value(0);
    }
    std::pair<double,double> evaluate_pair(const Matrix& a,const Matrix& b) {
        packed.resize(2*size_t(k)*(p+n+k)); pack(a,0); pack(b,1);
        flush(); packed_gpu.upload(packed);
        void* args[]={&logs_gpu.ptr,&z_gpu.ptr,&new_scale_gpu.ptr,&new_norm_gpu.ptr,&y_gpu.ptr,&scale_gpu.ptr,&norm_gpu.ptr,
                      &q_gpu.ptr,&packed_gpu.ptr,&n,&k,&p,&m,&beta};
        ctx->launch(ctx->pair,2*m,args); logs_gpu.download(logs);
        return {min_value(0),min_value(1)};
    }
    void accept(int index,bool needs_state=true) {
        score=min_value(index);
        if (needs_state) { pending=index; pending_n=n; factor_accept(slots[index]); }
        else pending=-1;
    }
};
py::tuple optimize_cuda(Array base,Array evaluation,double beta,double delta,int p0,int p1,
                        const std::string& source,const std::string& nvrtc,int ordinal,int architecture) {
    Matrix A=matrix(base),q=matrix(evaluation); validate(A,q,beta);
    if (A.cols<2 || A.rows>std::numeric_limits<int>::max()/(A.cols+1) || p0<0 || p1<0 ||
        p1>A.cols || (p0>1 && p1+p0-1>=A.cols) || !std::isfinite(delta) || ordinal<0)
        throw std::invalid_argument("Invalid CUDA SOFT search parameters.");
    Linalg blas; Result result;
    {
        py::gil_scoped_release release;
        auto ctx=context(source,nvrtc,ordinal,architecture);
        ContextScope scope(ctx->d,ctx->primary);
        CudaState state(blas,q,beta,A.rows*(A.cols+1),A.rows,ctx);
        result=optimize(std::move(A),delta,p0,p1,true,state);
    }
    return py::make_tuple(array(result.design),result.score);
}
} // namespace
void register_cuda(py::module_& m) { m.def("optimize_cuda",&optimize_cuda); }
} // namespace softdoe
