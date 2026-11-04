#include "_native.hpp"

namespace softdoe {
#ifdef SOFTDOE_NATIVE_METAL
void register_metal(py::module_& m);
#endif
#ifdef SOFTDOE_NATIVE_CUDA
void register_cuda(py::module_& m);
#endif
py::tuple optimize_cpu(Array base,Array evaluation,double beta,double delta,int p0,int p1,bool strict) {
    Matrix A=matrix(base),q=matrix(evaluation); validate(A,q,beta);
    if (A.cols<2 || A.rows>std::numeric_limits<int>::max()/(A.cols+1) ||
        p0<0 || p1<0 || p1>A.cols || (p0>1 && p1+p0-1>=A.cols) || !std::isfinite(delta))
        throw std::invalid_argument("Invalid SOFT search dimensions.");
    Linalg blas; Result result;
    {
        py::gil_scoped_release release;
        GridCache cache(A,q,delta,beta);
        CpuState state(blas,q,beta,A.rows*(A.cols+1),&cache);
        result=optimize(std::move(A),delta,p0,p1,strict,state);
    }
    return py::make_tuple(array(result.design),result.score);
}
py::object augment_cpu(Array initial,Array candidates,int count,double beta,bool return_score) {
    Matrix x=matrix(initial),points=matrix(candidates); validate(x,points,beta);
    if (count<1 || count>points.rows || x.rows>std::numeric_limits<int>::max()-count)
        throw std::invalid_argument("Invalid augmentation count.");
    Linalg blas; Matrix result(x.rows+count,x.cols); double score=0;
    {
        py::gil_scoped_release release;
        std::copy(x.data.begin(),x.data.end(),result.data.begin());
        CpuState state(blas,points,beta,x.rows+count); state.initialize(x);
        std::vector<bool> eligible(points.rows,true);
        for (int step=0;step<count;++step) {
            while (true) {
                int selected=-1; double best=std::numeric_limits<double>::infinity();
                for (int q=0;q<points.rows;++q) if (eligible[q] && state.logs[q]<best) {
                    selected=q; best=state.logs[q];
                }
                if (selected<0) throw std::runtime_error("No numerically eligible augmentation candidate remains.");
                eligible[selected]=false;
                Matrix block(1,points.cols); std::copy(points.row(selected),points.row(selected)+points.cols,block.data.begin());
                try {
                    if (step+1<count || return_score) { state.evaluate(block,0); state.accept(0); }
                    else { Block slot; state.factor_candidate(block,slot); }
                } catch (const FactorError&) { continue; }
                std::copy(block.data.begin(),block.data.end(),result.row(x.rows+step));
                break;
            }
        }
        score=state.score;
    }
    if (return_score) return py::make_tuple(array(result),score);
    return array(result);
}
Array evaluate_cpu(Array initial,Array evaluation,double beta,py::object addition,double alpha) {
    Matrix x=matrix(initial),q=matrix(evaluation); validate(x,q,beta);
    if (!(alpha>0) || !std::isfinite(alpha)) throw std::invalid_argument("alpha must be positive and finite.");
    Matrix extra; if (!addition.is_none()) extra=matrix(py::cast<Array>(addition));
    Linalg blas; Vec logs;
    {
        py::gil_scoped_release release;
        CpuState state(blas,q,beta,x.rows+extra.rows,nullptr,alpha); state.initialize(x);
        if (extra.rows) { state.evaluate(extra,0); state.accept(0); }
        logs=state.logs;
    }
    Array out(logs.size()); std::copy(logs.begin(),logs.end(),out.mutable_data()); return out;
}
} // namespace softdoe

PYBIND11_MODULE(_native,m) {
    using namespace softdoe;
    m.doc()="Reusable native Q evaluation and SOFT search.";
    m.def("optimize_cpu",&optimize_cpu,py::arg("A"),py::arg("evaluation"),py::arg("beta"),
          py::arg("delta"),py::arg("p0"),py::arg("p1"),py::arg("strict"));
    m.def("augment_cpu",&augment_cpu,py::arg("design"),py::arg("candidates"),
          py::arg("count"),py::arg("beta"),py::arg("return_score")=false);
    m.def("evaluate_cpu",&evaluate_cpu,py::arg("design"),py::arg("evaluation"),
          py::arg("beta"),py::arg("addition")=py::none(),py::arg("alpha")=1.0);
#ifdef SOFTDOE_NATIVE_METAL
    register_metal(m);
#endif
#ifdef SOFTDOE_NATIVE_CUDA
    register_cuda(m);
#endif
}
