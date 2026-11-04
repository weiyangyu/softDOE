"""Time public SOFT/A-SOFT workflows after runtime and PyDOE warmup.

Run sequentially in each environment being compared. No R source is changed.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import softDOE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', nargs='+', choices=['cpu', 'metal', 'cuda'], default=['cpu', 'metal'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--blas-threads', type=int,
                        help='Optional BLAS thread limit; requires threadpoolctl')
    parser.add_argument('--archive', type=Path, help='Record the installed archive SHA-256')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.blas_threads is not None and args.blas_threads < 1:
        parser.error('--blas-threads must be positive')
    torch = None
    if 'cuda' in args.backend:
        import torch
        if args.blas_threads is not None:
            torch.set_num_threads(args.blas_threads)
    context = nullcontext()
    if args.blas_threads is not None:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=args.blas_threads)

    def synchronize():
        if torch is not None:
            torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    capabilities = softDOE.backend_info()
    for backend in args.backend:
        if not capabilities[backend]['available']:
            parser.error(f'{backend} unavailable: {capabilities[backend]["reason"]}')
        # Exercise PyDOE as well as shader/runtime initialization.
        softDOE.soft(4,4,m=64,seed=123,backend=backend)
    report = dict(version=softDOE.__version__, python=platform.python_version(),
                  platform=platform.platform(), blas_threads=args.blas_threads,
                  packages={name: importlib.metadata.version(name) for name in ['numpy','scipy','pydoe']},
                  backends=capabilities, cases=[])
    if torch is not None:
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      device=torch.cuda.get_device_name())
    if args.archive is not None:
        report['archive_sha256'] = hashlib.sha256(args.archive.read_bytes()).hexdigest()

    def measure(key, fun, kwargs):
        times = []
        # Warm the exact workload, including strict runtime kernel compilation.
        warm = fun(**kwargs)
        if warm['backend_used'] != kwargs['backend']:
            raise RuntimeError(warm.get('fallback_reason', 'Unexpected CPU fallback'))
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter()
            result = fun(**kwargs)
            synchronize()
            times.append(time.perf_counter()-start)
            if result['backend_used'] != kwargs['backend']:
                raise RuntimeError(result.get('fallback_reason', 'Unexpected CPU fallback'))
        if 'design' in result:
            np.testing.assert_array_equal(result['design'], warm['design'])
            np.save(args.output/(key+'.npy'), result['design'])
        report['cases'].append(dict(case=key, seconds=times, median_seconds=float(np.median(times)),
                                   Q=result['Q'], normalized_Q=result['normalized_Q']))
        (args.output/'timings.json').write_text(json.dumps(report,indent=2)+'\n')
        print(key, times, flush=True)

    with context:
        for p,l,m,N,count in [(4,4,64,512,4),(20,6,500,10000,10)]:
            for backend in args.backend:
                for name in ['soft','a_soft']:
                    for structure in ['standard','strict']:
                        key = f'{name}_{structure}_{backend}_p{p}_l{l}'
                        kwargs = dict(p=p,l=l,structure=structure,m=m,seed=123,backend=backend)
                        if name == 'a_soft':
                            kwargs.update(N=N,n_aug=count)
                        measure(key,getattr(softDOE,name),kwargs)
        base = softDOE.soft(20,6,m=500,seed=123,backend='cpu')['design']
        rng = np.random.default_rng(42)
        design, evaluation = rng.random((512,20)), rng.random((32768,20))
        for backend in args.backend:
            measure('a_soft_supplied_'+backend, softDOE.a_soft,
                    dict(ofat_design=base,n_aug=20,N=10000,backend=backend))
            measure('Q_'+backend, softDOE.Q,
                    dict(alpha=1,beta=1/144,design=design,qmc=evaluation,backend=backend))


if __name__ == '__main__':
    main()
