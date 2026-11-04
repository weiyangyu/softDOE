# Keep Colab's preinstalled Numba compatible, then validate all 444 CUDA workflows.
checked('numpy-compatible', [sys.executable,'-m','pip','install','numpy==2.2.6'])
checked('ipython-dependency', [sys.executable,'-m','pip','install','jedi'])
checked('pip-check', [sys.executable,'-m','pip','check'])
checked('environment-compatible', [sys.executable,'-c',probe])
checked('pytest-compatible', [sys.executable,'-m','pytest','tests','-q','-ra','--junitxml='+str(audit/'pytest-compatible.xml')])
compatible_cases = ET.parse(audit/'pytest-compatible.xml').getroot().findall('.//testcase')
assert len(compatible_cases)==387
assert sum(is_cuda(c) for c in compatible_cases)==34
assert all(len(c)==0 for c in compatible_cases if is_cuda(c))
matrix_script = r'''
import json,sys,pathlib,traceback,torch
from collections import Counter
root=pathlib.Path(__file__).parent
sys.path.insert(0,str(root/'tests'))
from experiment_cases import experiment_cases
from run_mofat_experiments import prepare_cases,run_python_case,write_json,append_json
import softDOE
assert softDOE.backend_info(refresh=True)['cuda']['available']
torch.set_num_threads(1)
out=root/'cuda-full'
out.mkdir(exist_ok=False)
cases=prepare_cases(experiment_cases('full'))
write_json(out/'manifest.json',dict(profile='full',repeats=1,mofat_version='2.0',backends=['cuda'],r_backends=['cpu'],r_backend='cpu',cases=cases))
records=[]
for i,case in enumerate(cases,1):
    try:
        record=run_python_case(case,'cuda',1)
    except Exception as error:
        record=dict(status='error',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    record.update(id=case['id'],backend='cuda')
    records.append(record)
    append_json(out/'python-results.jsonl',record)
    if record['status']=='ok':
        append_json(out/'python-designs.jsonl',{k:record[k] for k in ['id','backend','design']})
    print(f"{i}/{len(cases)} {case['id']}: {record['status']}",flush=True)
counts=Counter(r['status'] for r in records)
summary=dict(total=len(records),counts=dict(counts),failed=[r for r in records if r['status'] not in ['ok','expected_error']],scope='Real CUDA invariants, repeatability, no fallback, Python CPU rescoring; R comparisons separate')
write_json(out/'summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k!='failed'}),flush=True)
assert len(records)==444 and not summary['failed']
'''
(audit/'run_cuda_matrix.py').write_text(matrix_script)
with (audit/'cuda-matrix.log').open('w') as log:
    matrix_process=subprocess.Popen([sys.executable,str(audit/'run_cuda_matrix.py')],cwd=audit,env=env,stdout=log,stderr=subprocess.STDOUT)
print('CUDA full matrix started:',matrix_process.pid)