# Verify exact local bytes, build on Linux, and run the complete current suite.
import pathlib, hashlib, zipfile, json, os, sys, subprocess, tarfile
bundle = pathlib.Path('/content/softdoe-colab-2026-10-22.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == '3d0799067576caca775beeb8518fdf08f20025118874b04877d1115128873aaa'
audit = pathlib.Path('/content/softdoe-verification-2026-10-22')
audit.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    z.extractall(audit)
manifest = json.loads((audit/'manifest.json').read_text())
for name, expected in manifest['validation_sha256'].items():
    assert hashlib.sha256((audit/name).read_bytes()).hexdigest() == expected, name
archive = audit/manifest['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
with tarfile.open(archive) as t:
    for name, expected in manifest['source_sha256'].items():
        match = [m for m in t.getmembers() if m.isfile() and m.name.endswith('/'+name)]
        assert len(match) == 1, name
        assert hashlib.sha256(t.extractfile(match[0]).read()).hexdigest() == expected, name
print('Verified all 29 package inputs and', len(manifest['validation_sha256']), 'validation files.', flush=True)
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1')
env.pop('PYTHONPATH', None)
def checked(label, command):
    result = subprocess.run(command, cwd=audit, env=env, capture_output=True, text=True)
    (audit/(label+'.log')).write_text(result.stdout+result.stderr)
    print(label, 'exit:', result.returncode, flush=True)
    print((result.stdout+result.stderr)[-14000:], flush=True)
    assert result.returncode == 0, label
checked('install', [sys.executable,'-m','pip','install','--no-deps','--force-reinstall',str(archive)])
checked('test-dependencies', [sys.executable,'-m','pip','install','pytest','pydoe','threadpoolctl'])
probe = '''import json,sys,platform,importlib.metadata as m,softDOE,torch
from softDOE import _native
from softDOE._cuda_native import configuration
info=softDOE.backend_info(refresh=True)
assert torch.cuda.is_available() and info['cuda']['available'],info
assert hasattr(_native,'optimize_cuda') and configuration(torch.cuda.current_device()) is not None
print(json.dumps(dict(python=sys.version,platform=platform.platform(),package=softDOE.__file__,versions={n:m.version(n) for n in ['softDOE','torch','numpy','scipy','pydoe','pytest']},cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),backends=info),indent=2))'''
checked('environment', [sys.executable,'-c',probe])
checked('pytest', [sys.executable,'-m','pytest','tests','-q','-ra','--junitxml='+str(audit/'pytest.xml')])
import xml.etree.ElementTree as ET
cases = ET.parse(audit/'pytest.xml').getroot().findall('.//testcase')
def is_cuda(c):
    return c.attrib['classname'].endswith('test_native_cuda') or (c.attrib['classname'].endswith('test_cuda_batching') and '[cuda-' in c.attrib['name']) or (c.attrib['classname'].endswith(('test_q','test_backends')) and '[cuda' in c.attrib['name'] and 'fallback' not in c.attrib['name'] and 'unavailable' not in c.attrib['name'])
cuda_cases = [c for c in cases if is_cuda(c)]
assert len(cuda_cases)==34, [c.attrib for c in cuda_cases]
assert all(len(c)==0 for c in cuda_cases), 'CUDA cases must run and pass without skips'
summary = dict(total=len(cases),passed=sum(len(c)==0 for c in cases),skipped=[c.attrib for c in cases if c.find('skipped') is not None],cuda_passed=len(cuda_cases),archive_sha256=manifest['archive_sha256'])
(audit/'summary.json').write_text(json.dumps(summary,indent=2))
print('CURRENT CUDA VERIFICATION:', json.dumps({k:v for k,v in summary.items() if k!='skipped'}), flush=True)
checked('example',[sys.executable,'examples/mofat_examples.py'])