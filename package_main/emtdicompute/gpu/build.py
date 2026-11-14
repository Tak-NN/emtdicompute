"""Build the ctypes backend and PTX without importing PyTorch or accessing a GPU."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def _windows_compiler(environment):
    compiler = shutil.which('cl.exe', path=environment.get('PATH'))
    if compiler:
        return compiler, environment
    vswhere = Path(environment.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.is_file():
        raise RuntimeError('MSVC C++ Build Tools are required. Run in an x64 Native Tools prompt or install Visual Studio C++ Build Tools.')
    installation = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not installation:
        raise RuntimeError('Visual Studio C++ Build Tools were not found.')
    vcvars = Path(installation) / 'VC/Auxiliary/Build/vcvars64.bat'
    command = f'call "{vcvars}" >nul && set'
    result = subprocess.run(f'"{environment["COMSPEC"]}" /d /s /c "{command}"',
                            capture_output=True, text=True, check=True)
    environment = dict(environment)
    for line in result.stdout.splitlines():
        if '=' in line and not line.startswith('='):
            name, value = line.split('=', 1)
            if name.upper() in {'PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'VCTOOLSINSTALLDIR', 'WINDOWSSDKDIR', 'WINDOWSSDKVERSION'}:
                environment[name] = value
    tools = next(value for name, value in environment.items() if name.upper() == 'VCTOOLSINSTALLDIR')
    return str(Path(tools) / 'bin/Hostx64/x64/cl.exe'), environment


def build(output_dir=None, optix_include=None, build_dir=None):
    source = Path(__file__).resolve().parent
    output = Path(output_dir or source / '_native').resolve()
    work = Path(build_dir or output.parent / '.native-build').resolve()
    include = Path(optix_include or os.environ.get('OPTIX_ROOT', ''))
    if not (include / 'optix.h').is_file():
        include = include / 'include'
    if not (include / 'optix.h').is_file():
        raise RuntimeError('OptiX 9.1 headers are required to build from source. Set OPTIX_ROOT to the SDK directory '
                           'or pass --optix-include <OptiX/include>. Platform wheels do not require build tools.')
    include = include.resolve()
    version = re.search(r'^#define\s+OPTIX_VERSION\s+(\d+)', (include / 'optix.h').read_text(), re.M)
    if not version or int(version.group(1)) != 90100:
        raise RuntimeError('This backend is built and tested with OptiX 9.1 headers (OPTIX_VERSION=90100).')
    environment = dict(os.environ)
    cuda_value = environment.get('CUDA_PATH') or environment.get('CUDA_HOME')
    nvcc = str(Path(cuda_value) / 'bin' / ('nvcc.exe' if sys.platform == 'win32' else 'nvcc')) if cuda_value else shutil.which('nvcc')
    if not nvcc or not Path(nvcc).is_file():
        raise RuntimeError('CUDA Toolkit (nvcc and CUDA Driver API headers/library) is required to build from source. Set CUDA_PATH or CUDA_HOME.')
    cuda = Path(nvcc).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    commands = [[nvcc, '--ptx', '--fmad=false', '--std=c++17', '-arch=compute_75',
        '-I'+str(include), '-I'+str(source/'native'), str(source/'native/rays.cu'), '-o', str(output/'rays.ptx')]]
    if sys.platform == 'win32':
        compiler, environment = _windows_compiler(environment)
        commands.append([compiler, '/nologo', '/LD', '/O2', '/EHsc', '/MD', '/std:c++17', '/utf-8',
            '/I'+str(cuda/'include'), '/I'+str(include), '/I'+str(source/'native'), str(source/'native/bridge.cpp'),
            '/Fo'+str(work/'bridge.obj'), '/Fd'+str(work/'bridge.pdb'), '/Fe'+str(output/'emtdi_optix.dll'),
            '/link', '/LIBPATH:'+str(cuda/'lib/x64'), 'cuda.lib', 'advapi32.lib',
            '/IMPLIB:'+str(work/'emtdi_optix.lib'), '/PDB:'+str(work/'emtdi_optix.pdb')])
    elif sys.platform == 'linux':
        compiler = environment.get('CXX', 'c++')
        # CUDA stubs suffice for linking; execution uses the installed driver.
        library_dirs = [cuda/'lib64/stubs', cuda/'lib64', cuda/'targets/x86_64-linux/lib/stubs']
        commands.append([compiler, '-shared', '-fPIC', '-O2', '-std=c++17', '-fvisibility=hidden',
            '-I'+str(cuda/'include'), '-I'+str(include), '-I'+str(source/'native'), str(source/'native/bridge.cpp'),
            *['-L'+str(path) for path in library_dirs], '-lcuda', '-ldl', '-o', str(output/'libemtdi_optix.so')])
    else:
        raise RuntimeError('GPU builds support Windows x64 and Linux x86_64; this platform has not been implemented.')
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=work, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        (work/f'compile_{index}.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Native build failed; see {work / f"compile_{index}.log"}\n{result.stdout[-4000:]}')
    (output/'build_info.json').write_text(json.dumps(dict(abi_version=1, optix_version=90100,
        ptx_target='compute_75', platform=sys.platform), indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--optix-include', type=Path)
    options = parser.parse_args()
    print(build(**vars(options)))
