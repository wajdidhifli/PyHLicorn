"""Best-effort mining RSS watchdog (not an OS hard memory limit).

Runs a separate Python interpreter, including from notebooks/macOS. Files avoid
large results in pipes. The subprocess is terminated on error or interruption.
"""
import json
import os
from pathlib import Path
import pickle
import subprocess
import shutil
import sys
import tempfile
import time
import numpy as np


def guarded_mine(matrix, names, arguments, memory_limit_gb, timeout_seconds):
    if memory_limit_gb is not None:
        try:
            import psutil
        except ImportError as exc:
            raise ImportError('RSS monitoring requires pip install psutil or pyhlicorn[monitor]') from exc
    output_directory = None
    transferred = False
    if arguments.get('algorithm') == 'vertical':
        output_directory = tempfile.mkdtemp(prefix='pyhlicorn-candidates-', dir=arguments.get('temp_dir'))
        arguments = {**arguments, '_output_directory': output_directory}
    try:
        with tempfile.TemporaryDirectory(prefix='pyhlicorn-mining-', dir=arguments.get('temp_dir')) as directory:
            path = Path(directory)
            np.save(path/'matrix.npy', matrix, allow_pickle=False)
            (path/'options.json').write_text(json.dumps(dict(names=names, arguments=arguments)))
            with (path/'worker.log').open('w') as log:
                worker = subprocess.Popen([sys.executable, '-m', 'pyhlicorn.pre_inference.mining_guard', directory],
                                          stdout=log, stderr=log)
                started = time.monotonic()
                peak = 0
                try:
                    process = psutil.Process(worker.pid) if memory_limit_gb is not None else None
                    while worker.poll() is None:
                        if process is not None:
                            try:
                                rss = process.memory_info().rss
                            except psutil.NoSuchProcess:
                                rss = 0
                            peak = max(peak, rss)
                            if rss > memory_limit_gb * 1024**3:
                                raise MemoryError(f'Mining exceeded its {memory_limit_gb:g} GiB RSS budget; '
                                                  'worker stopped, no incomplete candidates returned.')
                        if timeout_seconds is not None and time.monotonic()-started > timeout_seconds:
                            raise TimeoutError('Mining exceeded timeout_seconds; worker stopped.')
                        time.sleep(0.1)
                    if worker.returncode:
                        raise RuntimeError('Mining worker failed:\n' + (path/'worker.log').read_text()[-6000:])
                finally:
                    if worker.poll() is None:
                        worker.terminate()
                        try:
                            worker.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.wait()
            with (path/'result.pkl').open('rb') as handle:
                result = pickle.load(handle)
            if output_directory is not None:
                if getattr(result, 'directory', None) == output_directory:
                    result._owns = True
                else:
                    # No spill: the child returned compact arrays and never used this directory.
                    shutil.rmtree(output_directory, ignore_errors=True)
                    output_directory = None
            result.diagnostics.update(monitored_peak_rss_bytes=peak if memory_limit_gb is not None else None,
                                      memory_limit_gb=memory_limit_gb, timeout_seconds=timeout_seconds,
                                      guarded_elapsed_seconds=time.monotonic()-started)
            transferred = True
            return result
    finally:
        if output_directory is not None and not transferred:
            shutil.rmtree(output_directory, ignore_errors=True)


def main():
    from .patterns import _mine
    path = Path(sys.argv[1])
    options = json.loads((path/'options.json').read_text())
    matrix = np.load(path/'matrix.npy', mmap_mode='r')
    result = _mine(matrix, options['names'], **options['arguments'])
    with (path/'result.pkl').open('wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
