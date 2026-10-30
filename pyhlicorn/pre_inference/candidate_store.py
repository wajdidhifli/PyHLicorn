"""Compact RAM candidates with adaptive spill and bounded disk block reads."""
from collections import OrderedDict
from pathlib import Path
import shutil
import tempfile
import uuid
import numpy as np


class DiskCandidateSets:
    def __init__(self, names, paths, counts, dtype, chunk_bytes=1024*1024, directory=None, owns=False, spark=False):
        self.names = tuple(names)
        self.paths = dict(paths)
        self.counts = dict(counts)
        self.dtype = np.dtype(dtype).str
        self.chunk_bytes = chunk_bytes
        self.directory = directory
        self._owns = owns
        self._spark = spark
        self.diagnostics = {}

    def __len__(self):
        return sum(self.counts.values())

    @property
    def buffer_bytes(self):
        # This is disk payload size, NOT resident candidate RAM.
        return sum(k*n*np.dtype(self.dtype).itemsize for k,n in self.counts.items())

    def iter_indices(self):
        for size in sorted(self.paths):
            path = self.paths[size]
            if self._spark:
                from pyspark import SparkFiles
                path = SparkFiles.get(path)
            count = self.counts[size]
            records_per_chunk = max(1, self.chunk_bytes // (size*np.dtype(self.dtype).itemsize))
            with open(path, 'rb') as handle:
                remaining = count
                while remaining:
                    records = min(remaining, records_per_chunk)
                    values = np.fromfile(handle, dtype=self.dtype, count=records*size)
                    if values.size != records*size:
                        raise IOError('Incomplete candidate file: '+str(path))
                    values = values.reshape(records, size)
                    values.flags.writeable = False
                    yield from values
                    remaining -= records

    def __iter__(self):
        for indices in self.iter_indices():
            yield indices.tolist(), [self.names[int(i)] for i in indices]

    def for_spark(self, context):
        # Spark distributes disk files, while broadcast state contains metadata only.
        paths = {}
        for size, path in self.paths.items():
            context.addFile(str(Path(path).resolve()))
            paths[size] = Path(path).name
        clone = DiskCandidateSets(self.names, paths, self.counts, self.dtype,
                                  self.chunk_bytes, spark=True)
        clone.diagnostics = self.diagnostics.copy()
        return clone

    def close(self):
        if getattr(self, '_owns', False) and self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)
            self._owns = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Only the driver/creating process owns and deletes the files.
        state['_owns'] = False
        return state

    def __del__(self):
        self.close()


class CandidateWriter:
    def __init__(self, names, temp_dir=None, chunk_bytes=1024*1024, directory=None):
        self.directory = directory or tempfile.mkdtemp(prefix='pyhlicorn-candidates-', dir=temp_dir)
        self.owns = directory is None
        self.names = tuple(names)
        self.dtype = np.dtype('<u2' if len(names) <= 65536 else '<u4')
        self.paths, self.counts = {}, {}
        self.handles = OrderedDict()
        self.identifier = uuid.uuid4().hex
        self.chunk_bytes = chunk_bytes
        self.finished = False

    def append(self, indices):
        self.write_block(len(indices), np.asarray(indices, dtype=self.dtype).tobytes(), 1)

    def write_block(self, size, data, records):
        """Write a compact block directly, including adaptive spill buffers."""
        if len(data) != records*size*self.dtype.itemsize:
            raise ValueError('Candidate block size does not match its record count')
        if size not in self.paths:
            self.paths[size] = str(Path(self.directory)/f'{self.identifier}_{size}.bin')
            self.counts[size] = 0
        if size not in self.handles:
            if len(self.handles) >= 8:
                _, handle = self.handles.popitem(last=False)
                handle.close()
            self.handles[size] = open(self.paths[size], 'ab', buffering=65536)
        self.handles.move_to_end(size)
        self.handles[size].write(data)
        self.counts[size] += records

    def finish(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        result = DiskCandidateSets(self.names, self.paths, self.counts, self.dtype,
                                  self.chunk_bytes, self.directory, self.owns)
        self.finished = True
        return result

    def abort(self):
        for handle in self.handles.values():
            try:
                handle.close()
            except OSError:
                pass  # Preserve the original mining/I/O error while cleaning up.
        self.handles.clear()
        if self.owns:
            shutil.rmtree(self.directory, ignore_errors=True)

    def __del__(self):
        if hasattr(self, 'finished') and not self.finished:
            self.abort()


class MemoryCandidateSets:
    """Read-only contiguous matrices by cardinality; no candidate file reads."""
    def __init__(self, names, blocks):
        self.names = tuple(names)
        self.blocks = dict(blocks)
        self.diagnostics = {}
        for block in self.blocks.values():
            block.flags.writeable = False

    def __len__(self):
        return sum(len(block) for block in self.blocks.values())

    @property
    def buffer_bytes(self):
        return sum(block.nbytes for block in self.blocks.values())

    def iter_indices(self):
        for size in sorted(self.blocks):
            yield from self.blocks[size]

    def __iter__(self):
        for indices in self.iter_indices():
            yield indices.tolist(), [self.names[int(i)] for i in indices]

    def close(self):
        # Uniform store lifetime API; no external files to remove.
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)
        for block in self.blocks.values():
            block.flags.writeable = False


class AdaptiveCandidateWriter:
    """Keep compact payload in RAM up to a budget, then spill exactly once."""
    def __init__(self, names, spill_threshold_bytes=None, temp_dir=None,
                 chunk_bytes=1024*1024, directory=None):
        self.names = tuple(names)
        self.dtype = np.dtype('<u2' if len(names) <= 65536 else '<u4')
        self.spill_threshold_bytes = spill_threshold_bytes
        self.temp_dir, self.chunk_bytes, self.directory = temp_dir, chunk_bytes, directory
        self.buffers, self.counts = {}, {}
        self.payload_bytes = 0
        self.disk = None
        self.finished = False

    def append(self, indices):
        size = len(indices)
        record = np.asarray(indices, dtype=self.dtype).tobytes()
        if self.disk is None and self.spill_threshold_bytes is not None and self.payload_bytes+len(record) > self.spill_threshold_bytes:
            self.disk = CandidateWriter(self.names, self.temp_dir, self.chunk_bytes, self.directory)
            for cardinality in sorted(self.buffers):
                buffer = self.buffers.pop(cardinality)
                self.disk.write_block(cardinality, buffer, self.counts[cardinality])
                del buffer
            self.counts.clear()
        if self.disk is not None:
            self.disk.write_block(size, record, 1)
        else:
            if size not in self.buffers:
                self.buffers[size] = bytearray()
            self.buffers[size].extend(record)
            self.counts[size] = self.counts.get(size,0)+1
        self.payload_bytes += len(record)

    def finish(self):
        if self.disk is not None:
            result = self.disk.finish()
        else:
            # Views preserve backing buffers: no full-size conversion copy.
            blocks = {size:np.frombuffer(buffer,dtype=self.dtype).reshape(-1,size)
                      for size,buffer in self.buffers.items()}
            result = MemoryCandidateSets(self.names,blocks)
            self.buffers.clear()
            self.counts.clear()
        self.finished = True
        return result

    def abort(self):
        if self.disk is not None:
            self.disk.abort()
        self.buffers.clear()
        self.counts.clear()

    def __del__(self):
        if hasattr(self,'finished') and not self.finished:
            self.abort()
