"""Storage budget, exact spill transition, read-only data and worker ownership."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pyhlicorn import HLicorn
from pyhlicorn.pre_inference.candidate_store import MemoryCandidateSets, DiskCandidateSets
from pyhlicorn.pre_inference.patterns import FrequentItemsets
from pyhlicorn.pre_inference.mining_guard import guarded_mine


class AdaptiveTests(unittest.TestCase):
    def matrix(self):
        return pd.DataFrame(np.random.default_rng(4).random((20,8))<.5)

    def test_default_memory_has_no_candidate_io(self):
        with patch('pyhlicorn.pre_inference.candidate_store.tempfile.mkdtemp', side_effect=AssertionError('no candidate directory expected')),\
             patch('builtins.open', side_effect=AssertionError('no candidate file reads/writes expected')):
            store=FrequentItemsets.get_coregs(self.matrix(),.1,.1,3)
            self.assertIsInstance(store,MemoryCandidateSets)
            self.assertGreater(len(list(store)),0)
            self.assertEqual(store.diagnostics['storage'],'memory')
            self.assertEqual(store.diagnostics['candidate_file_bytes'],0)
            self.assertGreater(store.diagnostics['resident_candidate_payload_bytes'],0)
            self.assertTrue(all(block.flags.c_contiguous and not block.flags.writeable for block in store.blocks.values()))

    def test_budget_boundary_and_midstream_spill(self):
        data=pd.DataFrame(np.eye(5,dtype=bool)) # five uint16 singleton records = 10 bytes
        exact=FrequentItemsets.get_coregs(data,.1,.1,1,spill_threshold_bytes=10)
        self.assertIsInstance(exact,MemoryCandidateSets)
        expected=list(exact)
        with tempfile.TemporaryDirectory() as directory:
            below=FrequentItemsets.get_coregs(data,.1,.1,1,spill_threshold_bytes=9,temp_dir=directory)
            try:
                self.assertIsInstance(below,DiskCandidateSets)
                self.assertEqual(list(below),expected)
                self.assertEqual(below.buffer_bytes,10)
                self.assertEqual(sum(Path(p).stat().st_size for p in below.paths.values()),10)
            finally:below.close()
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_all_cardinalities_match_forced_disk_after_spill(self):
        outputs=[]
        for threshold in (0,20,50*1024*1024):
            store=FrequentItemsets.get_coregs(self.matrix(),.1,.1,3,spill_threshold_bytes=threshold,chunk_bytes=16)
            try:outputs.append(list(store))
            finally:store.close()
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(outputs[1],outputs[2])

    def test_memory_pickle_readonly_and_repeat_reads(self):
        store=FrequentItemsets.get_coregs(self.matrix(),.1,.1,3)
        restored=pickle.loads(pickle.dumps(store))
        self.assertEqual(list(restored),list(store))
        self.assertEqual(list(restored),list(store))
        self.assertTrue(all(not block.flags.writeable for block in restored.blocks.values()))

    def test_guarded_memory_cleans_unused_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            store=guarded_mine(self.matrix().to_numpy(),list('abcdefgh'),dict(
                algorithm='vertical',min_gene_support=.1,min_coreg_support=.1,max_coreg=3,
                timeout_seconds=30,temp_dir=directory),None,30)
            self.assertIsInstance(store,MemoryCandidateSets)
            self.assertGreater(len(list(store)),0)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_pipeline_parity_sequential_joblib_for_both_paths(self):
        rng=np.random.default_rng(42)
        x=pd.DataFrame(rng.normal(size=(8,20)),index=[f'g{i}' for i in range(8)])
        d=pd.DataFrame(np.where(x>.4,1,np.where(x<-.4,-1,0)),index=x.index)
        expected=None
        for threshold in (0,50*1024*1024):
            for jobs in (1,2):
                model=HLicorn(max_coreg=2,nGRN=2).setMining(spill_threshold_bytes=threshold)
                if jobs==2:model.setJoblib(n_jobs=2)
                frame,metadata,_=model.fit(x,list(x.index[:5]),d,collect=False)
                self.assertEqual(metadata['mining']['storage'],'disk' if threshold==0 else 'memory')
                if expected is None:expected=frame
                else:pd.testing.assert_frame_equal(frame,expected)

    def test_large_numpy_payload_reaches_joblib_workers_readonly(self):
        from joblib import Parallel, delayed
        matrix=np.tile(np.array([[0,1,2]],dtype=np.uint16),(300000,1))
        store=MemoryCandidateSets(list('abc'),{3:matrix})
        def summarize(candidate_store):
            block=candidate_store.blocks[3]
            return len(candidate_store),int(block.sum()),block.flags.writeable
        observed=Parallel(n_jobs=2)(delayed(summarize)(store) for _ in range(2))
        self.assertEqual(observed,[(300000,900000,False)]*2)

    def test_invalid_budget(self):
        for threshold in (-1,1.5,True):
            with self.assertRaises(ValueError):
                FrequentItemsets.get_coregs(self.matrix(),.1,.1,3,spill_threshold_bytes=threshold)

if __name__=='__main__':unittest.main()
