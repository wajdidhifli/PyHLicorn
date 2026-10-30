"""Exactness and lifecycle checks for the default miner independent of PyFIM."""
import itertools
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from pyhlicorn import HLicorn
from pyhlicorn._serialization import native, normalize_frame
from pyhlicorn.pre_inference.patterns import FrequentItemsets
from pyhlicorn.pre_inference.vertical_mining import bitsets_from_matrix, closed_sets, row_bitsets, minimum_count
from pyhlicorn.core.oneGeneHlicorn import oneGeneHlicorn

try:
    import fim
except ImportError:
    fim=None


class StreamingTests(unittest.TestCase):
    def test_exhaustive_closed_sets(self):
        rng=np.random.default_rng(12)
        for trial in range(50):
            matrix=rng.random((12,7))<rng.uniform(.1,.95)
            columns=bitsets_from_matrix(matrix)
            for minimum in (1,3,6):
                expected=set()
                for k in range(2,5):
                    for group in itertools.combinations(range(7),k):
                        extent=np.all(matrix[:,group],axis=1)
                        if extent.sum()<minimum:continue
                        closure=tuple(np.flatnonzero(np.all(matrix[extent],axis=0)))
                        if closure==group:expected.add((group,int(extent.sum())))
                ids=[i for i,t in enumerate(columns) if t and t.bit_count()>=minimum]
                rows=row_bitsets(matrix,ids)
                self.assertEqual(set(closed_sets(columns,len(matrix),minimum,2,4)),expected)
                self.assertEqual(set(closed_sets(columns,len(matrix),minimum,2,4,row_masks=rows)),expected)

    def test_closure_beyond_cap_keeps_all_frequent_pairs(self):
        matrix=pd.DataFrame(np.ones((10,3),bool),columns=list('abc'))
        result=FrequentItemsets.get_coregs(matrix,.1,.1,2)
        try:
            self.assertEqual({tuple(group) for group in result.iter_indices()},{(0,),(1,),(2,),(0,1),(0,2),(1,2)})
        finally:result.close()

    def test_bounded_candidates_against_independent_oracle(self):
        rng=np.random.default_rng(17)
        for trial in range(30):
            matrix=pd.DataFrame(rng.random((20,7))<rng.uniform(.1,.9),columns=list('abcdefg'))
            for support in (.1,.3,1):
                # No TF can occupy all signed transactions; retain an empty row
                # to exercise this same invariant in the reference comparison.
                matrix.iloc[-1,:]=False
                actual=FrequentItemsets.get_coregs(matrix,.2,support,3)
                try:
                    tx=[np.flatnonzero(row).tolist() for row in matrix.to_numpy()]
                    expected=set()
                    for size in range(1,4):
                        for group in itertools.combinations(range(7),size):
                            extent=np.all(matrix.to_numpy()[:,group],axis=1)
                            minimum=minimum_count(.2 if size==1 else support,len(matrix))
                            if extent.sum()<minimum: continue
                            if size==1 or size==3 or tuple(np.flatnonzero(np.all(matrix.to_numpy()[extent],axis=0)))==group:
                                expected.add(frozenset(group))
                    self.assertEqual({frozenset(group) for group in actual.iter_indices()},expected)
                finally:actual.close()

    def test_streaming_limits_and_cleanup(self):
        matrix=pd.DataFrame(np.eye(5,dtype=bool))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MemoryError):
                FrequentItemsets.get_coregs(matrix,.1,.1,2,max_itemsets=1,temp_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()),[])
            with self.assertRaises(TimeoutError):
                FrequentItemsets.get_coregs(matrix,.1,.1,2,timeout_seconds=1e-9,temp_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_chunked_store_pickle_does_not_copy_payload(self):
        matrix=pd.DataFrame(np.random.default_rng(8).random((30,12))<.5)
        store=FrequentItemsets.get_coregs(matrix,.1,.1,3,chunk_bytes=16,spill_threshold_bytes=0)
        try:
            expected=list(store)
            self.assertEqual(list(store),expected)
            payload=pickle.dumps(store)
            worker=pickle.loads(payload)
            self.assertFalse(worker._owns)
            self.assertEqual(list(worker),expected)
            worker.close()
            self.assertTrue(Path(store.directory).exists())
            self.assertTrue(all(not row.flags.writeable for row in store.iter_indices()))
            self.assertEqual(store.buffer_bytes,sum(Path(path).stat().st_size for path in store.paths.values()))
        finally:store.close()
        self.assertFalse(Path(store.directory).exists())

    def test_guarded_vertical_files_survive_input_cleanup(self):
        from pyhlicorn.pre_inference.mining_guard import guarded_mine
        matrix=np.random.default_rng(3).random((20,6))<.5
        store=guarded_mine(matrix,list('abcdef'),dict(algorithm='vertical',min_gene_support=.1,
                          min_coreg_support=.1,max_coreg=2,timeout_seconds=30,spill_threshold_bytes=0),None,30)
        try:
            self.assertTrue(store._owns)
            self.assertGreater(len(list(store)),0)
        finally:store.close()

    def test_inference_cleans_candidate_files_on_success_and_error(self):
        rng=np.random.default_rng(42)
        x=pd.DataFrame(rng.normal(size=(8,20)),index=[f'g{i}' for i in range(8)])
        d=pd.DataFrame(np.where(x>.4,1,np.where(x<-.4,-1,0)),index=x.index)
        with tempfile.TemporaryDirectory() as directory:
            model=HLicorn(max_coreg=2).setSequential().setMining(temp_dir=directory,chunk_bytes=16)
            model.fit(x,list(x.index[:5]),d)
            self.assertEqual(list(Path(directory).iterdir()),[])
            model=HLicorn(max_coreg=2,search_thresh=0,max_candidate_pairs=1).setSequential().setMining(temp_dir=directory)
            with self.assertRaises(MemoryError):model.fit(x,list(x.index[:5]),d)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_no_pyfim_import_for_inference_and_postprocessing(self):
        code='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='fim':raise ImportError('PyFIM intentionally absent')
sys.meta_path.insert(0,Block())
import numpy as np,pandas as pd
from pyhlicorn import HLicorn
from pyhlicorn.pre_inference.patterns import FrequentItemsets
rng=np.random.default_rng(42)
x=pd.DataFrame(rng.normal(size=(8,20)),index=[f'g{i}' for i in range(8)])
d=pd.DataFrame(np.where(x>.4,1,np.where(x<-.4,-1,0)),index=x.index)
net=HLicorn(max_coreg=2,nGRN=2).setSequential().fit(x,list(x.index[:5]),d)
result=FrequentItemsets.findMax([['a','b'],['a','b'],['a']],2,2,2)
assert len(result)==1 and result.iloc[0]['itemsets']==['a','b']
assert 'fim' not in sys.modules
'''
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

    @unittest.skipIf(fim is None,'optional reference PyFIM not installed')
    def test_maximal_postprocessing_matches_pyfim(self):
        rng=np.random.default_rng(5)
        for _ in range(30):
            matrix=rng.random((15,7))<rng.uniform(.1,.8)
            tx=[np.flatnonzero(row).tolist() for row in matrix]
            actual=FrequentItemsets.findMax(tx,3,2,3)
            raw=fim.fpgrowth(tx,supp=20,zmin=2,zmax=3,target='m',report='s')
            self.assertEqual({frozenset(row) for row in actual.itemsets},{frozenset(row[0]) for row in raw})
            expected={frozenset(row[0]):row[1] for row in raw}
            for row in actual.itertuples():self.assertAlmostEqual(row.support,expected[frozenset(row.itemsets)])

    @unittest.skipIf(fim is None,'optional reference PyFIM not installed')
    def test_inferred_models_and_coefficients_match_pyfim(self):
        rng=np.random.default_rng(33)
        x=pd.DataFrame(rng.normal(size=(9,24)),index=[f'g{i}' for i in range(9)])
        d=pd.DataFrame(np.where(x>.4,1,np.where(x<-.4,-1,0)),index=x.index)
        def rows(frame):
            result={}
            for row in native(frame.to_dict('records')):
                act,rep=row['coact'],row['corep']
                key=(row['Target'],tuple(sorted(act)),tuple(sorted(rep)))
                if 'Coef_Acts' in row:
                    row['Coef_Acts']=dict(zip(act,row['Coef_Acts'] or []))
                    row['Coef_Reps']=dict(zip(rep,row['Coef_Reps'] or []))
                row.pop('coact');row.pop('corep');row.pop('Target')
                result[key]=row
            return result
        for model in (None,'lm','cvlm'):
            outputs=[]
            for algorithm in ('vertical','fpgrowth'):
                frame,_,_=HLicorn(max_coreg=3,nGRN=3).setSequential().setLinearModel(model,cv_folds=np.arange(24)%4).setMining(algorithm=algorithm).fit(x,list(x.index[:6]),d,collect=False)
                outputs.append(rows(frame))
            self.assertEqual(outputs[0].keys(),outputs[1].keys())
            for key in outputs[0]:
                for column,value in outputs[0][key].items():
                    other=outputs[1][key][column]
                    if isinstance(value,dict):
                        self.assertEqual(value.keys(),other.keys())
                        for name in value:np.testing.assert_allclose(value[name],other[name],rtol=1e-8,atol=1e-10)
                    elif value is None or other is None:self.assertEqual(value,other)
                    else:np.testing.assert_allclose(value,other,rtol=1e-8,atol=1e-10)

    def test_spark_candidate_metadata_transfer_and_repeat_reads(self):
        try:
            from pyspark import SparkFiles
        except ImportError:
            self.skipTest('optional pyspark not installed')
        import shutil
        from unittest.mock import patch
        matrix=pd.DataFrame(np.random.default_rng(4).random((20,6))<.5)
        store=FrequentItemsets.get_coregs(matrix,.1,.1,3,chunk_bytes=16,spill_threshold_bytes=0)
        try:
            expected=list(store)
            with tempfile.TemporaryDirectory() as directory:
                class Context:
                    def addFile(self,path):shutil.copy(path,Path(directory)/Path(path).name)
                worker=store.for_spark(Context())
                self.assertFalse(worker._owns)
                self.assertTrue(all('/' not in path for path in worker.paths.values()))
                with patch.object(SparkFiles,'get',side_effect=lambda name:str(Path(directory)/name)):
                    self.assertEqual(list(worker),expected)
                    self.assertEqual(list(pickle.loads(pickle.dumps(worker))),expected)
        finally:store.close()

    def test_default_execution_is_sequential_and_streaming(self):
        model=HLicorn()
        self.assertIsNone(model._parallel)
        matrix=pd.DataFrame(np.eye(5,dtype=bool))
        store=FrequentItemsets.get_coregs(matrix,.1,.1,2)
        try:self.assertEqual(store.diagnostics['algorithm'],'vertical')
        finally:store.close()

if __name__=='__main__':unittest.main()
