import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from pyhlicorn import HLicorn
from pyhlicorn.core.memory import parse_memory_budget, plan_memory
from pyhlicorn.pre_inference.patterns import FrequentItemsets

class MemoryBudgetTests(unittest.TestCase):
    def test_units_and_invalid_values(self):
        self.assertEqual(parse_memory_budget('16GB'),16_000_000_000)
        self.assertEqual(parse_memory_budget('1.5 GiB'),int(1.5*1024**3))
        self.assertEqual(parse_memory_budget(1024),1024)
        self.assertIsNone(parse_memory_budget(None))
        for value in ('16','0GB','-1GB','nanGB',False,0,1.5):
            with self.assertRaises(ValueError):parse_memory_budget(value)

    def test_default_storage_never_spills(self):
        frame=pd.DataFrame(np.ones((8,6),bool))
        with tempfile.TemporaryDirectory() as directory:
            store=FrequentItemsets.get_coregs(frame,.1,.2,3,temp_dir=directory)
            try:
                self.assertEqual(store.diagnostics['storage'],'memory')
                self.assertIsNone(store.diagnostics['spill_threshold_bytes'])
                self.assertEqual(list(Path(directory).iterdir()),[])
            finally:store.close()
        self.assertIsNone(HLicorn()._max_candidate_pairs)

    def test_concurrency_reduces_and_recovers_with_larger_budget(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            small=plan_memory('1GiB',10_000_000,100000,100,3,4)
            large=plan_memory('64GiB',10_000_000,100000,100,3,4)
        self.assertLessEqual(small['effective_workers'],large['effective_workers'])
        self.assertEqual(large['effective_workers'],large['requested_workers'])
        self.assertFalse(small['hard_cap'])

    def test_budget_spill_and_local_backend_parity(self):
        rng=np.random.default_rng(223)
        x=pd.DataFrame(rng.normal(size=(14,24)),index=[f'g{i}' for i in range(14)])
        d=pd.DataFrame(np.where(x>.4,1,np.where(x<-.4,-1,0)),index=x.index,columns=x.columns)
        outputs=[]
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for budget,parallel in [(None,False),('1KiB',False),('1KiB',True),('4GiB',True)]:
                model=HLicorn(max_coreg=3,nGRN=3,memory_budget=budget).setMining(temp_dir=directory)
                if parallel:model.setJoblib(2)
                with contextlib.redirect_stdout(io.StringIO()):
                    net=model.fit(x,list(x.index[:10]),d)
                outputs.append(net.GRN)
                self.assertEqual(net.metadata['mining']['storage'],'disk' if budget == '1KiB' else 'memory')
                if budget == '1KiB':
                    self.assertEqual(net.metadata['memory_plan']['effective_workers'],1)
                self.assertEqual(list(Path(directory).iterdir()),[])
        for frame in outputs[1:]:pd.testing.assert_frame_equal(outputs[0],frame)

    def test_cli_help_has_one_memory_control(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(root/'run_inference.py'),'--help'],capture_output=True,text=True,check=True)
        self.assertIn('--memory-budget',result.stdout)
        self.assertNotIn('--spill-mb',result.stdout)
        self.assertNotIn('--max-candidate-pairs',result.stdout)

    def test_budget_rejects_nonstreaming_miner(self):
        x=pd.DataFrame(np.ones((6,12)))
        with self.assertRaisesRegex(ValueError,'vertical'):
            HLicorn(memory_budget='1GB').setMining(algorithm='fpgrowth').fit(x,list(x.index))
