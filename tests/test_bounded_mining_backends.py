import importlib.util
import itertools
import unittest
import numpy as np
import pandas as pd
from pyhlicorn.pre_inference.patterns import FrequentItemsets
from pyhlicorn.pre_inference.vertical_mining import minimum_count

class BoundedBackendTests(unittest.TestCase):
    def test_identical_four_regulators_at_cap(self):
        frame=pd.DataFrame([[1]*4,[1]*4,[0]*4,[0]*4],columns=list('ABCD'),dtype=bool)
        backends=['vertical'] + (['fpgrowth','apriori'] if importlib.util.find_spec('fim') else [])
        for algorithm in backends:
            store=FrequentItemsets.get_coregs(frame,.1,.2,3,algorithm=algorithm)
            try:
                expected={(i,) for i in range(4)}|set(itertools.combinations(range(4),3))
                self.assertEqual({tuple(sorted(g)) for g in store.iter_indices()},expected,algorithm)
            finally:
                if hasattr(store,'close'):store.close()

    @unittest.skipUnless(importlib.util.find_spec('fim'),'optional PyFIM not installed')
    def test_optional_backends_match_bounded_oracle(self):
        rng=np.random.default_rng(99)
        for trial in range(20):
            x=rng.random((12,6))<rng.uniform(.2,.9);x[-1]=False
            for cap in (1,2,3):
                expected=set()
                for size in range(1,cap+1):
                    for g in itertools.combinations(range(6),size):
                        extent=np.all(x[:,g],axis=1)
                        if extent.sum()<minimum_count(.2,len(x)):continue
                        if size==1 or size==cap or tuple(np.flatnonzero(np.all(x[extent],axis=0)))==g:
                            expected.add(g)
                for algorithm in ('vertical','fpgrowth','apriori'):
                    result=FrequentItemsets.get_coregs(pd.DataFrame(x),.2,.2,cap,algorithm=algorithm)
                    try:self.assertEqual({tuple(sorted(g)) for g in result.iter_indices()},expected,(trial,cap,algorithm))
                    finally:
                        if hasattr(result,'close'):result.close()
