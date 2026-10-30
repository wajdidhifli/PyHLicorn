"""Focused correctness checks; these are not CoRegNet equivalence benchmarks."""
import importlib
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pyhlicorn import HLicorn, PreProcess
from pyhlicorn.cpylibs import comblicorn
from pyhlicorn.core.oneGeneHlicorn import oneGeneHlicorn
from statsmodels.stats.multitest import multipletests


class RegressionTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame(rng.normal(size=(8, 20)), index=[f'g{i}' for i in range(8)])
        d = pd.DataFrame(np.where(x > .4, 1, np.where(x < -.4, -1, 0)), index=x.index)
        return x, d, list(x.index[:5])

    def test_constant_rows_are_neutral(self):
        d = PreProcess.discretize_gene_expression(pd.DataFrame([[4.]*12, [0.]*12]))
        self.assertTrue((d.to_numpy() == 0).all())

    def test_reordered_discrete_input_rejected(self):
        x,d,tf = self.fixture()
        with self.assertRaisesRegex(ValueError, 'identical ordered'):
            HLicorn().setSequential().fit(x,tf,d.iloc[::-1])

    def test_missing_discrete_values_rejected(self):
        x,d,tf = self.fixture(); d = d.astype(float); d.iloc[0,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing values'):
            HLicorn().setSequential().fit(x,tf,d)

    def test_search_endpoints_and_small_threshold(self):
        x,d,tf = self.fixture()
        for threshold in [0, .01, 1]:
            grn,_,_ = HLicorn(max_coreg=1, search_thresh=threshold, nGRN=1).setSequential().setLinearModel(None).fit(x,tf,d,collect=False)
            self.assertIn('Target',grn.columns)

    def test_partially_overlapping_sets_rejected(self):
        reg = np.array([[1,-1,1,-1],[1,-1,1,-1],[1,-1,1,-1]])
        bit = np.concatenate([reg==1,reg==-1],axis=1).T
        items = (([0,1],['a','b']),([1,2],['b','c']))
        out = oneGeneHlicorn('target',reg[0],reg,items,bit,0,reg.T,reg[0],100,None,False)
        self.assertGreater(len(out),0)
        for row in out.itertuples():
            a = row.coact if isinstance(row.coact,(list,tuple)) else []
            r = row.corep if isinstance(row.corep,(list,tuple)) else []
            self.assertFalse(set(a)&set(r))
            self.assertTrue(a or r)

    def test_c_kernel_truth_table_and_reference_counts(self):
        a=np.array([-1,0,1],dtype=np.int32)
        r=np.array([-1,0,2],dtype=np.int32)
        g=np.array([-1,0,1],dtype=np.int32)
        out=np.zeros(1); ia=np.zeros(1,dtype=np.int32); ir=ia.copy()
        before = [sys.getrefcount(v) for v in (out,ia,ir)]
        for _ in range(100):
            result=comblicorn.combLicorn(a,1,r,1,g,3,out,ia,ir)
            self.assertAlmostEqual(result[0][0],np.mean(np.abs(g-np.clip(a-r,-1,1))))
            del result
        self.assertEqual(before,[sys.getrefcount(v) for v in (out,ia,ir)])

    def test_c_kernel_rejects_unsafe_buffers(self):
        a=np.array([-1,0,1],dtype=np.int32)
        for bad in [a.astype(float), a[:1], [-1,0,1]]:
            with self.assertRaises((TypeError,ValueError)):
                comblicorn.combLicorn(bad,1,a,1,a,3,np.zeros(1),np.zeros(1,dtype=np.int32),np.zeros(1,dtype=np.int32))

    def test_fisher_adjustment_uses_full_family(self):
        module=importlib.import_module('pyhlicorn.post_inference._coregulators')
        bytf={'a':{'act':['g1','g2','g3'],'rep':[]},'b':{'act':['g1','g2'],'rep':[]},'c':{'act':['g3','g4'],'rep':[]}}
        obj=SimpleNamespace(adjlist=SimpleNamespace(bygene={f'g{i}':{} for i in range(1,11)},bytf=bytf))
        fr=pd.DataFrame({'itemsets':[['a','b'],['a','c']], 'support':[.2,.1]})
        out=module.__dict__['__doFisher'](obj,fr,10,'greater','holm',(1,'sequential'))
        np.testing.assert_allclose(out.adjustedPvalue,multipletests(out.fisherTest,method='holm')[1])
        self.assertTrue((out.adjustedPvalue > out.fisherTest).any())

    def test_shared_cv_folds_agree_between_backends(self):
        from pyhlicorn.during_inference.__LinearModel import make_cv_folds
        x,d,tf=self.fixture()
        folds=make_cv_folds(x.shape[1],random_state=19)
        a,_,_=HLicorn(max_coreg=2,nGRN=2).setSequential().setLinearModel('cvlm',cv_folds=folds).fit(x,tf,d,collect=False)
        b,_,_=HLicorn(max_coreg=2,nGRN=2).setJoblib(n_jobs=2).setLinearModel('cvlm',cv_folds=folds).fit(x,tf,d,collect=False)
        pd.testing.assert_frame_equal(a,b)

    def test_cv_seed_does_not_mutate_global_rng(self):
        from pyhlicorn.during_inference.__LinearModel import make_cv_folds
        np.random.seed(123)
        before=np.random.get_state()
        a=make_cv_folds(25,random_state=7)
        np.testing.assert_array_equal(a,make_cv_folds(25,random_state=7))
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])
        self.assertEqual(len(np.unique(a)),10)

    def test_cv_rejects_small_default_or_invalid_folds(self):
        from pyhlicorn.during_inference.__LinearModel import make_cv_folds
        with self.assertRaises(ValueError):make_cv_folds(5)
        with self.assertRaises(ValueError):make_cv_folds(10,[0]*10)
        with self.assertRaises(ValueError):make_cv_folds(10,[0,1])
        np.testing.assert_array_equal(make_cv_folds(4,[0,0,1,1]),[0,0,1,1])

    def test_cv_matches_manual_heldout_predictions(self):
        lm = importlib.import_module('pyhlicorn.during_inference.__LinearModel')
        x=np.arange(24,dtype=float).reshape(12,2)
        x[:,1]=np.sin(x[:,1]);y=1+x[:,0]*.2+x[:,1]*.7
        folds=np.arange(12)%3
        coefs,r2,rmse=lm.__dict__['__cvlm_numpy_linreg'](x,y,fold_ids=folds)
        np.testing.assert_allclose(coefs,[.2,.7],atol=1e-10)
        self.assertAlmostEqual(r2,1)
        self.assertLess(rmse,1e-10)

    def test_sequential_and_joblib_agree(self):
        x,d,tf=self.fixture()
        a,_,_=HLicorn(max_coreg=2,nGRN=2).setSequential().fit(x,tf,d,collect=False)
        b,_,_=HLicorn(max_coreg=2,nGRN=2).setJoblib(n_jobs=2).fit(x,tf,d,collect=False)
        pd.testing.assert_frame_equal(a,b)

if __name__=='__main__':
    unittest.main()
