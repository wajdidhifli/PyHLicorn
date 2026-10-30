"""Independent native checks and normalized equivalence to frozen v3 behavior."""
import importlib
import sys
import unittest
import numpy as np
import pandas as pd
from pyhlicorn import HLicorn
from pyhlicorn.cpylibs import comblicorn
from pyhlicorn.core.oneGeneHlicorn import oneGeneHlicorn
from pyhlicorn.during_inference.__comblicorn import score_disjoint, _membership
from pyhlicorn._serialization import normalize_frame, native
from pyhlicorn.pre_inference.patterns import FrequentItemsets
from _reference_v3 import oneGeneHlicorn as v3


class V4Tests(unittest.TestCase):
    def test_native_matches_independent_disjoint_reference(self):
        rng = np.random.default_rng(29)
        acts = [[2, 0], [1], [2, 3], []]
        reps = [[1, 2], [4], [0], []]
        a = rng.integers(-1, 2, (len(acts), 17), dtype=np.int32)
        r = rng.choice(np.array([-1, 0, 2], np.int32), (len(reps), 17))
        a[-1] = 0; r[-1] = 0
        g = rng.integers(-1, 2, 17, dtype=np.int32)
        expected = [(np.mean(np.abs(g-np.clip(a[i]-r[j], -1, 1))), i, j)
                    for i in range(len(acts)) for j in range(len(reps))
                    if (acts[i] or reps[j]) and not(set(acts[i]) & set(reps[j]))]
        mae, ia, ir = score_disjoint(a.ravel(), acts, r.ravel(), reps, g)
        self.assertEqual(list(zip(ia, ir)), [(i,j) for _,i,j in expected])
        np.testing.assert_array_equal(mae, [m for m,_,_ in expected])

    def native_arguments(self):
        a=np.array([1,-1,0,0],np.int32);g=np.array([1,-1],np.int32)
        f,o=_membership([[0],[]]);rf,ro=_membership([[1],[]])
        rep=a.copy();rep[rep==1]=2
        return [a,2,rep,2,g,2,f,o,rf,ro,np.empty(4),np.empty(4,np.int32),np.empty(4,np.int32)]

    def test_native_rejects_unsafe_membership(self):
        for bad in [np.array([0,2,1],np.int64), np.array([0,-1,1],np.int64),
                    np.array([0,10,1],np.int64),np.array([1,1,1],np.int64),
                    np.array([0,1,1],np.int32)]:
            args=self.native_arguments();args[7]=bad
            with self.assertRaises(ValueError):comblicorn.combLicornDisjoint(*args)
        for flat,offset in [([1,0],[0,2,2]),([0,0],[0,2,2]),([-1],[0,1,1])]:
            args=self.native_arguments();args[6]=np.array(flat,np.int32);args[7]=np.array(offset,np.int64)
            with self.assertRaises(ValueError):comblicorn.combLicornDisjoint(*args)
        args=self.native_arguments();args[0]=args[0].astype(float)
        with self.assertRaises(ValueError):comblicorn.combLicornDisjoint(*args)
        args=self.native_arguments();args[4]=np.array([2**30,0],np.int32)
        with self.assertRaises(ValueError):comblicorn.combLicornDisjoint(*args)

    def test_native_output_reference_counts(self):
        args=self.native_arguments();buffers=args[-3:]
        before=[sys.getrefcount(x) for x in buffers]
        for _ in range(100):
            output=comblicorn.combLicornDisjoint(*args)
            del output
        self.assertEqual(before,[sys.getrefcount(x) for x in buffers])

    def test_pruning_and_ranking_match_v3(self):
        for seed in (2, 9, 24):
            rng=np.random.default_rng(seed)
            reg=rng.choice([-1,0,1],size=(6,23))
            bit=np.concatenate([reg==1,reg==-1],axis=1).T
            items=FrequentItemsets.get_coregs(pd.DataFrame(bit,columns=list('abcdef')),.1,.1,3)
            numerical=rng.normal(size=reg.T.shape)
            target=rng.choice([-1,0,1],size=23)
            y=rng.normal(size=23)
            for threshold in (0,.01,.3,1/3,.5,1):
                for model in (None,'lm','cvlm'):
                    folds=np.arange(23)%4 if model=='cvlm' else None
                    # Names/feature order, accepted pair order, rank ties, and scores must match.
                    args=('target',target,reg,items,bit,threshold,numerical,y,3,model,False)
                    expected=normalize_frame(v3(*args,cv_folds=folds)).reset_index(drop=True)
                    actual=oneGeneHlicorn(*args,cv_folds=folds).reset_index(drop=True)
                    pd.testing.assert_frame_equal(expected,actual,check_dtype=False)
            args=('target',target,reg,items,bit,.3,numerical,y,None,None,False)
            pd.testing.assert_frame_equal(normalize_frame(v3(*args)).reset_index(drop=True),
                                          oneGeneHlicorn(*args).reset_index(drop=True),check_dtype=False)

    def test_names_only_resolved_for_selected_models(self):
        rng=np.random.default_rng(4)
        reg=rng.choice([-1,0,1],size=(6,20))
        bit=np.concatenate([reg==1,reg==-1],axis=1).T
        items=FrequentItemsets.get_coregs(pd.DataFrame(bit,columns=list('abcdef')),.1,.1,3)
        class CountNames:
            def __init__(self,names):self.names=names;self.calls=0
            def __getitem__(self,i):self.calls+=1;return self.names[i]
        names=CountNames(items.names);items.names=names
        diagnostics={}
        out=oneGeneHlicorn('target',reg[0],reg,items,bit,.3,reg.T,reg[0],1,None,False,diagnostics=diagnostics)
        self.assertEqual(names.calls,sum(map(len,out.coact))+sum(map(len,out.corep)))
        self.assertLess(diagnostics['multi_conjunction_checks'],diagnostics['multi_conjunction_checks_without_pruning'])
        self.assertTrue(all(not view.flags.writeable for view in items.iter_indices()))

    def test_raw_local_backends_normalize_regulator_lists(self):
        rng=np.random.default_rng(42)
        x=pd.DataFrame(rng.normal(size=(8,20)),index=[f'g{i}' for i in range(8)])
        d=pd.DataFrame(np.where(x>.4,1,np.where(x<-.4,-1,0)),index=x.index)
        for configure in (lambda m:m.setSequential(),lambda m:m.setJoblib(n_jobs=2)):
            result,_,_=configure(HLicorn(max_coreg=2,nGRN=2)).fit(x,list(x.index[:5]),d,collect=False)
            self.assertTrue(result[['coact','corep']].map(lambda value:isinstance(value,list)).to_numpy().all())

if __name__=='__main__':unittest.main()
