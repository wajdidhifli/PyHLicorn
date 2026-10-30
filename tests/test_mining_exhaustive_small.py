"""Independent exhaustive closedness oracle, with and without transaction masks."""
import itertools
import unittest
import numpy as np
from pyhlicorn.pre_inference.vertical_mining import bitsets_from_matrix, row_bitsets, closed_sets, bounded_closed_sets

class ExhaustiveMiningTests(unittest.TestCase):
    def test_all_four_transaction_three_attribute_databases(self):
        groups=[tuple(c) for k in range(1,4) for c in itertools.combinations(range(3),k)]
        for encoding in range(1<<12):
            matrix=np.array([(encoding>>bit)&1 for bit in range(12)],dtype=bool).reshape(4,3)
            transactions=[set(np.flatnonzero(row)) for row in matrix]
            oracle=[]
            supports=[]
            for group in groups:
                matching=[row for row in transactions if set(group)<=row]
                if matching: supports.append((group,len(matching)))
                if matching and set.intersection(*matching)==set(group):
                    oracle.append((group,len(matching)))
            columns=bitsets_from_matrix(matrix)
            for minimum in range(1,5):
                ids=[i for i,c in enumerate(columns) if c and c.bit_count()>=minimum]
                masks=row_bitsets(matrix,ids)
                for cap in range(1,4):
                    expected={(g,n) for g,n in oracle if len(g)<=cap and n>=minimum}
                    for rows in (None,masks):
                        actual=list(closed_sets(columns,4,minimum,1,cap,row_masks=rows))
                        self.assertEqual(set(actual),expected,(encoding,minimum,cap,rows is not None))
                        self.assertEqual(len(actual),len(set(actual)))
                        bounded_expected = {(g,n) for g,n in supports
                                            if n>=minimum and (len(g)==cap or (len(g)<cap and (g,n) in oracle))}
                        bounded_actual = list(bounded_closed_sets(columns,4,minimum,1,cap,row_masks=rows))
                        self.assertEqual(set(bounded_actual),bounded_expected,(encoding,minimum,cap))
                        self.assertEqual(len(bounded_actual),len(set(bounded_actual)))
