# Import Libaries
import numpy as np
import pandas as pd
# Algorithm
from scipy.stats import rankdata

from pyhlicorn.during_inference import (__eand, __comblicorn, Linear_Regression)


def get_regs(reg_arr,indices,ncarr,search_thresh):
    item_rows = reg_arr[:,indices]
    # true_row = np.all(item_rows, axis=1).sum()
    # true_row = np.bitwise_and.reduce(item_rows, axis=1).sum()
    true_row = np.count_nonzero(np.bitwise_and.reduce(item_rows, axis=1))


    support =  true_row / ncarr
    if support >= search_thresh:
        return indices

def oneGeneHlicorn(
    gene,
    gene_disc_arr,
    reg_disc_exp,
    items,
    trans_reg_bit_data,
    search_thresh,
    reg_num_exp,
    Y,
    nGRN  ,
    linear_model,
    convert,
    cv_folds=None,
    max_candidate_pairs=5_000_000,
    ):
    
    if convert:
        gene_disc_arr = np.asarray(gene_disc_arr)
        reg_disc_exp = np.asarray(reg_disc_exp)
        trans_reg_bit_data = np.asarray(trans_reg_bit_data)
        reg_num_exp = np.asarray(reg_num_exp)
        Y = np.asarray(Y)

    

    # Number of instances/columns
    shift = len(gene_disc_arr)

    pos = np.where(gene_disc_arr == 1)[0]
    neg = np.where(gene_disc_arr == -1)[0] + shift

    ncarr = len(pos) + len(neg)

    co_act_arr = trans_reg_bit_data[np.concatenate((pos, neg))]

    pos = pos + shift
    neg = neg - shift
    
    co_rep_arr = trans_reg_bit_data[np.concatenate((pos, neg))]
    
    co_act = []
    co_rep = []
    
    co_act_exp = []
    co_rep_exp = []
    
    
    for tups in items:
        indices = tups[0]
        
        icoact = get_regs(co_act_arr,indices,ncarr,search_thresh)
        icorep = get_regs(co_rep_arr,indices,ncarr,search_thresh)

        if icoact is not None:
            actexp = __eand(icoact,reg_disc_exp)
            co_act.append(tups)
            co_act_exp.append(actexp)
        
        if icorep is not None:
            repexp = __eand(icorep,reg_disc_exp)
            co_rep.append(tups)
            co_rep_exp.append(repexp)

    pairs = (len(co_act) + 1) * (len(co_rep) + 1)
    if max_candidate_pairs is not None and pairs > max_candidate_pairs:
        raise MemoryError(f"Target {gene}: {pairs:,} candidate pairs exceeds "
                          f"max_candidate_pairs={max_candidate_pairs:,}. "
                          "No candidates were silently discarded.")

    # This is for the empty string at the end of the co_act & co_rep list
    last = np.zeros(reg_disc_exp.shape[1])
        
    co_act.append("")
    co_act_exp.append(last)
    co_act_exp = np.vstack(co_act_exp)
    co_act_exp = co_act_exp.flatten()
    
    co_rep.append("")
    co_rep_exp.append(last)
    co_rep_exp = np.vstack(co_rep_exp)
    co_rep_exp = co_rep_exp.flatten()
    
    co_rep_exp[np.where(co_rep_exp==1)] = 2 

    mae, iact, irep = __comblicorn(co_act_exp, co_act, co_rep_exp, co_rep, gene_disc_arr)

    iact = (iact - 1)
    irep = (irep - 1)
    
    co_act = np.array(co_act,dtype=object)
    co_rep = np.array(co_rep,dtype=object)

    # A regulator cannot activate and repress the same target model.
    # Equality alone misses partially overlapping regulator sets.
    good_index = np.array([
        i for i, (a, r) in enumerate(zip(iact, irep))
        if (co_act[a] != "" or co_rep[r] != "")
        and not (set(co_act[a][0] if co_act[a] != "" else ())
                 & set(co_rep[r][0] if co_rep[r] != "" else ()))
    ], dtype=int)
    
    sel_act = co_act[iact[good_index]]
    sel_rep = co_rep[irep[good_index]]
    mae = mae[good_index]        
    
    # Rank The data based on MAE
    rank = rankdata(mae, method='min')

    if nGRN != None:
        best_index =np.where(rank <=nGRN)[0]  
    else:
        best_index = np.where(rank == 1)[0]
    
    # gact and grep are the alternative to co_act_name & co_rep_name
    # they include the items' index as well as the item's names
    gact = sel_act[best_index]
    grep = sel_rep[best_index]
    
    

    # The mask is used to allow the gact & grep to be unzipped
    # EX: gact = [(29,A2ML1),("","")] -> gact = [29,""], co_act_name = [A2ML1,""]
    mask = np.empty(len(best_index), dtype=object)  
    mask[:] = [("", pd.NA)] * len(best_index)     
    
    co_act_name = []
    if len(gact) > 0:
        gact = np.array(gact, dtype=object)        
        gact = np.where(gact == "", mask, gact)
        gact, co_act_name = zip(*gact)

    co_rep_name = []
    if len(grep) > 0:
        grep = np.array(grep, dtype=object)
        grep = np.where(grep == "", mask, grep)
        grep, co_rep_name = zip(*grep)




    if linear_model == None:
        grn =  pd.DataFrame({
            "Target": np.full(len(best_index),gene),
            "coact": co_act_name,
            "corep": co_rep_name,
            })
        
        
        
            
        return grn
    else:
        grn = np.full((len(best_index),6),pd.NA,dtype=object)
        grn = Linear_Regression(gact,grep,grn,reg_num_exp,Y,linear_model,cv_folds=cv_folds)
        
        
        grn = pd.DataFrame.from_records(grn, columns=["Coef_Acts", "Coef_Reps", "Coef_coActs", "Coef_coReps", "R2", "RMSE"])
        grn.insert(0, "Target", gene)
        grn.insert(1, "coact", co_act_name)
        grn.insert(2, "corep", co_rep_name)
        if not grn.empty:
            grn = grn.sort_values(by=['R2', 'RMSE'], ascending=[False, True]) 

        return grn 


    
