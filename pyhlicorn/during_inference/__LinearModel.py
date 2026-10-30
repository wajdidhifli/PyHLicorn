import pandas as pd
import numpy as np

def __check(reg,nreg,reg_exp):

    if nreg > 0:
        reg = np.array(reg)
        arr_reg = reg_exp[:,reg]
        if nreg > 1:
            arr_reg = np.concatenate((arr_reg, np.prod(arr_reg, axis=1).reshape(-1, 1),),axis=1)
            nreg = nreg + 1
    else:
        arr_reg = None
        nreg = 0
        
    return arr_reg , nreg

def __numpy_linreg(X: np.ndarray, y: np.ndarray):
    # Add intercept column
    X_ = np.hstack([np.ones((X.shape[0], 1)), X])

    # Closed-form solution: beta = (XᵀX)^-1 Xᵀy
    beta = np.linalg.lstsq(X_, y, rcond=None)[0]

    y_pred = X_ @ beta

    # Metrics
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(ss_res / len(y))

    coefs = beta[1:]  # exclude intercept
    return coefs, r2, rmse

def Linear_Regression(gact,grep,grn,reg_exp,Y,linear_model,cv_folds=None):
    
    for act,rep, iresults in zip(gact,grep,grn):
        
        nact = len(act)
        nrep = len(rep)
        
        arr_act , nact = __check(act,nact,reg_exp)
        arr_rep , nrep = __check(rep,nrep,reg_exp)
        
        if arr_act is not None and arr_rep is not None :
            arr = np.hstack((arr_act, arr_rep))
        elif arr_act is not None:
            arr = arr_act
        elif arr_rep is not None:
            arr = arr_rep
                            
        if linear_model == "lm":
            coefs, r2, rmse = __numpy_linreg(arr,Y)
        elif linear_model == "cvlm":
            coefs, r2, rmse = __cvlm_numpy_linreg(arr,Y,fold_ids=cv_folds)
       
            

        iresults[4] = float(r2)
        iresults[5] = float(rmse)

        coefs = coefs.astype(float)
        
        if nact == 0:
            iresults[0] = pd.NA
        elif nact == 1:
            iresults[0] = [float(coefs[0])]
        else:
            iresults[0] = coefs[0:nact-1].tolist()
            iresults[2] = float(coefs[nact-1])
            
        coefs = np.array(coefs[nact:],dtype=float)
        coefs = coefs.astype(float)
        
        if nrep == 0:
            iresults[1] = pd.NA
        elif nrep == 1:
            iresults[1] = [float(coefs[0])]
        else:
            iresults[1] = coefs[0:nrep-1].tolist()
            iresults[3] = float(coefs[nrep-1])

    return grn


def make_cv_folds(n_samples, fold_ids=None, random_state=0, k=10):
    """Return sample-aligned folds shared by all models; do not mutate global RNG."""
    if fold_ids is None:
        if n_samples < k:
            raise ValueError("Ten-fold CV requires at least ten samples; supply explicit folds otherwise.")
        rng = np.random.default_rng(random_state)
        folds = np.empty(n_samples, dtype=int)
        for fold, positions in enumerate(np.array_split(rng.permutation(n_samples), k)):
            folds[positions] = fold
    else:
        folds = np.asarray(fold_ids)
        if folds.shape != (n_samples,) or not np.issubdtype(folds.dtype, np.integer):
            raise ValueError("cv_folds must contain one integer fold label per sample.")
    labels = np.unique(folds)
    if len(labels) < 2 or len(labels) > n_samples:
        raise ValueError("At least two nonempty folds are required.")
    return folds.copy()


def __cvlm_numpy_linreg(X, y, k=10, seed=0, fold_ids=None):
    folds = make_cv_folds(len(y), fold_ids, seed, k)
    fitted = np.empty(len(y), dtype=float)
    coefs_list = []
    for fold in np.unique(folds):
        val = folds == fold
        X_train = np.column_stack([np.ones((~val).sum()), X[~val]])
        X_val = np.column_stack([np.ones(val.sum()), X[val]])
        beta = np.linalg.lstsq(X_train, y[~val], rcond=None)[0]
        fitted[val] = X_val @ beta
        coefs_list.append(beta[1:])
    if np.std(y) == 0 or np.std(fitted) == 0:
        r2 = np.nan  # correlation is undefined, not evidence of zero performance
    else:
        r2 = np.corrcoef(y, fitted)[0, 1] ** 2
    return np.mean(coefs_list, axis=0), r2, np.sqrt(np.mean((y-fitted)**2))
