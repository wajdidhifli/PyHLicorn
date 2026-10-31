from __future__ import annotations
from typing import Literal, Optional
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, delayed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from ..pre_inference import PreProcess, FrequentItemsets , utils
from .oneGeneHlicorn import oneGeneHlicorn
from .._serialization import normalize_frame





def hlicorn(
    # Input Paramerters
    numerical_expression: pd.DataFrame,
    tf_list: pd.DataFrame | pd.Series | list,
    discrete_expression: pd.DataFrame = None,
    gene_list: pd.DataFrame = None,
    # inferenceParameters
    min_gene_support: float = 0.1,
    min_coreg_support: float = 0.1,
    max_coreg: int = 3,
    search_thresh: float = 1 / 3,
    nGRN: int = 100,
    # Rules
    joblib : tuple = None,
    cluster: SparkSession = None,
    linear_model: Optional[Literal['cvlm', 'lm', None]] = 'lm',
    verbose: Optional[Literal[True, False]] = False,
    collect = True,
    cv_folds = None,
    random_state = 0,
    mining_options = None,
    spark_partitions = None,
    max_candidate_pairs = None,
    memory_budget = None
):
        
    # Generate/validate folds once; reuse across every target and candidate.
    if linear_model == 'cvlm':
        from ..during_inference.__LinearModel import make_cv_folds
        cv_folds = make_cv_folds(numerical_expression.shape[1], cv_folds, random_state)
    ######### INPUT MINGING #################################################################################
    
    # Ensure indices of numerical_expression and discrete_expression are of type str
    if not all(isinstance(index, str) for index in numerical_expression.index):
        raise ValueError("All indices of numerical_expression must be of type str, The Gene names should be the index of the Dataframe")
    
    # Normalize tf_list to a flat list of strings
    if isinstance(tf_list, pd.DataFrame):
        if tf_list.shape[1] > 0:
            tf_list = tf_list.iloc[:, 0].dropna().tolist()
        else:
            tf_list = tf_list.index.dropna().tolist()
    elif isinstance(tf_list, (pd.Series, pd.Index)):
        tf_list = tf_list.dropna().tolist()
    elif isinstance(tf_list, (list, tuple, set)):
        tf_list = list(tf_list)
    else:
        raise TypeError("tf_list must be a list, tuple, set, Series, or DataFrame.")
    tf_list = [str(tf) for tf in tf_list]

    if not numerical_expression.index.is_unique or not numerical_expression.columns.is_unique:
        raise ValueError("Expression gene and sample identifiers must be unique.")
    if numerical_expression.empty:
        raise ValueError("Numerical expression must be nonempty and finite.")
    block_rows = max(1, (32*1024*1024)//(8*numerical_expression.shape[1]))
    for start in range(0, len(numerical_expression), block_rows):
        if not np.isfinite(numerical_expression.iloc[start:start+block_rows].to_numpy(dtype=float)).all():
            raise ValueError("Numerical expression must be nonempty and finite.")
    if gene_list is not None:
        if isinstance(gene_list, pd.DataFrame):
            if gene_list.shape[1] > 0:
                gene_list = gene_list.iloc[:, 0].dropna().tolist()
            else:
                gene_list = gene_list.index.dropna().tolist()
        elif isinstance(gene_list, (pd.Series, pd.Index)):
            gene_list = gene_list.dropna().tolist()
        elif isinstance(gene_list, (list, tuple, set)):
            gene_list = list(gene_list)
        gene_list = [str(g) for g in gene_list]

    dis_row_names = None
    if discrete_expression is None:
        discrete_expression = PreProcess.discretize_gene_expression(numerical_expression)
    

    if (not numerical_expression.index.equals(discrete_expression.index)
            or not numerical_expression.columns.equals(discrete_expression.columns)):
        raise ValueError("Numerical and discrete expression must have identical ordered gene and sample labels.")
    for start in range(0, len(discrete_expression), block_rows):
        if not np.isin(discrete_expression.iloc[start:start+block_rows].to_numpy(), [-1,0,1]).all():
            raise ValueError("Discrete expression must contain only -1, 0, and 1, with no missing values.")
    # Discrete states need one byte, rather than int64/float64 internal copies.
    discrete_expression = discrete_expression.astype(np.int8, copy=False)
    if not 0 <= search_thresh <= 1:
        raise ValueError("search_thresh must be between 0 and 1.")

    if not discrete_expression.empty:
        dis_row_names = discrete_expression.index.tolist()

    num_row_names = None
    if gene_list is None:
        num_row_names = numerical_expression.index.tolist()
        gene_list = list(set(num_row_names) - set(tf_list))
    else:
        num_row_names = numerical_expression.index.tolist()

    ######### INPUT VERIFICATIONS #################################################################################

    if not all(isinstance(index, str) for index in discrete_expression.index):
        raise ValueError("All indices of discrete_expression must be of type str. The Gene names should be the index of the Dataframe")
    
    # Checking for gene duplicates in numerical_expression
    if len(num_row_names) > len(set(num_row_names)):
        raise ValueError("No gene duplicates are allowed in the row names.")

    num_row_names_sort = num_row_names.copy()
    dis_row_names_sort = dis_row_names.copy()

    if numerical_expression.shape[0] != discrete_expression.shape[0] or sorted(num_row_names_sort) != sorted(dis_row_names_sort):
        raise ValueError("The Discrete expression and Numerical expression data should have the same dimensions")

    if sorted(num_row_names_sort) != sorted(dis_row_names_sort):
        raise ValueError("The Discrete expression and Numerical expression data should have the same rownames (gene/tf names)")
    
    if len(set(num_row_names).intersection(set(tf_list))) <= 1:
        raise ValueError("At least 2 of the provided regulators/transcription factor (tf_list) should be in the rownames in the gene expression matrix")

    if len(set(num_row_names).intersection(set(gene_list))) == 0:
        raise ValueError("The list of genes (gene_list) should be in the rownames in the gene expression matrix")

    if not all(0 <= support <= 1 for support in (min_gene_support, min_coreg_support, search_thresh)):
        raise ValueError("All parameters must be floats between 0 and 1.")

    # Verbose is a parameter used to print messages about the steps being performed
    if verbose:
        print("Pre-process.")

    # Identify indices of genes expressed in more conditions than the threshold based on minimum support
    active_counts = np.empty(len(discrete_expression), dtype=np.int64)
    for start in range(0, len(discrete_expression), block_rows):
        active_counts[start:start+block_rows] = np.count_nonzero(
            discrete_expression.iloc[start:start+block_rows].to_numpy(), axis=1)
    genes_support = np.flatnonzero(active_counts > numerical_expression.shape[1]*min_gene_support)
    
    # Filter dis & num expressions based on genes support
    discrete_expression = discrete_expression.iloc[genes_support]
    numerical_expression = numerical_expression.iloc[genes_support]

    tf_list = list(set(numerical_expression.index).intersection(tf_list))
    gene_list = list(set(numerical_expression.index).intersection(gene_list))

    if max_coreg is None:
        max_coreg = len(tf_list)
        warnings.warn(
            "max_coreg is not specified. Setting max_coreg to the number of transcription factors (TFs): {}".format(max_coreg),
            UserWarning
        )
    
    if max_coreg < 1:
        raise ValueError("max_coreg must be at least 1.")
    elif max_coreg > 10:
        warnings.warn(
            f"max_coreg is set to {max_coreg}, which is high (>10). This allows up to {max_coreg * 2} regulators per gene, which is biologically unlikely. "
            f"Consider lowering max_coreg to reflect more realistic regulatory scenarios.",
            UserWarning
        )
    
    # Input verifications post genes selection
    if len(tf_list) < 5:
        raise ValueError("Less than 5 of the provided TF are suitable to infer a network. Either provide more TF, more variations in the discrete dataset (more 1 or -1) or decrease the min_gene_support parameter to select more but less variant TFs.")

    if len(gene_list) == 0:
        raise ValueError("No genes were suitable to infer regulators. Either provide more variations in the discrete dataset (more 1 or -1) or decrease the min_gene_support parameter to allow the selection of more but less variant Genes.")

    # Getting DFs for target and regulator genes expression both dis & num
    gene_num_exp = numerical_expression[numerical_expression.index.isin(gene_list)]
    gene_disc_exp = discrete_expression[discrete_expression.index.isin(gene_list)]

    reg_num_exp = numerical_expression[numerical_expression.index.isin(tf_list)]
    reg_disc_exp = discrete_expression[discrete_expression.index.isin(tf_list)]

    # Build only the final boolean signed-transaction matrix. Avoid the two
    # int64 transformed DataFrames and their int64 concatenation.
    reg_values = reg_disc_exp.to_numpy(copy=False)
    bit_values = np.empty((2*reg_values.shape[1], reg_values.shape[0]), dtype=bool)
    np.equal(reg_values.T, 1, out=bit_values[:reg_values.shape[1]])
    np.equal(reg_values.T, -1, out=bit_values[reg_values.shape[1]:])
    trans_reg_bit_data = pd.DataFrame(bit_values, columns=reg_disc_exp.index)

    if verbose:
        print("Mining coregulator ...")

    memory_plan = None
    input_bytes = 0
    if memory_budget is not None:
        input_bytes = int(numerical_expression.memory_usage(index=True, deep=True).sum()
                          + discrete_expression.memory_usage(index=True, deep=True).sum())
        from .memory import parse_memory_budget
        memory_budget = parse_memory_budget(memory_budget)
        if cluster is not None:
            raise ValueError('memory_budget is local-only; configure Spark executor memory separately')
        mining_options = dict(mining_options or {})
        if mining_options.get('algorithm', 'vertical') != 'vertical':
            raise ValueError('memory_budget requires the streaming vertical miner')
        limit = max(1, memory_budget // 10)
        explicit = mining_options.get('spill_threshold_bytes')
        mining_options['spill_threshold_bytes'] = limit if explicit is None else min(explicit, limit)

    ######### MINING FOR FREQUENT coregs USING APRIORI #################################################################################
    
    items = FrequentItemsets.get_coregs(
        data = trans_reg_bit_data,
        min_gene_support = min_gene_support,
        min_coreg_support = min_coreg_support,
        max_coreg = max_coreg,
        **(mining_options or {}),
        )
        
    retain_candidates = False
    try:
        if memory_budget is not None:
            from .memory import plan_memory
            memory_plan = plan_memory(memory_budget, input_bytes, len(items), numerical_expression.shape[1],
                                      max_coreg, joblib[0] if joblib is not None and joblib[1] != 'sequential' else 1,
                                      items.buffer_bytes if items.diagnostics.get('storage') == 'memory' else 0)
            if joblib is not None:
                joblib = (memory_plan['effective_workers'], joblib[1])
            if verbose:
                print(f"Memory budget: {memory_budget / 1024**3:.2f} GiB; "
                      f"{memory_plan['effective_workers']} local worker(s); best-effort, not a hard cap.")
        if verbose:
            message = (
                f"\n"
                f"Gene Regulatory Network Search Summary:\n"
                f"  • Target genes: {len(gene_list)}\n"
                f"  • Regulators (TFs): {len(tf_list)}\n"
                f"  • Potential co-regulator sets: {len(items)}\n\n"
                f"Search Parameters:\n"
                f"  • Maximum size of co-regulator sets: {max_coreg}\n"
                f"  • Number of putative GRNs per gene: {nGRN}\n"
                f"  • Minimum DE samples per gene: {min_gene_support}\n"
                f"  • Minimum DE samples per co-regulator set: {min_coreg_support}\n"
            )
            print(message)
                
        #########   PreProcessing    #################################################################################
        def reindextoNumpy(exp : pd.DataFrame, index = pd.DataFrame.index)->np.ndarray:
            return exp.reindex(index).to_numpy()
        
        # The One Gene Algrothim is applied on the gene list which is the index of gene_disc_exp
        gene_list = list(gene_disc_exp.index)
        gene_num_exp = reindextoNumpy(gene_num_exp,gene_list)
        gene_disc_exp = gene_disc_exp.to_numpy()
        
        # The reg_num_exp and reg_disc_exp indexs are tf_list which is the same as trans_reg_bit_data.columns
        potential_coregs = trans_reg_bit_data.columns
        reg_num_exp = reindextoNumpy(reg_num_exp,potential_coregs).T # NOTE : SHOUDL BE Trasnposed
        reg_disc_exp = reindextoNumpy(reg_disc_exp,potential_coregs)

        # Finalizing the datas
        trans_reg_bit_data = trans_reg_bit_data.astype(bool)
        trans_reg_bit_data = trans_reg_bit_data.to_numpy()
        
        got_net = False
        # Execute once at the requested threshold, including the valid endpoints.
        while got_net is False:
            if len(gene_list) >= 1:  

                if joblib is not None:
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=DeprecationWarning)
                        
                # Reduce data specific to each gene
                gene_exp_tups = utils.gene_specific_generator(gene_list, gene_disc_exp, gene_num_exp)

                if joblib is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=DeprecationWarning)
                        
                        low_memory = False # Currently off
                        if low_memory:
                            reg_disc_exp = utils.load_shared_memmap( * utils.sharedMemory(reg_disc_exp, "reg_disc_exp"))
                            trans_reg_bit_data = utils.load_shared_memmap( * utils.sharedMemory(trans_reg_bit_data, "trans_reg_bit_data"))
                            reg_num_exp = utils.load_shared_memmap( * utils.sharedMemory(reg_num_exp, "reg_num_exp"))
                        
                        def oneGenewrapper(gene, gene_disc_arr , Y):
                            return oneGeneHlicorn(
                                gene,
                                gene_disc_arr,  
                                reg_disc_exp,
                                items,
                                trans_reg_bit_data,
                                search_thresh,
                                reg_num_exp,
                                Y,
                                nGRN,
                                linear_model,
                                convert = False,
                                cv_folds=cv_folds,
                                max_candidate_pairs=max_candidate_pairs,
                            )
                            
                        results = Parallel(*joblib, max_nbytes='1M' if memory_budget is not None else None,
                                           pre_dispatch=memory_plan['effective_workers'] if memory_plan else '2*n_jobs',
                                           batch_size=1 if memory_plan else 'auto')(
                            delayed(oneGenewrapper)(
                                gene,
                                gene_disc_arr,  
                                Y,
                            ) for gene, gene_disc_arr, Y in tqdm(gene_exp_tups, total=len(gene_list), desc="Processing Genes ", unit=' gene', unit_scale=True, leave=True,ncols=150))
                        
                        results = pd.concat(results,ignore_index=True)
                        results = results.replace(np.nan,pd.NA)
                    got_net = True
                
                elif cluster is not None:
                    from .spark_backend import infer_spark, release_spark_resources
                    state = (reg_disc_exp, items, trans_reg_bit_data, reg_num_exp,
                             search_thresh, nGRN, linear_model, cv_folds, max_candidate_pairs)
                    df_final = infer_spark(cluster, list(gene_exp_tups), state, spark_partitions)
                    if collect:
                        try:
                            results = normalize_frame(df_final.toPandas())
                        finally:
                            release_spark_resources(df_final)
                    else:
                        retain_candidates = True
                        return df_final, {'gene_list': len(gene_list), 'tf_list': len(tf_list),
                                          'transcription_factors': None, 'target_genes': None,
                                          'mining': items.diagnostics, 'memory_plan': memory_plan}
                    got_net = True
                    
                        
                else:
                    results = []
                    for gene, gene_disc_arr, Y in tqdm(gene_exp_tups, desc="Processing Genes", unit='gene', unit_scale=True, leave=True):
                        result = oneGeneHlicorn(
                            gene,
                            gene_disc_arr,
                            reg_disc_exp,
                            items,
                            trans_reg_bit_data,
                            search_thresh,
                            reg_num_exp,
                            Y,
                            nGRN,
                            linear_model,
                            convert=False,
                            cv_folds=cv_folds,
                            max_candidate_pairs=max_candidate_pairs,
                        )
                        results.append(result)

                    results = pd.concat(results, ignore_index=True)
                    results = results.replace(np.nan, pd.NA)
                    got_net = True




        return normalize_frame(results), {'gene_list' : len(gene_list), 'tf_list': len(tf_list), 'transcription_factors' : None, 'target_genes' : None, 'mining': items.diagnostics, 'memory_plan': memory_plan}


    finally:
        if not retain_candidates and hasattr(items, 'close'):
            items.close()
