from __future__ import annotations
from typing import Optional, Union, Tuple, List, Literal, Dict, Any ,overload

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame as SparkDataFrame

from .hlicorn import hlicorn 
from .network import Network

def _set_thresholds(min_gene_support, min_coreg_support, max_coreg, search_thresh, nGRN):
    if not all(0 <= support <= 1 for support in (min_gene_support, min_coreg_support, search_thresh)):
        raise ValueError("All parameters must be floats between 0 and 1.")
    return {
        'min_gene_support': min_gene_support,
        'min_coreg_support': min_coreg_support,
        'max_coreg': max_coreg,
        'search_thresh': search_thresh,
        'nGRN': nGRN
    }

class HLicorn:
    """
    Hybrid Learning of Co-operative Regulation Network.

    This function infers a gene regulatory network using hybrid learning techniques by analyzing gene expression data 
    and transcription factors.
    
    Parameters
    ----------
    min_gene_support : float, optional, default: 0.1
        A value between 0 and 1 indicating the minimum proportion of samples 
        in which a gene must have a non-zero discretized expression value 
        to be considered for network inference.

    min_coreg_support : float, optional, default: 0.1
        A value between 0 and 1 indicating the minimum proportion of samples 
        in which a set of co-regulators must share the same non-zero discretized 
        expression value to be considered a potential set of co-regulators.

    max_coreg : int, optional, default: None
        The maximum size of co-regulator groups to consider. 
        Defaults to the number of TFs. Can be reduced to save memory.

    search_thresh : float, optional, default: 1/3
        A value between 0 and 1 specifying the minimum proportion of samples 
        in which a gene and a set of co-regulators must share the same non-zero 
        discretized expression value to be considered co-regulators of the gene.

    nGRN : int, optional, default: 100
        The number of gene regulatory networks to infer.

    memory_budget : str or int, optional
        Opt-in local working-memory planning, e.g. "16GB" or integer bytes.
        Enables candidate spill and conservative worker scheduling; not a hard RSS cap.

    Returns
    -------
    HLicorn class object

    """
    
    def __init__(
        self,
        *,
        min_gene_support: float = 0.1,
        min_coreg_support: float = 0.1,
        max_coreg: int = 5,
        search_thresh: float = 1/3,
        nGRN: int = 100,
        verbose : bool = False,
        max_candidate_pairs: int | None = None,
        memory_budget: str | int | None = None
    ): 
        
        self._thresholds = _set_thresholds(
            min_gene_support,
            min_coreg_support,
            max_coreg,
            search_thresh,
            nGRN
        )
        
        # Memory-first default: parallel workers are explicitly opt-in.
        self._parallel = None
        self._cluster = None    
        self._lmodel = 'lm'
        self._cv_folds = None
        self._random_state = 0
        self._verbose = verbose
        if max_candidate_pairs is not None and (not isinstance(max_candidate_pairs, int) or max_candidate_pairs < 1):
            raise ValueError('max_candidate_pairs must be a positive integer or None')
        self._max_candidate_pairs = max_candidate_pairs
        from .memory import parse_memory_budget
        self._memory_budget = parse_memory_budget(memory_budget)
        self._mining_options = {}
        self._spark_partitions = None
        
    def setMining(self, *, algorithm='vertical', memory_limit_gb=None,
                  timeout_seconds=None, max_itemsets=None, temp_dir=None,
                  chunk_bytes=1024*1024, spill_threshold_bytes=None) -> 'HLicorn':
        """Configure exact closed mining. Limits abort; they never truncate.

        memory_limit_gb is a best-effort child-process RSS budget in GiB,
        sampled every 100 ms (requires psutil). It is not a whole-job cap.
        vertical is an exact streaming miner with adaptive RAM/disk candidates. It checks
        max_itemsets incrementally and timeout_seconds cooperatively. Legacy PyFIM
        limits are checked after native calls. temp_dir must have sufficient disk space.
        spill_threshold_bytes budgets compact candidate payload (default None: RAM only).
        Set zero to force disk storage. It is not a whole-job RSS cap.
        chunk_bytes bounds spilled candidate reads, not total inference RAM.
        """
        self._mining_options = dict(algorithm=algorithm, memory_limit_gb=memory_limit_gb,
                                    timeout_seconds=timeout_seconds, max_itemsets=max_itemsets,
                                    temp_dir=temp_dir, chunk_bytes=chunk_bytes,
                                    spill_threshold_bytes=spill_threshold_bytes)
        return self

    def setJoblib(self, n_jobs: int = -1, backend: Optional[Literal['loky', 'multiprocessing', 'sequential', 'threading']] = 'loky') -> 'HLicorn':
        """
        Configure the joblib parallelization settings for the HLicorn instance.

        Parameters
        ----------
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is -1, which means using all available processors.
        backend : {'loky', 'multiprocessing', 'sequential', 'threading'}, optional
            The parallelization backend to use. Default is 'loky'.

        Returns
        -------
        HLicorn
            The current instance of HLicorn with updated parallelization settings.
        """
        self._parallel = (n_jobs, backend)
        self._cluster = None
        return self
    
    def setSpark(self, cluster: SparkSession, *, partitions: int | None = None) -> 'HLicorn':
        """
        Configures the Spark session for the HLicorn instance.

        Parameters
        ----------
        cluster : SparkSession
            The Spark session to be associated with the HLicorn instance.
        partitions : int, optional
            Number of target partitions. Collection is controlled by fit(collect=...).

        Returns
        -------
        HLicorn
            The current instance of the object with default settings applied.
        """
        if partitions is not None and (not isinstance(partitions, int) or partitions < 1):
            raise ValueError('partitions must be a positive integer or None')
        self._spark_partitions = partitions
        self._cluster = cluster
        self._parallel = None
        return self
    
    def setSequential(self)-> 'HLicorn':
        """
        Inferring inference in a for loop

        Returns
        -------
        HLicorn
            The current instance of the object with default settings applied.
        """
        self._cluster = None
        self._parallel = None
        return self
    
    
    def setLinearModel(self, model: Optional[Literal['cvlm', 'lm', None]] = 'lm', *, cv_folds=None, random_state=0) -> 'HLicorn':
        """
        Sets the linear model for the HLicorn instance.

        Parameters
        ----------
        model : {'cvlm', 'lm', None}, optional
            The type of linear model to set. Options are:
            - 'cvlm': Cross-validated linear model.
            - 'lm': Standard linear model.
            - None: Gene Regulatory Network without coefficients
            Default is 'lm'.
        cv_folds : sequence of int, optional
            One fold label per sample, reused for every target and candidate.
        random_state : int or None, default: 0
            Seed for default ten-fold generation; explicit folds take precedence.

        Returns
        -------
        HLicorn
            The instance of HLicorn with the updated linear model.
        """
        if model not in ('lm', 'cvlm', None):
            raise ValueError("model must be 'lm', 'cvlm', or None")
        self._lmodel = model
        self._cv_folds = cv_folds
        self._random_state = random_state
        return self
    
    
    @overload
    def fit(
        self,
        numerical_expression: pd.DataFrame,
        tf_list: Union[pd.DataFrame, List],
        discrete_expression: Optional[pd.DataFrame] = None,
        gene_list: Optional[Union[pd.DataFrame, List]] = None,
        *,
        collect: Literal[False]
    ) -> Tuple[Union[SparkDataFrame, pd.DataFrame], Dict, Dict]: ...

    @overload
    def fit(
        self,
        numerical_expression: pd.DataFrame,
        tf_list: Union[pd.DataFrame, List],
        discrete_expression: Optional[pd.DataFrame] = None,
        gene_list: Optional[Union[pd.DataFrame, List]] = None,
        *,
        collect: Literal[True] = True
    ) -> Network: ...

    def fit(
        self,
        numerical_expression: pd.DataFrame,
        tf_list: Union[pd.DataFrame, List],
        discrete_expression: Optional[pd.DataFrame] = None,
        gene_list: Optional[Union[pd.DataFrame, List]] = None,
        *,
        collect: bool = True
    ) -> Union[Network, Tuple[Union[SparkDataFrame, pd.DataFrame], Dict, Dict]]:
        """
        Fits a network model using the provided expression data and parameters.
        
        Parameters
        ----------
        numerical_expression : pandas.DataFrame
            A numerical matrix containing the expression levels of genes and 
            transcription factors. This will be used to infer the network.
        
        tf_list : pandas.DataFrame or list
            A list of genes designated as transcription factors (TFs) or regulators. 
            This should contain gene names.
            
        discrete_expression : pandas.DataFrame, optional
            A matrix with the same dimensions, columns, and indices as `numerical_expression`. 
            This should only contain values `{-1, 0, 1}`, where:
            
            - -1 indicates under-expression,
            - 0 indicates no change, and 
            - 1 indicates over-expression.
        
        gene_list : pandas.DataFrame or list, optional
            A list of genes for which regulatory networks should be inferred. 
            These genes should appear in the row names of the expression data. 
            If not provided, all genes in the row names of the expression data 
            that are not in `tf_list` will be used.
        
        collect : bool, optional, default: True
            A flag indicating whether to collect results to the driver node. 
            If False, returns the raw inference result (only for Spark).

        Returns
        -------
        Network or tuple
            If `collect` is True, returns a `Network` object containing the inferred network, 
            metadata, and inference parameters.
            If `collect` is False, returns a tuple of (inference_result, metadata, thresholds)
            where inference_result is either a Spark DataFrame or pandas DataFrame.
        """
        

        if self._memory_budget is not None and self._cluster is not None:
            raise ValueError('memory_budget currently supports local sequential/joblib execution; '
                             'Spark requires executor-specific memory configuration')
        mining_options = self._mining_options.copy()
        if self._memory_budget is not None:
            if mining_options.get('algorithm', 'vertical') != 'vertical':
                raise ValueError('memory_budget requires the streaming vertical miner')
            candidate_budget = max(1, self._memory_budget // 10)
            explicit = mining_options.get('spill_threshold_bytes')
            mining_options['spill_threshold_bytes'] = candidate_budget if explicit is None else min(explicit, candidate_budget)

        inferenceResult, metadata = hlicorn(
            numerical_expression=numerical_expression,
            tf_list=tf_list,
            discrete_expression=discrete_expression,
            gene_list=gene_list,
            min_gene_support=self._thresholds['min_gene_support'],
            min_coreg_support=self._thresholds['min_coreg_support'],
            max_coreg=self._thresholds['max_coreg'],
            search_thresh=self._thresholds['search_thresh'],
            nGRN=self._thresholds['nGRN'],
            joblib=self._parallel,            
            cluster=self._cluster,
            linear_model=self._lmodel,
            cv_folds=self._cv_folds,
            random_state=self._random_state,
            verbose=self._verbose,
            mining_options=mining_options,
            memory_budget=self._memory_budget,
            spark_partitions=self._spark_partitions,
            max_candidate_pairs=self._max_candidate_pairs,
            collect = collect
        )

        metadata['execution'] = {'mining_options': mining_options.copy(),
                                 'memory_budget_bytes': self._memory_budget,
                                 'max_candidate_pairs': self._max_candidate_pairs,
                                 'spark_partitions': self._spark_partitions}
        if collect:
            return Network(
                GRN=inferenceResult,
                metadata=metadata,
                inferenceParameters=self._thresholds,
            )
        else:
            return inferenceResult, metadata, self._thresholds