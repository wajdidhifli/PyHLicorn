import numpy as np
import pandas as pd
from typing import Literal, Optional
    
class PreProcess:
    """
    A class for preprocessing gene expression data, including discretization and 
    target gene list extraction.
    """
    
    @staticmethod
    def discretize_gene_expression(
        numerical_expression : pd.DataFrame, 
        threshold: float = 1, 
        refSamples: Optional[pd.DataFrame] = None, 
        sdBy: Literal['genes','global'] = 'genes'
    ) -> pd.DataFrame:
        
        """
        Discretizes numerical gene expression data based on a threshold of standard deviations.
        
        Parameters
        ----------
        numerical_expression : pandas.DataFrame
            A DataFrame containing numerical gene expression data, where rows represent genes 
            and columns represent samples.
        threshold : float, optional
            The threshold for standard deviations to determine discretization boundaries. 
            Default is 1.
        refSamples : pandas.DataFrame or None, optional
            A DataFrame containing reference samples for calculating mean and standard deviation. 
            If None, the mean and standard deviation are calculated from `numerical_expression`. 
            Default is None.
        sdBy : {'genes', 'global'}, optional
            Specifies whether to calculate standard deviation by genes (rows) or global (across all values). 
            Default is 'genes'.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame of the same shape as `numerical_expression`, containing discretized values:
            -1 for values below the negative threshold, 
            0 for values within the threshold range, and 
            1 for values above the positive threshold.
            
        Notes
        -----
        - If `numerical_expression` contains only non-negative values, centering is performed by subtracting the mean. Otherwise, no centering is applied.
        - NaN values in the centered data are replaced with 0 before discretization.
        - The `sdBy` parameter determines whether standard deviation is calculated per gene (row-wise) or globally across all values.
        """
        
        if sdBy not in {'genes', 'global'}:
            raise ValueError("sdBy must be either 'genes' or 'global'")

        numerical_expressionArray = np.array(numerical_expression)
        if refSamples is not None:
            refSamples=np.array(refSamples)
        
        if sdBy=='genes':
            if refSamples is not None:
                ref_means=np.mean(refSamples,axis=1,keepdims=True)
                centered=numerical_expressionArray - ref_means
            else:
                if np.all(numerical_expressionArray>=0):
                    centered=numerical_expressionArray - np.mean(numerical_expressionArray,axis=1,keepdims=True)
                else:
                    centered=numerical_expressionArray

        else: 
            # Computing 1 mean for all (1 mean value for all)
            if refSamples is not None:
                ref_means=np.mean(refSamples)
                centered=numerical_expressionArray-ref_means
            else:
                if np.all(numerical_expressionArray>=0):
                    centered=numerical_expressionArray-np.mean(numerical_expressionArray)
                else:
                    centered=numerical_expressionArray


        centered[np.isnan(centered)]=0

        if sdBy=='genes':
            sds=np.std(centered, axis=1,keepdims=True)
        else:
            sds=np.std(centered)

        standardDeviationThreshold=threshold*sds

        discretized=np.zeros_like(centered)
        discretized[centered <= -standardDeviationThreshold] = -1
        discretized[centered >= standardDeviationThreshold] = 1
        # Zero-variance rows carry no differential-expression evidence.
        discretized = np.where(sds == 0, 0, discretized)
        discretized = pd.DataFrame(discretized.astype(int), index=numerical_expression.index, columns=numerical_expression.columns)
        return discretized


    # Get gene list 
    @staticmethod
    def target_gene_list(numerical_expression : pd.DataFrame, tf_list: pd.DataFrame):
        """
        A list of genes for which regulatory networks should be inferred. 
        These genes should appear in the row names of the expression data. 
        If not provided, all genes in the row names of the expression data 
        that are not in `tf_list` will be used.
        
        Parameters
        ----------
        numerical_expression : pandas.DataFrame
            A numerical matrix containing the expression levels of genes and 
            transcription factors. This will be used to infer the network.
            
        tf_list : pandas.DataFrame
            A list of genes designated as transcription factors (TFs) or regulators. 
            This should contain gene names.
            
        Returns
        -------
        list
        """
        num_row_names = numerical_expression.index.tolist()
        gene_list = list(set(num_row_names) - set(tf_list))
        target_gene_list = list(set(numerical_expression.index).intersection(gene_list))
        
        return target_gene_list

