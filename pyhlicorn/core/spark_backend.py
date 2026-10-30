"""Typed partition inference with explicitly owned shared broadcasts."""
import numpy as np
from .._serialization import native, normalize_frame
from .oneGeneHlicorn import oneGeneHlicorn


def result_schema(linear_model):
    from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
    fields = [StructField('Target', StringType()),
              StructField('coact', ArrayType(StringType())),
              StructField('corep', ArrayType(StringType()))]
    if linear_model is not None:
        fields += [StructField(name, ArrayType(DoubleType())) for name in ('Coef_Acts', 'Coef_Reps')]
        fields += [StructField(name, DoubleType()) for name in ('Coef_coActs', 'Coef_coReps', 'R2', 'RMSE')]
    return StructType(fields)


def infer_partition(rows, state):
    # Loaded once per partition; numpy arrays are reused for every target.
    reg_disc, items, bit, reg_num, threshold, ngrn, model, folds, max_pairs = state
    for gene, discrete, numerical in rows:
        out = oneGeneHlicorn(gene, np.asarray(discrete), reg_disc, items, bit,
                            threshold, reg_num, np.asarray(numerical), ngrn,
                            model, False, cv_folds=folds, max_candidate_pairs=max_pairs)
        for row in normalize_frame(out).itertuples(index=False, name=None):
            yield tuple(native(value) for value in row)


def infer_spark(session, rows, state, partitions=None):
    candidate_store = state[1]
    if hasattr(candidate_store, 'for_spark'):
        worker_store = candidate_store.for_spark(session.sparkContext)
        state = (state[0], worker_store, *state[2:])
    broadcast = session.sparkContext.broadcast(state)
    try:
        schema = result_schema(state[6])
        def process_partition(iterator):
            yield from infer_partition(iterator, broadcast.value)
        rdd = session.sparkContext.parallelize(rows, numSlices=partitions)
        result = session.createDataFrame(rdd.mapPartitions(process_partition), schema)
    except BaseException:
        broadcast.destroy()
        raise
    # Keep valid for repeated actions on a lazy result. Explicit release only.
    result._pyhlicorn_broadcasts = [broadcast]
    result._pyhlicorn_candidate_store = candidate_store
    return result


def release_spark_resources(frame):
    """Destroy inference broadcasts AFTER the final action; frame cannot be reused.

    Pass the original frame returned by fit(collect=False), even if actions were
    performed on derived frames. SparkSession.stop also cleans application state.
    """
    for broadcast in getattr(frame, '_pyhlicorn_broadcasts', []):
        broadcast.destroy(blocking=True)
    frame._pyhlicorn_broadcasts = []
    candidate_store = getattr(frame, '_pyhlicorn_candidate_store', None)
    if candidate_store is not None and hasattr(candidate_store, 'close'):
        candidate_store.close()
    frame._pyhlicorn_candidate_store = None
