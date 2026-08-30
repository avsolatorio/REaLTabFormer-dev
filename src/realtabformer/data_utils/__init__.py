"""This package contains the implementation for the data preprocessing,
tokenization, and vocabulary building used for tabular and relational
data modeling.

It was split out of a single flat `data_utils.py` module into focused
submodules (constants, column-naming, per-type value transforms, vocab
building, dataset/tokenization, and the `process_data` orchestration) for
maintainability. Every name below is re-exported here so existing imports
(`from realtabformer.data_utils import X` / `realtabformer.data_utils.X`)
keep working unchanged.
"""

from .columns import (  # noqa: F401
    decode_column_values,
    decode_partition_numeric_col,
    decode_processed_column,
    encode_column_values,
    encode_partition_numeric_col,
    encode_processed_column,
    extract_processed_column,
    is_categorical_col,
    is_datetime_col,
    is_numeric_col,
    is_numeric_datetime_col,
)
from .constants import (  # noqa: F401
    INVALID_NUMS_RE,
    NUMERIC_NA_TOKEN,
    SPECIAL_COL_SEP,
    TEACHER_FORCING_PRE,
    ColDataType,
    ModelFileName,
    ModelType,
    SpecialTokens,
    TabularArtefact,
    get_uuid,
)
from .dataset import (  # noqa: F401
    get_input_ids,
    get_relational_input_ids,
    get_token_id,
    make_dataset,
    make_relational_dataset,
)
from .process import process_data  # noqa: F401
from .transform import (  # noqa: F401
    NumericTransformData,
    fix_multi_decimal,
    process_categorical_data,
    process_datetime_data,
    process_numeric_data,
    tokenize_numeric_col,
)
from .vocab import build_vocab  # noqa: F401
