import time
import warnings
from typing import Dict, Tuple, TypedDict

import pandas as pd

from .columns import encode_partition_numeric_col
from .constants import INVALID_NUMS_RE, NUMERIC_NA_TOKEN


class NumericTransformData(TypedDict, total=False):
    """Documents the implicit fit/transform schema that
    `process_numeric_data`/`process_datetime_data` build up in their
    `transform_data` dict.

    This is a pure static-typing aid -- at runtime it's still a plain
    `dict`. `col_transform_data` (the outer `Dict[str, Dict]` keyed by
    column name, built in `data_utils.process.process_data`) is dumped
    verbatim as raw JSON by `REaLTabFormer.save()` and restored via a
    generic `setattr` loop with no key remapping, so these literal key
    names must not change without a compatibility plan for already-saved
    models.

    Keys are populated conditionally, hence `total=False`:
    - `mx_sig`/`zfill` are set for values with no decimal point.
    - `mx_sig`/`ljust` are set for values with a decimal point.
    - `mean_date` is set only for datetime-derived columns.
    - `numeric_nparts` is stamped by `process_data`, not by these
      functions themselves.
    """

    max_len: int
    numeric_precision: int
    mx_sig: int
    zfill: int
    ljust: int
    mean_date: int
    numeric_nparts: int


def fix_multi_decimal(v):
    if v.count(".") > 1:
        v = v.split(".")
        v = ".".join([v[0], "".join(v[1:])])
    return v


def process_numeric_data(
    series: pd.Series,
    max_len: int = 10,
    numeric_precision: int = 4,
    transform_data: Dict = None,
) -> Tuple[pd.Series, NumericTransformData]:
    is_transform = True

    if transform_data is None:
        transform_data = dict()
        is_transform = False

    if is_transform:
        warnings.warn(
            "Default values will be overridden because transform_data was passed..."
        )
        max_len = transform_data["max_len"]
        numeric_precision = transform_data["numeric_precision"]
    else:
        transform_data["max_len"] = max_len
        transform_data["numeric_precision"] = numeric_precision

    # Note that at this point, we should have casted int-like values to
    # pd.Int64Dtype but just to be very sure, let's do that again here.
    try:
        series = series.astype(pd.Int64Dtype())
    except TypeError:
        pass
    except ValueError:
        pass

    if series.dtype == pd.Int64Dtype():
        series = series.astype(str)
    else:
        # We convert float-like values to string with the specified
        # maximum precision.
        # NOTE: Don't use series.round(numeric_precision).astype(str)
        # In some cases, this introduces scientific notation in the
        # string version causing "invalid" values.

        # Note that our purpose for doing this is to actually truncate
        # the precision and not increase the precision.
        # So, we strip the right trailing zeros because the formatting
        # pads the series to the numeric_precision even when not needed.
        series = series.map(lambda x: f"{x:.{numeric_precision}f}").str.rstrip("0")

    # Get the most significant digit
    if is_transform:
        mx_sig = transform_data["mx_sig"]
    else:
        mx_sig = series.str.find(".").max()
        transform_data["mx_sig"] = int(mx_sig)

    if mx_sig <= 0:
        # The data has no decimal point.
        # Pad the data with leading zeros if not
        # aligned to the largest value.
        # We also don't apply the max_len to integral
        # valued data because it will basically
        # remove important information.
        if is_transform:
            zfill = transform_data["zfill"]
        else:
            zfill = series.map(len).max()
            transform_data["zfill"] = int(zfill)
        series = series.str.zfill(zfill)
    else:
        # Make sure that we don't exessively truncate the data.
        # The max_len should be greater than the mx_sig.
        # Add a +1 to generate a minimum of tenth place resolution
        # for this data.
        assert max_len > (mx_sig + 1), (
            f"The target length {max_len} of the data doesn't include the numeric precision at {mx_sig}. Increase max_len to at least {max_len + (mx_sig + 2 - max_len)}."
        )

        # Left align first based on the magnitude of the values.
        # We compute the difference in the most significant digits
        # of all values with respect to the largest value.
        # We then pad a leading zero to values with lower most significant
        # digits.
        # For example we have the values 1029.61 and 4.269. This will
        # determine that 1029.61 has the largest magnitude, with most significant
        # digit of 4. It will pad the value 4.269 with three zeros and convert it
        # to 0004.269.
        series = (mx_sig - series.str.find(".")).map(lambda x: "0" * x) + series
        series = series.str[:max_len]

        # We additionally apply left justify to align based on the trailing precision.
        # For example, we have 1029.61 and 0004.269 as values. This time we transform the first
        # value to become 1029.610 to align with the precision of the second value.
        if is_transform:
            ljust = transform_data["ljust"]
        else:
            ljust = series.map(len).max()
            transform_data["ljust"] = int(ljust)

        series = series.str.ljust(ljust, "0")

    # If a number has a negative sign, make sure that it is placed properly.
    series.loc[series.str.contains("-", regex=False)] = "-" + series.loc[
        series.str.contains("-", regex=False)
    ].str.replace("-", "", regex=False)

    return series, transform_data


def process_datetime_data(
    series, transform_data: Dict = None
) -> Tuple[pd.Series, NumericTransformData]:
    # Get the max_len from the current time.
    # This will be ignored later if the actual max_len
    # is shorter.
    max_len = len(str(int(time.time())))

    # Convert the datetimes to
    # their equivalent timestamp values.

    # Make sure that we don't convert the NaT
    # to some integer.
    series = series.copy()

    # Track null values (NaT)
    null_idx = series.isnull()

    # Convert to the numerical representation
    # of the datetime (UNIX timestamp)
    series = series.astype("int64") / 1e9

    # Fill NA
    series.loc[null_idx] = pd.NA

    # Cast as integer type
    series = series.astype("Int64")

    # Take the mean value to re-align the data.
    # This will help reduce the scale of the numeric
    # data that will need to be generated. Let's just
    # add this offset back later before casting.
    mean_date = None

    if transform_data is None:
        mean_date = int(series.mean())
        series -= mean_date
    else:
        # The mean_date should have been
        # stored during fitting.
        series -= transform_data["mean_date"]

    # Then apply the numeric data processing
    # pipeline.
    series, transform_data = process_numeric_data(
        series,
        max_len=max_len,
        numeric_precision=0,
        transform_data=transform_data,
    )

    # Store the `mean_date` here because `process_numeric_data`
    # expects a None transform_data during fitting.
    if mean_date is not None:
        transform_data["mean_date"] = mean_date

    return series, transform_data


def process_categorical_data(series: pd.Series) -> pd.Series:
    # Simply convert the categorical data to string.
    return series.astype(str)


def tokenize_numeric_col(series: pd.Series, nparts=2, col_zfill=2):
    # After normalizing the numeric values, we then segment
    # them based on a fixed partition size (nparts).
    col = series.name
    max_len = series.map(len).min()

    # Take the observations that have non-numeric characters.
    # These are NaNs.
    nan_obs = series.str.contains(INVALID_NUMS_RE, regex=True)

    if nparts > max_len > 2:
        # Allow minimum of 0-99 as acceptable singleton range.
        raise ValueError(
            f"Partition size {nparts} is greater than the value length {max_len}. Consider reducing the number of partitions..."
        )
    mx = series.map(len).max()

    tr = pd.concat([series.str[i : i + nparts] for i in range(0, mx, nparts)], axis=1)

    # Replace values with NUMERIC_NA_TOKEN
    tr.loc[nan_obs] = NUMERIC_NA_TOKEN * nparts

    tr.columns = encode_partition_numeric_col(col, tr, col_zfill)

    return tr
