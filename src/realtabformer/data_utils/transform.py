import time
import warnings
from typing import Dict, Tuple, TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

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
    - `quantile_encoding=True` columns set neither `mx_sig`, `zfill`, nor
      `ljust` -- they use a dedicated fixed-width digit-index formatting
      path (see `process_numeric_data`) that bypasses the generic
      magnitude-alignment code those three keys support, and nothing
      outside `process_numeric_data` itself reads them.
    """

    max_len: int
    numeric_precision: int
    mx_sig: int
    zfill: int
    ljust: int
    mean_date: int
    numeric_nparts: int
    quantile_encoding: bool
    quantile_values: list
    quantile_positions: list


def fix_multi_decimal(v):
    if v.count(".") > 1:
        v = v.split(".")
        v = ".".join([v[0], "".join(v[1:])])
    return v


# A single value recurring in at least this fraction of a column's rows
# (e.g. the zero in a zero-inflated capital-gain/loss-style column) is
# treated as a dedicated point mass by `_fit_quantile_breakpoints` rather
# than left to consume its proportional share of `n_quantiles` breakpoints
# via the plain QuantileTransformer fit. Below this, an ordinary tie in
# continuous data isn't worth the extra fit complexity -- the boundary
# grid-snap in `_apply_quantile_encoding` already makes it decode
# correctly, just without the resolution optimization below.
_POINT_MASS_THRESHOLD = 0.05


def _fit_quantile_breakpoints(
    valid: pd.Series, n_quantiles: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the `(quantile_values, quantile_positions)` breakpoint arrays
    for `valid`'s empirical distribution, `n_quantiles` at a time.

    Detects a single dominant point mass -- one value recurring in at
    least `_POINT_MASS_THRESHOLD` of rows, the pattern a zero-inflated
    column like the UCI Adult dataset's `capital-gain`/`capital-loss`
    columns has (>90% exact zeros there). A plain fit on the *full*
    column spends a share of its `n_quantiles` breakpoints proportional
    to that value's frequency describing the same repeated value over
    and over -- e.g. 955 of 1000 breakpoints for a 95.5%-zero column --
    leaving only the remainder to describe the part of the distribution
    that actually varies. `_apply_quantile_encoding`'s boundary grid-snap
    already makes this *correct* (the point mass decodes back exactly),
    but it's a real resolution waste: confirmed on that exact column,
    quantile encoding's Wasserstein distance to the training distribution
    was still ~5x worse than the fixed-width baseline's after that fix,
    even though the KS statistic (marginal shape) had fully recovered.

    When a point mass is found, this excises it from the fit entirely --
    `QuantileTransformer` only ever sees the *non*-dominant values, so
    every one of its `n_quantiles` breakpoints describes genuine
    variation -- then reserves the dominant value its own single
    breakpoint, sized to its true empirical frequency `p0` and placed at
    its correct rank: values below it get rescaled into `[0, below_frac)`,
    values above into `[below_frac + p0, 1)`, and the dominant value
    itself sits at the midpoint of the `[below_frac, below_frac + p0)`
    gap in between. This is exactly what the *true* empirical CDF of the
    untouched column already looks like (a vertical jump of height `p0`
    at the dominant value) -- the only change is spending the finite
    breakpoint budget on the part of the curve that isn't flat.

    Falls back to a plain single fit (today's behavior, unchanged) when
    no value clears `_POINT_MASS_THRESHOLD`, or when excising the
    dominant value would leave nothing to fit (a column that actually is
    just one repeated value, which `numeric_categorical_threshold`, not
    this, is the intended fix for).
    """
    counts = valid.value_counts()
    top_value = float(counts.index[0])
    p0 = counts.iloc[0] / len(valid)
    remainder = valid[valid != top_value]

    if p0 < _POINT_MASS_THRESHOLD or remainder.empty:
        n_q = min(n_quantiles, len(valid))
        qt = QuantileTransformer(n_quantiles=n_q, output_distribution="uniform")
        qt.fit(valid.to_numpy().reshape(-1, 1))
        return qt.quantiles_.ravel(), qt.references_

    n_q = min(n_quantiles, len(remainder))
    qt = QuantileTransformer(n_quantiles=n_q, output_distribution="uniform")
    qt.fit(remainder.to_numpy().reshape(-1, 1))
    values = qt.quantiles_.ravel()
    positions = qt.references_.copy()

    # `remainder` excludes top_value by construction, so every fitted
    # breakpoint value is strictly below or strictly above it -- these
    # two masks partition `values` exactly, no third case.
    below_mask = values < top_value
    above_mask = ~below_mask
    positions[below_mask] = positions[below_mask] * (1 - p0)
    positions[above_mask] = positions[above_mask] * (1 - p0) + p0

    below_frac = (remainder < top_value).mean() * (1 - p0)
    insert_at = np.searchsorted(values, top_value)
    values = np.insert(values, insert_at, top_value)
    positions = np.insert(positions, insert_at, below_frac + p0 / 2)

    return values, positions


def _apply_quantile_encoding(
    series: pd.Series,
    transform_data: Dict,
    is_transform: bool,
    n_quantiles: int = 1000,
) -> Tuple[pd.Series, Dict]:
    """Transform `series` (raw numeric values) to its quantile position
    `q` under the column's own empirical distribution, `q ~ Uniform(0, 1)`
    by the probability integral transform -- makes every digit-chunk
    position of the eventual formatted string informative regardless of
    the column's shape (heavy-tailed, bimodal, whatever), unlike fixed
    absolute-precision formatting, which manufactures near-constant
    leading chunks for heavy-tailed columns (see the digit_entropy_weighting/
    numeric_categorical_threshold docstrings this is the representation-level
    counterpart to).

    Only the *breakpoints* used to compute `q` are fitted with
    `sklearn.preprocessing.QuantileTransformer` (already a project
    dependency -- see rtf_analyze.py) -- reusing its tested
    duplicate-handling/subsampling logic for that one step. Its fitted
    object is not what gets persisted: `quantiles_`/`references_` are
    pulled out as plain Python lists into `transform_data` (JSON-safe,
    sklearn-version-independent, stored exactly like `mx_sig`/`zfill`/
    `ljust` already are), and every actual forward/inverse mapping --
    here, and later at decode time in `rtf_sampler.py::_recover_data_values`
    -- is a single `np.interp` call against those stored arrays.
    `np.interp`'s default clamp-to-boundary behavior for out-of-range
    input *is* the extrapolation rule: a quantile value generated outside
    what was observed in training clips to the training min/max, rather
    than attempting to model a parametric tail.
    """
    numeric_series = series.astype("float64")
    na_mask = numeric_series.isna()
    valid = numeric_series[~na_mask]

    if is_transform:
        quantile_values = np.array(transform_data["quantile_values"])
        quantile_positions = np.array(transform_data["quantile_positions"])
    else:
        n_unique = valid.nunique()
        max_levels = 10**transform_data["numeric_precision"]
        if n_unique > max_levels:
            warnings.warn(
                f"Column {series.name!r} has {n_unique} unique values but "
                f"numeric_precision={transform_data['numeric_precision']} only "
                f"supports {max_levels} distinguishable quantile levels -- "
                "distinct values may collide onto the same reconstructed value. "
                "Consider increasing numeric_precision."
            )

        quantile_values, quantile_positions = _fit_quantile_breakpoints(
            valid, n_quantiles
        )

        # Point-mass safety: a value that recurs many times (e.g. a
        # zero-inflated column) collapses a long run of quantile_positions
        # onto the *same* quantile_values entry, so np.interp assigns every
        # occurrence of that value to one exact position -- the run's right
        # edge (its last, highest-index reference). If that edge's raw
        # float, e.g. 0.954954954954955, isn't exactly representable at
        # `numeric_precision` decimal digits, formatting it to a string and
        # parsing it back (as generation always does) rounds it to the
        # nearest grid point, e.g. "0.9550" -- which can land on the *other*
        # side of the boundary into the next distinct value's segment.
        # Where that segment's value jumps sharply (exactly what a point
        # mass produces), this turns an imperceptible float-rounding nudge
        # into a catastrophically wrong decoded value: confirmed on the
        # UCI Adult `capital-loss` column (95.5% exact zeros), where every
        # zero-valued row decoded to ~85 instead of 0.
        #
        # Fixed by snapping every stored position down to the
        # numeric_precision grid before it is ever used, so formatting a
        # position that's already grid-aligned is a no-op -- no rounding
        # step remains that could cross a boundary. Flooring (not rounding)
        # keeps each position on the same side of its true, unsnapped
        # position, which is what preserves the boundary's correct side.
        # A single monotonic non-decreasing pass afterwards is a cheap
        # safety net for the (already-warned-about) case where distinct
        # values are packed closer together than the precision grid can
        # resolve -- it cannot introduce a *new* misordering, only leave
        # the pre-existing one in place, deterministically. Also covers
        # `_fit_quantile_breakpoints`'s own explicitly-inserted point-mass
        # position (e.g. `below_frac + p0 / 2`), which is just as capable
        # of landing off-grid as any breakpoint QuantileTransformer itself
        # produces.
        grid = 10 ** transform_data["numeric_precision"]
        quantile_positions = np.floor(quantile_positions * grid) / grid
        quantile_positions = np.maximum.accumulate(quantile_positions)

        transform_data["quantile_values"] = quantile_values.tolist()
        transform_data["quantile_positions"] = quantile_positions.tolist()

    transformed = numeric_series.copy()
    transformed[~na_mask] = np.interp(valid.to_numpy(), quantile_values, quantile_positions)

    return transformed, transform_data


def process_numeric_data(
    series: pd.Series,
    max_len: int = 10,
    numeric_precision: int = 4,
    transform_data: Dict = None,
    quantile_encoding: bool = False,
    quantile_n_bins: int = 1000,
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
        # A frozen fit-time decision, like every other key here -- replayed,
        # never re-decided from whatever the caller happens to pass.
        quantile_encoding = transform_data.get("quantile_encoding", False)
    else:
        transform_data["max_len"] = max_len
        transform_data["numeric_precision"] = numeric_precision
        # Only stamped when actually used -- `.get("quantile_encoding",
        # False)` above already treats an absent key as False on replay,
        # so leaving it out entirely when unused keeps a default
        # (non-quantile) column's transform_data byte-identical to
        # before this feature existed, instead of growing an inert key
        # on every column regardless of whether anyone asked for this.
        if quantile_encoding:
            transform_data["quantile_encoding"] = quantile_encoding

    if quantile_encoding:
        assert numeric_precision > 0, (
            "quantile_encoding requires numeric_precision > 0 -- a quantile "
            "value with zero fractional digits has no resolution at all."
        )
        series, transform_data = _apply_quantile_encoding(
            series, transform_data, is_transform, n_quantiles=quantile_n_bins
        )

        # `series` now holds q in [0, 1) (NaN preserved by
        # `_apply_quantile_encoding`). Format it as a bare zero-padded
        # digit-index string -- q quantized to numeric_precision decimal
        # digits, written with no decimal point -- instead of routing it
        # through the generic "0.XXXX" formatting path below. Dropping the
        # "0." saves two structurally-constant characters per row: q is
        # *always* < 1 by construction, so the leading digit is always "0"
        # and the point's position never varies -- neither carries any
        # information the model needs to learn or the vocab needs a token
        # for. The width is fixed at exactly numeric_precision digits by
        # construction here, unlike the generic zfill branch below (whose
        # width is `series.str.len().max()`, i.e. data-dependent) -- a low
        # digit-index like 5 ("0005" at precision=4) is a perfectly
        # ordinary value, not a case that should shrink the column's width.
        # `rtf_sampler.py::_recover_data_values` reverses this: divides the
        # recovered digit-index back by the same grid to recover q, then
        # applies dequantization dithering before the inverse np.interp.
        grid = 10**numeric_precision

        def _format_quantile_digits(x, grid=grid, numeric_precision=numeric_precision):
            if pd.isna(x):
                # Matches the plain-float path's own NaN formatting
                # (`f"{x:.{p}f}"` on NaN gives the literal "nan") so the
                # existing INVALID_NUMS_RE-based NaN detection in
                # `tokenize_numeric_col` catches it identically, with no
                # separate handling needed there.
                return "nan"
            d = int(np.clip(round(x * grid), 0, grid - 1))
            return str(d).zfill(numeric_precision)

        series = series.map(_format_quantile_digits)
        return series, transform_data

    # Note that at this point, we should have casted int-like values to
    # pd.Int64Dtype but just to be very sure, let's do that again here.
    # (quantile_encoding=True never reaches this point at all -- its
    # branch above returns directly with its own dedicated, always-integer
    # digit-index formatting, so there's no analogous whole-number/
    # misrouting hazard here to guard against.)
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
        #
        # Deliberately left as a Python-level `.map(lambda)` rather than
        # vectorized: there's no numpy/pandas primitive that reproduces
        # this exact fixed-point formatting without the scientific-notation
        # hazard `series.round(numeric_precision).astype(str)` has above --
        # `numpy.format_float_positional`/`np.vectorize` don't remove the
        # per-row Python call either, they just replace one Python-level
        # formatter with another. This runs once per numeric/datetime
        # *column* (small N), not per row x column, so the correctness
        # risk isn't worth taking for a comparatively small win.
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
            zfill = series.str.len().max()
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
        #
        # Deliberately left as a Python-level `.map(lambda)`: numpy has no
        # ragged/variable-length string-repeat primitive, so a vectorized
        # version would need to pad to a fixed width and slice instead --
        # about the same cost with more code and edge-case risk. Same
        # once-per-column (not per-row-x-column) scale argument as above.
        series = (mx_sig - series.str.find(".")).map(lambda x: "0" * x) + series
        series = series.str[:max_len]

        # We additionally apply left justify to align based on the trailing precision.
        # For example, we have 1029.61 and 0004.269 as values. This time we transform the first
        # value to become 1029.610 to align with the precision of the second value.
        if is_transform:
            ljust = transform_data["ljust"]
        else:
            ljust = series.str.len().max()
            transform_data["ljust"] = int(ljust)

        series = series.str.ljust(ljust, "0")

    # If a number has a negative sign, make sure that it is placed properly.
    neg_mask = series.str.contains("-", regex=False)
    series.loc[neg_mask] = "-" + series.loc[neg_mask].str.replace(
        "-", "", regex=False
    )

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
    max_len = series.str.len().min()

    # Take the observations that have non-numeric characters.
    # These are NaNs.
    nan_obs = series.str.contains(INVALID_NUMS_RE, regex=True)

    if nparts > max_len > 2:
        # Allow minimum of 0-99 as acceptable singleton range.
        raise ValueError(
            f"Partition size {nparts} is greater than the value length {max_len}. Consider reducing the number of partitions..."
        )
    mx = series.str.len().max()

    tr = pd.concat([series.str[i : i + nparts] for i in range(0, mx, nparts)], axis=1)

    # Replace values with NUMERIC_NA_TOKEN
    tr.loc[nan_obs] = NUMERIC_NA_TOKEN * nparts

    tr.columns = encode_partition_numeric_col(col, tr, col_zfill)

    return tr
