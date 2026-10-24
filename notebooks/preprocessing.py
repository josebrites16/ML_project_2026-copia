"""The final preparation and the evaluation ruler, in one place.

The decisions made in one notebook reach the next through this file, which they all
import. The experiments and their justification live in 02_preprocessing.ipynb; this
file only holds what was chosen.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "repeat_purchase_90d"
GROUP = "customer_id"
DATE = "purchase_date"
OUT_OF_MODEL = [GROUP, DATE]   # kept in the table, never given to the model
METRIC = "f1"                  # the metric that decides; accuracy is shown beside it


# ----------------------------------------------------------------------------- data, splits and evaluation
def load_train(path):
    """train.csv sorted by date, without the rows that have no date."""
    raw = pd.read_csv(path, index_col="ID")
    known_time = raw[DATE].notna()
    # kind="stable" keeps purchases from the same day in file order
    time_order = raw.loc[known_time, DATE].sort_values(kind="stable").index
    return raw, raw.loc[time_order]


def time_splits(frame):
    """5 time-based splits: each one trains on the past and validates on what comes next."""
    return list(TimeSeriesSplit(n_splits=5).split(frame))


def model_columns(frame):
    """The columns the model sees: numeric and categorical (no target, customer or date)."""
    features = [c for c in frame.columns if c not in OUT_OF_MODEL + [TARGET]]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def as_object(block):
    """scikit-learn reads np.nan as a gap but not pd.NA, so every gap becomes np.nan."""
    return block.astype(object).fillna(np.nan).infer_objects()


def prepare_base(train, test):
    """Baseline: median for numeric and mode for categorical columns, fitted on train only."""
    train, test = train.copy(), test.copy()
    numeric, categorical = model_columns(train)
    imputer = SimpleImputer(strategy="median").fit(train[numeric])
    train[numeric], test[numeric] = imputer.transform(train[numeric]), imputer.transform(test[numeric])
    imputer = SimpleImputer(strategy="most_frequent").fit(as_object(train[categorical]))
    train[categorical] = imputer.transform(as_object(train[categorical]))
    test[categorical] = imputer.transform(as_object(test[categorical]))
    return train, test


def fit_predict(prepare, train, test, scaler=StandardScaler(), encoder=None, model=None,
                select=None, threshold=None, proba=False):
    """Prepare, train on `train` and return the predictions (0 or 1) for `test`.

    Everything is fitted on `train` only: `prepare`, the encoder and the scaler.
    `scaler=None` does not scale. `encoder=None` is the baseline one-hot; other encoders can
    be passed. The encoder also receives the training target, which only target encoding uses.
    `model=None` is the baseline logistic regression; other models can be passed.
    `select=None` keeps every column; a selector can be passed, fitted on train only after
    the encoder and the scaler. `threshold=None` uses the model's predict (threshold 0.5);
    with a number, it predicts 1 when the probability of class 1 reaches that threshold.
    `proba=True` returns that probability instead of labels, to choose the threshold on the
    precision-recall curve. `test` does not need the target.
    """
    train, test = prepare(train, test)
    numeric, categorical = model_columns(train)
    if encoder is None:
        # handle_unknown="ignore": a level that only appears in test becomes all zeros
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(as_object(train[categorical]), train[TARGET])
    a = np.hstack([train[numeric], encoder.transform(as_object(train[categorical]))])
    b = np.hstack([test[numeric], encoder.transform(as_object(test[categorical]))])
    if scaler is not None:
        scaler.fit(a)
        a, b = scaler.transform(a), scaler.transform(b)
    if select is not None:
        # with column names, so we can tell later which ones were kept
        names = numeric + list(encoder.get_feature_names_out())
        select.fit(pd.DataFrame(a, columns=names), train[TARGET])
        a = select.transform(pd.DataFrame(a, columns=names))
        b = select.transform(pd.DataFrame(b, columns=names))
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(a, train[TARGET])
    if proba:
        return model.predict_proba(b)[:, 1]
    if threshold is None:
        return model.predict(b)
    return (model.predict_proba(b)[:, 1] >= threshold).astype(int)


def evaluate(prepare, frame, splits, scaler=StandardScaler(), encoder=None, model=None,
             select=None, threshold=None):
    """F1 and accuracy of one alternative `prepare(train, test)` on each split.

    On each split the model trains on the training rows and predicts the validation rows
    (fit_predict).
    """
    rows = []
    for train_idx, val_idx in splits:
        train, test = frame.iloc[train_idx], frame.iloc[val_idx]
        predicted = fit_predict(prepare, train, test, scaler, encoder, model, select, threshold)
        row = {"f1": f1_score(test[TARGET], predicted),
               "accuracy": accuracy_score(test[TARGET], predicted)}
        if select is not None:
            # the columns the selector kept on this split
            row["kept"] = list(select.get_feature_names_out())
            row["columns"] = len(row["kept"])
        rows.append(row)
    return pd.DataFrame(rows)


def sem(values):
    """Standard error of the mean: how much the mean would move with other splits."""
    return values.std(ddof=1) / np.sqrt(len(values))


def compare(runs, reference, metric=METRIC):
    """Each alternative against the reference, on the same splits.

    A difference smaller than 2 x SEM is noise.
    """
    table = {}
    for name, run in runs.items():
        difference = run[metric] - reference[metric]
        table[name] = {metric: run[metric].mean(),
                       "accuracy": run["accuracy"].mean(),
                       "vs base": difference.mean(),
                       "SEM": sem(difference),
                       "better on": f"{int((difference > 0).sum())} of {len(difference)}"}
    return pd.DataFrame(table).T


def verdict(table):
    """better / worse if the difference passes 2 x SEM, otherwise tie."""
    return np.where(table["vs base"] > 2 * table["SEM"], "better",
                    np.where(table["vs base"] < -2 * table["SEM"], "worse", "tie"))


# ----------------------------------------------------------------------------- impossible values
# Domain rules from the EDA: True where the value is impossible
invalid_errors = {
    "days_since_previous_purchase": lambda values: values < 0,          # negative days (-4)
    "prior_mean_gap_days": lambda values: values < 0,                   # negative days (-4)
    "prior_mean_purchase_value": lambda values: (values < 0) | (values == 1e9),
    "spend_365d": lambda values: (values < 0) | (values == 1e9),
}


def invalid_flags(frame):
    """One column per rule, True where the value is impossible.

    Plus the two-column rule from the EDA: the mean of a customer's purchases cannot be
    larger than the largest of those purchases.
    """
    flags = pd.concat([rule(frame[c]) for c, rule in invalid_errors.items()], axis=1)
    flags["prior_mean_purchase_value"] |= (frame["prior_mean_purchase_value"]
                                           > frame["prior_max_purchase_value"] + 0.02)
    return flags


def blank_invalid(frame):
    """Impossible values become NaN.

    It learns nothing from the data, so it is applied the same way to train, validation
    and test.
    """
    out = frame.copy()
    out[list(invalid_errors)] = out[list(invalid_errors)].mask(invalid_flags(frame))
    return out


# ----------------------------------------------------------------------------- long tails
# The numeric columns with a long right tail (EDA): all but random_noise and customer_tenure_days
SKEWED = ["Total a Pagar", "Total Bruto", "Total Liquido", "Total Imp_", "Total Portes",
          "Total Desc_ Global", "Total Desc_ Linha", "Custo", "N_ Detalhes",
          "invoice_count_current_purchase", "days_since_previous_purchase",
          "prior_purchase_count", "prior_total_spend", "prior_mean_purchase_value",
          "prior_std_purchase_value", "prior_min_purchase_value", "prior_max_purchase_value",
          "prior_mean_gap_days", "prior_std_gap_days", "purchase_count_30d", "spend_30d",
          "purchase_count_90d", "spend_90d", "purchase_count_180d", "spend_180d",
          "purchase_count_365d", "spend_365d"]


def prepare_log1p(train, test):
    """log1p on the long-tailed columns.

    It learns nothing from the data, so it is applied the same way to train and test.
    """
    train, test = blank_invalid(train), blank_invalid(test)
    train[SKEWED], test[SKEWED] = np.log1p(train[SKEWED]), np.log1p(test[SKEWED])
    return prepare_base(train, test)


# ----------------------------------------------------------------------------- final preparation
prepare_final = prepare_log1p   # impossible values -> null, log1p, median / mode
ENCODER_FINAL = None            # None = one-hot with handle_unknown="ignore"
SCALER_FINAL = MinMaxScaler()


# ----------------------------------------------------------------------------- final feature set
def prepare_without(columns):
    """prepare_final, then remove `columns`, before the one-hot and the model.

    Removing after the preparation gives the same result as removing before (the median,
    mode and log1p of a column do not depend on the others), and it keeps the SKEWED list
    and the impossible-value rule, which uses prior_max_purchase_value, unchanged.
    """
    def prepare(train, test):
        train, test = prepare_final(train, test)
        return train.drop(columns=columns), test.drop(columns=columns)
    return prepare


# The 14 columns removed by feature selection
DROPPED_COLUMNS = ["Custo", "Total Bruto", "Total Imp_", "Total Liquido", "commercial_zone",
           "prior_purchase_count", "prior_std_gap_days", "prior_std_purchase_value",
           "purchase_count_180d", "purchase_count_30d", "purchase_count_365d", "random_noise",
           "spend_180d", "transport_method"]
prepare_selected = prepare_without(DROPPED_COLUMNS)
SELECT_FINAL = None             # no selector on top: the columns are already chosen
