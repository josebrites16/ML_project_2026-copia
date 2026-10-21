"""A preparação final das tarefas 2 e 3 e a régua de avaliação, num só sítio.

Como o preprocessing.py das práticas (W5 a W9): as decisões tomadas num notebook passam ao
seguinte por um ficheiro que todos importam. As experiências e as justificações ficam no
02_preprocessing.ipynb; aqui fica só o que foi escolhido.
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
OUT_OF_MODEL = [GROUP, DATE]   # ficam na tabela, mas nunca entram no modelo
METRIC = "f1"                  # métrica que decide (escolhida pela equipa na tarefa 3);
                               # a accuracy aparece ao lado, só como informação


# ----------------------------------------------------------------------------- 2.0 base comum
def load_train(path):
    """Passo 1: o train.csv ordenado por data (W5, secção 3.4), sem as linhas sem data."""
    raw = pd.read_csv(path, index_col="ID")
    known_time = raw[DATE].notna()
    # kind="stable" mantém pela ordem do ficheiro as compras do mesmo dia (W5)
    time_order = raw.loc[known_time, DATE].sort_values(kind="stable").index
    return raw, raw.loc[time_order]


def time_splits(frame):
    """Passo 3: 5 divisões por tempo; cada uma treina no passado e valida no que vem a seguir."""
    return list(TimeSeriesSplit(n_splits=5).split(frame))


def model_columns(frame):
    """Colunas que o modelo vê: numéricas e categóricas (sem alvo, cliente e data)."""
    features = [c for c in frame.columns if c not in OUT_OF_MODEL + [TARGET]]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def as_object(block):
    """O scikit-learn lê np.nan como nulo, mas não pd.NA: tudo passa a np.nan (W3)."""
    return block.astype(object).fillna(np.nan).infer_objects()


def prepare_base(train, test):
    """Linha de base: mediana nas numéricas e moda nas categóricas, ajustadas só no treino."""
    train, test = train.copy(), test.copy()
    numeric, categorical = model_columns(train)
    imputer = SimpleImputer(strategy="median").fit(train[numeric])
    train[numeric], test[numeric] = imputer.transform(train[numeric]), imputer.transform(test[numeric])
    imputer = SimpleImputer(strategy="most_frequent").fit(as_object(train[categorical]))
    train[categorical] = imputer.transform(as_object(train[categorical]))
    test[categorical] = imputer.transform(as_object(test[categorical]))
    return train, test


def fit_predict(prepare, train, test, scaler=StandardScaler(), encoder=None, model=None,
                select=None, threshold=None):
    """Prepara, treina no `train` e devolve as previsões (0 ou 1) para o `test`.

    Tudo é ajustado só no `train`: `prepare`, o encoder e o scaler.
    `scaler=None` não escala. `encoder=None` é o one-hot da linha de base; o 2.3 passa
    outros. O encoder recebe também o alvo do treino, que só o target encoding usa
    (W4, passo 5). `model=None` é a regressão logística da linha de base; as tarefas
    3 e 4 passam outros modelos. `select=None` usa todas as colunas; a tarefa 3 passa um
    seletor, ajustado só no treino depois do encoder e do scaler (W4: `select` do
    heldout_f1). `threshold=None` usa o predict do modelo (limiar 0.5); com um número,
    prevê 1 quando a probabilidade da classe 1 chega a esse limiar (W5, secção 2.7).
    O `test` não precisa de ter o alvo.
    """
    train, test = prepare(train, test)
    numeric, categorical = model_columns(train)
    if encoder is None:
        # handle_unknown="ignore": um nível que só aparece no test fica a zeros
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(as_object(train[categorical]), train[TARGET])
    a = np.hstack([train[numeric], encoder.transform(as_object(train[categorical]))])
    b = np.hstack([test[numeric], encoder.transform(as_object(test[categorical]))])
    if scaler is not None:
        scaler.fit(a)
        a, b = scaler.transform(a), scaler.transform(b)
    if select is not None:
        # com os nomes das colunas, como na W4, para se saber depois quais ficaram
        names = numeric + list(encoder.get_feature_names_out())
        select.fit(pd.DataFrame(a, columns=names), train[TARGET])
        a = select.transform(pd.DataFrame(a, columns=names))
        b = select.transform(pd.DataFrame(b, columns=names))
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(a, train[TARGET])
    if threshold is None:
        return model.predict(b)
    return (model.predict_proba(b)[:, 1] >= threshold).astype(int)


def evaluate(prepare, frame, splits, scaler=StandardScaler(), encoder=None, model=None,
             select=None, threshold=None):
    """F1 e accuracy de uma alternativa `prepare(train, test)` em cada divisão.

    Em cada divisão o modelo treina no treino e prevê a validação (fit_predict).
    Como o `held_out_f1` da W3.
    """
    rows = []
    for train_idx, val_idx in splits:
        train, test = frame.iloc[train_idx], frame.iloc[val_idx]
        predicted = fit_predict(prepare, train, test, scaler, encoder, model, select, threshold)
        row = {"f1": f1_score(test[TARGET], predicted),
               "accuracy": accuracy_score(test[TARGET], predicted)}
        if select is not None:
            # as colunas que o seletor manteve nesta divisão (W4: `kept`)
            row["kept"] = list(select.get_feature_names_out())
            row["colunas"] = len(row["kept"])
        rows.append(row)
    return pd.DataFrame(rows)


def sem(values):
    """Erro-padrão da média: quanto a média mudaria com outras divisões (W3)."""
    return values.std(ddof=1) / np.sqrt(len(values))


def compare(runs, reference, metric=METRIC):
    """Cada alternativa contra a linha de base, nas mesmas divisões (como na W3).

    Uma diferença menor do que 2 x SEM é ruído.
    """
    table = {}
    for name, run in runs.items():
        difference = run[metric] - reference[metric]
        table[name] = {metric: run[metric].mean(),
                       "accuracy": run["accuracy"].mean(),
                       "vs base": difference.mean(),
                       "SEM": sem(difference),
                       "melhor em": f"{int((difference > 0).sum())} de {len(difference)}"}
    return pd.DataFrame(table).T


def verdict(table):
    """melhor / pior se a diferença passar os 2 x SEM, senão empata (a regra da tarefa 2)."""
    return np.where(table["vs base"] > 2 * table["SEM"], "melhor",
                    np.where(table["vs base"] < -2 * table["SEM"], "pior", "empata"))


# ----------------------------------------------------------------------------- 2.1 valores impossíveis
# Regras de domínio da EDA (passos 9 a 11): True onde o valor é impossível
invalid_errors = {
    "days_since_previous_purchase": lambda values: values < 0,          # dias negativos (-4)
    "prior_mean_gap_days": lambda values: values < 0,                   # dias negativos (-4)
    "prior_mean_purchase_value": lambda values: (values < 0) | (values == 1e9),
    "spend_365d": lambda values: (values < 0) | (values == 1e9),
}


def invalid_flags(frame):
    """Uma coluna por regra, True onde o valor é impossível (como o domain_flags da W3).

    Mais a regra de duas colunas da EDA: uma média de compras não pode ser maior
    do que a maior dessas compras.
    """
    flags = pd.concat([rule(frame[c]) for c, rule in invalid_errors.items()], axis=1)
    flags["prior_mean_purchase_value"] |= (frame["prior_mean_purchase_value"]
                                           > frame["prior_max_purchase_value"] + 0.02)
    return flags


def blank_invalid(frame):
    """Os valores impossíveis passam a NaN (W3: "blank the cell").

    Não aprende nada com os dados, por isso aplica-se igual ao treino, à validação e ao teste.
    """
    out = frame.copy()
    out[list(invalid_errors)] = out[list(invalid_errors)].mask(invalid_flags(frame))
    return out


# ----------------------------------------------------------------------------- 2.4 distribuições e escala
# As numéricas com cauda longa (EDA, passo 7): todas menos random_noise e customer_tenure_days
SKEWED = ["Total a Pagar", "Total Bruto", "Total Liquido", "Total Imp_", "Total Portes",
          "Total Desc_ Global", "Total Desc_ Linha", "Custo", "N_ Detalhes",
          "invoice_count_current_purchase", "days_since_previous_purchase",
          "prior_purchase_count", "prior_total_spend", "prior_mean_purchase_value",
          "prior_std_purchase_value", "prior_min_purchase_value", "prior_max_purchase_value",
          "prior_mean_gap_days", "prior_std_gap_days", "purchase_count_30d", "spend_30d",
          "purchase_count_90d", "spend_90d", "purchase_count_180d", "spend_180d",
          "purchase_count_365d", "spend_365d"]


def prepare_log1p(train, test):
    """log1p nas colunas com cauda longa (W3, célula 100: log1p_of).

    Não aprende nada com os dados, por isso aplica-se igual ao treino e ao teste.
    """
    train, test = blank_invalid(train), blank_invalid(test)
    train[SKEWED], test[SKEWED] = np.log1p(train[SKEWED]), np.log1p(test[SKEWED])
    return prepare_base(train, test)


# ----------------------------------------------------------------------------- 2.7 preparação final
prepare_final = prepare_log1p   # valores impossíveis -> nulo, log1p, mediana / moda
ENCODER_FINAL = None            # None = one-hot com handle_unknown="ignore"
SCALER_FINAL = MinMaxScaler()


# ----------------------------------------------------------------------------- 3.4 / 3.5 seleção
def prepare_without(columns):
    """prepare_final e depois retira `columns`, antes do one-hot e do modelo.

    Retirar depois da preparação dá o mesmo que retirar antes (a mediana, a moda e o log1p de
    cada coluna não dependem das outras) e mantém a lista SKEWED do log1p e a regra de
    valores impossíveis do 2.1, que usa prior_max_purchase_value, tal como estavam.
    """
    def prepare(train, test):
        train, test = prepare_final(train, test)
        return train.drop(columns=columns), test.drop(columns=columns)
    return prepare


# As 14 colunas que a tarefa 3 retira (passos 48 a 51 e 60)
DROP_34 = ["Custo", "Total Bruto", "Total Imp_", "Total Liquido", "commercial_zone",
           "prior_purchase_count", "prior_std_gap_days", "prior_std_purchase_value",
           "purchase_count_180d", "purchase_count_30d", "purchase_count_365d", "random_noise",
           "spend_180d", "transport_method"]
prepare_selected = prepare_without(DROP_34)
SELECT_FINAL = None             # nenhum seletor por cima: as colunas já estão escolhidas
