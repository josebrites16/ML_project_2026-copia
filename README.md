# ML_project — 90-Day Repeat Purchase Prediction (Grupo 37)

Prevê, para cada compra, se o mesmo cliente volta a comprar nos 90 dias seguintes
(`repeat_purchase_90d`). Homework e projeto de Machine Learning 2026/2027, NOVA IMS.

## Como correr

1. **Dados:** copiar `train.csv`, `test.csv`, `clientes.csv` e `sample_submission.csv` da
   competição do Kaggle para a pasta `data/` (não estão no git).
2. **Ambiente:** `conda env create -f environment.yml` e depois `conda activate machine_learning`.
3. **Notebooks**, por esta ordem, com o kernel do ambiente `machine_learning`:

| Notebook | O que faz | Tempo aproximado |
|---|---|---|
| `notebooks/01_eda.ipynb` | Exploração dos dados (tarefa 1) | < 1 min |
| `notebooks/02_preprocessing.ipynb` | Pré-processamento e seleção de features (tarefas 2 e 3) | ~7 min |
| `notebooks/03_modelling.ipynb` | Modelos, limiar, modelo final e submissão (tarefa 4) | ~10 min |

`notebooks/preprocessing.py` tem a preparação final e a régua de avaliação; o 02 e o 03
importam-no, por isso tem de ficar na mesma pasta que os notebooks.

## Submissões

`submissions/ML_Group_37_VersionXX.csv`, geradas pela última célula do 03. O `VERSION` é
escrito à mão, e a célula recusa reescrever um ficheiro já existente com outras previsões.
