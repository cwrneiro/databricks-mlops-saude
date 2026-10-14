# Demo MLOps end-to-end — Hiperutilização assistencial

Demo de MLOps de ponta a ponta na Databricks, baseada na parte *advanced* do dbdemos
[`mlops-end2end`](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/mlops-end-to-end-pipeline),
adaptada para a classificação de hiperutilização assistencial em uma operadora de saúde.

Os dados são **sintéticos**, no formato de uma base de atendimentos de operadora
(uma linha por atendimento), sem dados pessoais.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `notebooks/_resources/00-setup` | Configuração (catálogo/schema), nomes compartilhados e geradores de dados sintéticos |
| `notebooks/00_mlops_end2end_advanced` | Introdução ao caso de uso e ao fluxo de MLOps |
| `notebooks/01_feature_engineering` | Metodologia de classificação (score de hiperutilização → `classe_uso`), tabela de labels, feature table e função on-demand |
| `notebooks/02_model_training_hpo_optuna` | Treino multiclasse com Optuna + MLflow (F1 macro) |
| `notebooks/03a_create_deployment_job` | Cria o Model Deployment Job do MLflow 3 e liga ao modelo no UC |
| `notebooks/03b_from_notebook_to_models_in_uc` | Registra o melhor modelo no UC como `Challenger` (dispara o Deployment Job) |
| `notebooks/04a_challenger_validation` | Checagens e métrica de negócio; promove a `Champion` |
| `notebooks/04b_challenger_approval` | Aprovação (human-in-the-loop) via tag |
| `notebooks/05_batch_inference` | Inferência batch com `@Champion` |
| `notebooks/06_serve_features_and_model` | Online feature store (Lakebase) + endpoint de Model Serving |
| `notebooks/07_model_monitoring` | Lakehouse Monitoring da tabela de inferência |
| `notebooks/08_drift_detection` | Drift simulado, contagem de violações e gatilho de retreino |

## Pré-requisitos

- Workspace com Unity Catalog, compute serverless, Model Serving, Lakebase (online feature store) e Lakehouse Monitoring

## Como implantar e executar

1. No workspace, crie um Git folder (**Workspace → Create → Git folder**) apontando para este repositório.
2. Ajuste `catalog` e `db` na célula **Configuração** de `notebooks/_resources/00-setup` (padrão: `main.mlops_hiper`).
3. Rode os notebooks em ordem, com compute serverless:
   `00` → `01` → `02` → `03a` → `03b` (dispara o Deployment Job: `04a` → `04b` → `06`) → `05` → `07` → `08`.

Para recriar os dados do zero, rode `00_mlops_end2end_advanced` com o widget `reset_all_data = true`.

## Metodologia de classificação

- `score_hiper` = atendimentos do mês ÷ mediana do estrato (faixa etária, sexo, status, tipo de produção médica, ano, grupo, internado, crônico). Usa a mediana mensal se o estrato tiver ≥ 10 beneficiários; senão, a anual.
- Limites: percentis 90 e 95 globais de 2021–2024 → `uso_tipico` / `uso_elevado` (≥ p90) / `hiper_utilizacao` (≥ p95).
- O modelo prevê a classe do mês corrente **sem** as colunas que definem o label: `qtd_atendimentos_mes`, `E_Y_mes`, `E_Y_ano`, `n_estrato_mes`, `n_estrato_ano`, `score_hiper`, `tipo_referencia`, `p75..p99`, `n_estrato`.
- Split temporal: treino 2021–2024, teste 2025.

## Limpeza (custo)

O endpoint, a online store e o monitor geram custo. Para remover:

```python
from databricks.sdk import WorkspaceClient
from databricks.feature_engineering import FeatureEngineeringClient

w = WorkspaceClient()
w.serving_endpoints.delete("hiperutilizacao_<usuario>")
FeatureEngineeringClient().delete_online_store(name="hiperstore<usuario>")
w.quality_monitors.delete(table_name="<catalog>.mlops_hiper.hiper_inferencia")
```

Depois apague o job `Hiper-Model-Deployment-Job`, o experimento `/Users/<usuario>/mlops_hiper` e o schema `<catalog>.mlops_hiper`.

## Licença

Baseado no [dbdemos](https://github.com/databricks-demos/dbdemos) `mlops-end2end`, distribuído sob a Databricks License: veja [`notebooks/_resources/LICENSE`](notebooks/_resources/LICENSE.py) e [`NOTICE`](notebooks/_resources/NOTICE.py).
