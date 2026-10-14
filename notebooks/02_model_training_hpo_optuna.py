# Databricks notebook source
# MAGIC %md
# MAGIC # Treino com HPO usando Optuna e MLflow
# MAGIC
# MAGIC Vamos rodar alguns treinos para testar diferentes modelos de classificação da utilização assistencial (`uso_tipico`, `uso_elevado`, `hiper_utilizacao`).
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-2-v2.png?raw=True" width="1200">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC databricks-feature-engineering==0.13.0a8
# MAGIC mlflow==3.3.2
# MAGIC optuna>=4.4.0
# MAGIC DBR/MLR16.4LTS (Recomendado - cluster sem autoscaling) OU Serverless v2
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow --upgrade lightgbm optuna
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monta o conjunto de treino
# MAGIC Carrega os dados diretamente da feature store
# MAGIC

# COMMAND ----------

display(spark.table("hiper_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Também usamos uma feature function para features calculadas sob demanda.
# MAGIC
# MAGIC Lembre que definimos a função `custo_por_especialidade` no [notebook de feature engineering]($./01_feature_engineering)

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE FUNCTION custo_por_especialidade

# COMMAND ----------

# MAGIC %md
# MAGIC Cria as especificações de features.
# MAGIC
# MAGIC O feature lookup define quais tabelas usar como feature tables e a chave para buscar os valores das features.
# MAGIC
# MAGIC A feature function define quais colunas da feature table são passadas como entrada da função.
# MAGIC
# MAGIC O Feature Engineering client usa essas definições para montar o conjunto de treino a partir da tabela de labels, da feature table e da feature function.

# COMMAND ----------

# DBTITLE 1,Define feature lookups
from databricks.feature_store import FeatureFunction, FeatureLookup

feature_lookups_n_functions = [
    FeatureLookup(
      table_name=f"{catalog}.{db}.hiper_features",
      lookup_key=["ID_PESSOA", "ID_USR_CONTR", "CD_TIPO_PRODUCAO_MEDICA"],
      timestamp_lookup_key="dt_referencia"
    ),
    FeatureFunction(
      udf_name=f"{catalog}.{db}.custo_por_especialidade",
      input_bindings={"custo_total_mes_in": "custo_total_mes", "qtd_especialidades_mes_in": "qtd_especialidades_mes"},
      output_name="custo_por_especialidade"
    )
]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Lê a tabela de labels.

# COMMAND ----------

# DBTITLE 1,Labels para treino/teste
# Apenas chaves, label e split: score_hiper e qtd_atendimentos_mes definem o label e ficam fora do modelo
labels_df = spark.read.table("hiper_labels").select(*primary_keys, label_col, "split")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Especificação do conjunto de treino
# MAGIC
# MAGIC Define como o conjunto de treino é montado a partir da tabela de labels, da feature table e da feature function usando o método `create_training_set()` do FE ([AWS](https://docs.databricks.com/aws/en/machine-learning/feature-store/python-api)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.databricks.com/gcp/en/machine-learning/feature-store/python-api))
# MAGIC
# MAGIC Os joins entre features e labels são [Point-In-Time](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series): cada label recebe as features do seu mês de referência.

# COMMAND ----------

# DBTITLE 1,Create Training specifications
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

def build_training_set(split: str):
  """Conjunto de um split. O split é filtrado antes para não virar coluna de entrada exigida pelo modelo."""
  return fe.create_training_set(
    df=labels_df.filter(f"split = '{split}'").drop("split"),
    label=label_col,
    feature_lookups=feature_lookups_n_functions,
    exclude_columns=["ID_PESSOA", "ID_USR_CONTR", "dt_referencia"],  # CD_TIPO_PRODUCAO_MEDICA continua como feature
    exclude_null_labels=True,
  )

training_set_specs = build_training_set("train")  # registrado junto com o modelo

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Com a especificação, montamos o conjunto de treino.
# MAGIC
# MAGIC `training_set_specs.load_df()` retorna um DataFrame PySpark, que convertemos para Pandas.

# COMMAND ----------

# DBTITLE 1,Load training set as Pandas dataframe
training_pdf = training_set_specs.load_df().toPandas()
test_pdf = build_training_set("test").load_df().toPandas()

# Nota: amostra de 30k linhas para o HPO caber no tempo de uma demo ao vivo; remova para treinar na base toda
training_pdf = training_pdf.sample(n=min(30_000, len(training_pdf)), random_state=2025)

X_train, Y_train = (training_pdf.drop(label_col, axis=1), training_pdf[label_col])

# Mesmo conjunto de teste (2025) para todos os runs: reprodutível e comparação justa entre modelos
X_test, Y_test = (test_pdf.drop(label_col, axis=1), test_pdf[label_col])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pré-processamento

# COMMAND ----------

# MAGIC %md
# MAGIC ### Colunas booleanas
# MAGIC Para cada coluna, imputa valores faltantes e converte para zeros e uns.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


bool_imputers = []

bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
    ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
])

bool_transformers = [("boolean", bool_pipeline, ["SEXO", "STATUS", "GRUPO", "PCT_INTERNADO", "PCT_CRONICO"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Colunas numéricas
# MAGIC
# MAGIC Valores faltantes são imputados com a média.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler


num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["custo_total_mes", "custo_assistencial_mes", "qtd_especialidades_mes", "custo_por_especialidade", "mes"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["custo_total_mes", "custo_assistencial_mes", "qtd_especialidades_mes", "custo_por_especialidade", "mes"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Colunas categóricas

# COMMAND ----------

# MAGIC %md
# MAGIC #### Categóricas de baixa cardinalidade
# MAGIC Cada coluna categórica vira várias colunas binárias via one-hot encoding (uma por valor distinto).

# COMMAND ----------

one_hot_imputers = []
one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", SklearnOneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["FAIXA_ETARIA_ATENDIMENTO", "CD_TIPO_PRODUCAO_MEDICA"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer


transformers = bool_transformers + numerical_transformers + categorical_one_hot_transformers
preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escalando a otimização de hiperparâmetros (HPO) com Optuna, Spark e MLflow
# MAGIC
# MAGIC Optuna é um framework avançado de otimização de hiperparâmetros para machine learning. Ele otimiza principalmente por meio de:
# MAGIC
# MAGIC 1. Algoritmos de amostragem:
# MAGIC     1. **Tree-structured Parzen Estimator (padrão)** - otimização Bayesiana para explorar o espaço de hiperparâmetros com eficiência.
# MAGIC     2. Random Sampling - amostra valores aleatoriamente no espaço de busca.
# MAGIC     3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - algoritmo evolutivo para problemas de otimização difíceis, não lineares e não convexos.
# MAGIC     4. Grid Search - busca exaustiva em um subconjunto definido manualmente.
# MAGIC     5. Quasi-Monte Carlo - usa sequências de baixa discrepância para cobrir o espaço de forma mais uniforme que a amostragem aleatória.
# MAGIC     6. NSGA-II (Non-dominated Sorting Genetic Algorithm II) - algoritmo de otimização multiobjetivo.
# MAGIC     7. Amostragem com processos Gaussianos (Kriging) - otimização Bayesiana com processos Gaussianos.
# MAGIC     8. Samplers customizados, herdando da classe `BaseSampler`.
# MAGIC 2. Pruning: interrompe cedo trials pouco promissores
# MAGIC 3. Objeto Study: gerencia a otimização. O método study.optimize() inicia a otimização, recebendo a função objetivo e o número de trials.
# MAGIC 4. Execução paralela: em um único nó ou em múltiplos nós.
# MAGIC
# MAGIC
# MAGIC Usaremos os novos objetos `MLflowStorage` e `MlflowSparkStudy`.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Função objetivo e espaço de busca
# MAGIC O espaço de busca é definido chamando funções como `suggest_categorical`, `suggest_float` e `suggest_int` no objeto Trial passado à função objetivo. O Optuna permite definir o espaço de busca dinamicamente.
# MAGIC
# MAGIC Como `uso_elevado` e `hiper_utilizacao` representam ~5% dos casos cada, otimizamos o **F1 macro** (todas as classes com o mesmo peso) e treinamos com `class_weight="balanced"`.
# MAGIC
# MAGIC Consulte a documentação:
# MAGIC * [optuna.samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) para os samplers
# MAGIC * [optuna.pruners](https://optuna.readthedocs.io/en/stable/reference/pruners.html) para os pruners
# MAGIC * [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) para a lista completa de funções de espaço de busca.

# COMMAND ----------

import os
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def build_classifier(classifier_name: str, params: dict, rng_seed: int):
  """Instancia o classificador. class_weight='balanced' porque hiper_utilizacao/uso_elevado são ~5% cada."""
  if classifier_name == "LogisticRegression":
    return LogisticRegression(**params, class_weight="balanced", max_iter=1000, random_state=rng_seed)
  if classifier_name == "RandomForest":
    return RandomForestClassifier(**params, class_weight="balanced", random_state=rng_seed)
  return LGBMClassifier(**params, class_weight="balanced", force_row_wise=True, verbose=-1, random_state=rng_seed)


class ObjectiveOptuna(object):
  """Função objetivo do Optuna: recebe os dados no construtor para não recarregá-los a cada trial."""
  def __init__(self, X_train_in: pd.DataFrame, Y_train_in: pd.Series, preprocessor_in: ColumnTransformer, rng_seed: int = 2025):
    self.preprocessor = preprocessor_in
    self.rng_seed = rng_seed
    self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
      X_train_in, Y_train_in, test_size=0.1, random_state=rng_seed, stratify=Y_train_in)

  def __call__(self, trial):
    """Treina um pipeline e retorna o F1 macro na validação (métrica a maximizar)."""
    # Roda nos executors do Spark: lá o Databricks SDK não tem credenciais, então o MLflow
    # deve registrar os artefatos dos trials pela API REST (que funciona nos executors)
    os.environ["MLFLOW_DISABLE_DATABRICKS_SDK_FOR_RUN_ARTIFACTS"] = "true"
    classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "RandomForest", "LightGBM"])

    if classifier_name == "LogisticRegression":
      params = {"C": trial.suggest_float("C", 1e-2, 1, log=True),
                "tol": trial.suggest_float("tol", 1e-6, 1e-3, step=1e-6)}
    elif classifier_name == "RandomForest":
      params = {"n_estimators": trial.suggest_int("n_estimators", 10, 200, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)}
    else:
      params = {"n_estimators": trial.suggest_int("n_estimators", 10, 200, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.9),
                "max_bin": trial.suggest_int("max_bin", 2, 256),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256)}

    this_model = Pipeline(steps=[("preprocessor", self.preprocessor), ("classifier", build_classifier(classifier_name, params, self.rng_seed))])
    mlflow.sklearn.autolog(disable=True) # Desliga o autolog para não registrar artefatos a cada trial
    this_model.fit(self.X_train, self.Y_train)
    return f1_score(self.Y_val, this_model.predict(self.X_val), average="macro")

# COMMAND ----------

# DBTITLE 1,Define the Sampler
# Um sampler novo para cada study: o TPESampler guarda estado do primeiro study em que é usado
def new_sampler():
  return optuna.samplers.TPESampler(seed=2025) # Semente do gerador de números aleatórios

# COMMAND ----------

# DBTITLE 1,Define the Pruner
from optuna.pruners import BasePruner


class NoneValuePruner(BasePruner):
  """Pruner customizado para ignorar trials que falharam (valor None)."""

  def prune(self, study, trial):
    # Se o valor do trial é None, descarta
    if trial.value is None:
      return True

    else:
      return False

# COMMAND ----------

# MAGIC %md
# MAGIC #### Teste rápido local

# COMMAND ----------

objective_fn = ObjectiveOptuna(X_train, Y_train, preprocessor)

# COMMAND ----------

# DBTITLE 1,without mlflow
study_debug = optuna.create_study(direction="maximize", study_name="test_debug", sampler=new_sampler(), pruner= NoneValuePruner())
study_debug.optimize(objective_fn, n_trials=4, n_jobs=-1)

# COMMAND ----------

print("Best trial:")
best_trial = study_debug.best_trial
print(f"  F1 macro: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Usando o flavor nativo `MLFlow.spark.optuna`: storage compartilhado para otimização distribuída
# MAGIC Com o `MlflowStorage`, o MLflow Tracking Server serve como backend de storage. Veja a documentação para [AWS](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl-hyperparam-tuning/optuna)|[GCP](https://docs.databricks.com/gcp/en/machine-learning/automl-hyperparam-tuning/optuna).
# MAGIC
# MAGIC **Observação:** a métrica objetivo de cada trial é registrada como `value`.

# COMMAND ----------

# DBTITLE 1,Point/Set experiment
experiment_name = f"{xp_path}/{xp_name}" # Experimento compartilhado (Staging/Prod)
# experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get() # Experimento do notebook (Dev)

try:
  print(f"Loading experiment: {experiment_name}")
  experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

except Exception as e:
  print(f"Creating experiment: {experiment_name}")
  experiment_id = mlflow.create_experiment(name=experiment_name, tags={"demo": "hiperutilizacao"})

# COMMAND ----------

import mlflow
from mlflow.optuna.storage import MlflowStorage
from mlflow.pyspark.optuna.study import MlflowSparkStudy


mlflow.set_experiment(f"{xp_path}/{xp_name}")
print(f"Set experiment to: {xp_name}")

mlflow_storage = MlflowStorage(experiment_id=experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Teste rápido local
# MAGIC com MLflow

# COMMAND ----------

mlflow_optuna_study_debug = MlflowSparkStudy(
  pruner= NoneValuePruner(),
  sampler=new_sampler(),
  study_name="mlflow-smoke-test",
  storage=mlflow_storage,
)
mlflow_optuna_study_debug._directions = ["maximize"]

mlflow_optuna_study_debug.optimize(objective_fn, n_trials=4, n_jobs=4)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Função principal de treino

# COMMAND ----------

display(X_train)

# COMMAND ----------

import warnings
import pandas as pd
from mlflow.types.utils import _infer_schema
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc


def optuna_hpo_fn(n_trials: int, X_train: pd.DataFrame, Y_train: pd.Series, X_test: pd.DataFrame, Y_test: pd.Series, training_set_specs_in, preprocessor_in: ColumnTransformer, experiment_id: str, rng_seed_in: int = 2025, run_name:str = "spark-mlflow-tuning", optuna_sampler_in: optuna.samplers.TPESampler = None, optuna_pruner_in: optuna.pruners.BasePruner = None, n_jobs: int = 2) -> optuna.study.study.Study:
    """
    Aumentar `n_jobs` pode fazer o experimento falhar por trials que retornam None e não podem ser descartados em modo paralelo
    """

    # Dispara o HPO distribuído como runs aninhados
    objective_fn = ObjectiveOptuna(X_train, Y_train, preprocessor_in, rng_seed_in)
    mlflow_optuna_study = MlflowSparkStudy(
        pruner=optuna_pruner_in,
        sampler=optuna_sampler_in,
        study_name=run_name,
        storage=mlflow_storage,
    )
    mlflow_optuna_study._directions = ["maximize"]
    mlflow_optuna_study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)

    # Reproduz o melhor classificador
    best_model_params = dict(mlflow_optuna_study.best_params)
    classifier_type = best_model_params.pop("classifier")
    best_model = build_classifier(classifier_type, best_model_params, rng_seed_in)

    # Logging automático de amostras de entrada, métricas e parâmetros
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False, silent=True)

    active_run = mlflow.active_run()
    if not active_run:
        active_run = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1)[0]

    run_id = active_run.info.run_id

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        # Treina o melhor modelo e registra com o FE client no run pai
        model_pipeline = Pipeline(steps=[("preprocessor", objective_fn.preprocessor), ("classifier", best_model)])
        model_pipeline.fit(X_train, Y_train)

        # Avalia o modelo e registra no experimento
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model_pipeline)

        # Métricas no conjunto de treino
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(label_col):Y_train}),
            targets=label_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False, "metric_prefix": "training_"}
        )

        # Métricas no conjunto de teste
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(label_col):Y_test}),
            targets=label_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False, "metric_prefix": "test_"}
        )

        # Métrica usada para seleção/validação do modelo (F1 macro, todas as classes com o mesmo peso)
        mlflow.log_metric("test_f1_macro", f1_score(Y_test, model_pipeline.predict(X_test), average="macro"))

        # Registra o modelo com os feature lookups embutidos
        fe.log_model(
            model=model_pipeline,
            artifact_path="model",
            flavor=mlflow.sklearn,
            training_set=training_set_specs_in,
            # env_manager="uv"
        )
        mlflow.end_run()

    return mlflow_optuna_study

# COMMAND ----------

# MAGIC %md
# MAGIC ### Executa os runs de treino

# COMMAND ----------

# Executa o treino, distribuído entre os workers/nós
distributed_study = optuna_hpo_fn(
  n_trials=4, # Aumente (ex.: 16) fora de demos ao vivo
  X_train=X_train,
  X_test=X_test,
  Y_train=Y_train,
  Y_test=Y_test,
  training_set_specs_in=training_set_specs,
  preprocessor_in=preprocessor,
  experiment_id=experiment_id,
  rng_seed_in=2025,
  run_name="hiper-hpo-best-run",
  optuna_sampler_in=new_sampler(),
  optuna_pruner_in=NoneValuePruner(),
  n_jobs=4, # Trials em paralelo
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automatiza a validação para promoção do modelo
# MAGIC
# MAGIC Próximos passos:
# MAGIC - [Criar o Model Deployment Job]($./03a_create_deployment_job)
# MAGIC - [Buscar o melhor run e disparar a validação para promoção]($./03b_from_notebook_to_models_in_uc)
