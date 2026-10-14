# Databricks notebook source
# MAGIC %md
# MAGIC # Setup técnico da demo
# MAGIC Configura catálogo/schema e experimento MLflow e gera os dados sintéticos de atendimentos.
# MAGIC Para usar em outro workspace, altere apenas a célula **Configuração**.

# COMMAND ----------

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.dropdown("gen_synthetic_data", "false", ["true", "false"], "Generate Synthetic data for Drift Detection")
dbutils.widgets.dropdown("adv_mlops", "true", ["true", "false"], "Setup for advanced MLOps demo")
dbutils.widgets.dropdown("setup_adv_inference_data", "false", ["true", "false"], "Setup inference data for advanced demo")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
setup_adv_inference_data = dbutils.widgets.get("setup_adv_inference_data") == "true"
generate_synthetic_data = dbutils.widgets.get("gen_synthetic_data") == "true"

# COMMAND ----------

# DBTITLE 1,Configuração
import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
reformat_current_user = current_user.split("@")[0].lower().replace(".", "_")
current_user_az = re.sub(r'[^A-Za-z]', '', reformat_current_user)

catalog = "main"
dbName = db = "mlops_hiper"
online_store_name = "hiperstore" + current_user_az
n_beneficiarios = 3000  # tamanho da base sintética

# COMMAND ----------

import mlflow
import pandas as pd
import warnings
import logging
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

assert catalog not in ["hive_metastore", "spark_catalog"], "Esta demo requer Unity Catalog."
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{db}`")
spark.sql(f"USE `{catalog}`.`{db}`")

# Nomes compartilhados entre os notebooks
bronze_table_name = "atendimentos_brutos"
primary_keys = ["ID_PESSOA", "ID_USR_CONTR", "CD_TIPO_PRODUCAO_MEDICA", "dt_referencia"]
label_col = "classe_uso"
model_name = f"{catalog}.{db}.modelo_hiperutilizacao"
inference_table_name = "hiper_inferencia"

# COMMAND ----------

# Registry no Unity Catalog e pasta do experimento
mlflow.set_registry_uri("databricks-uc")
xp_path = f"/Users/{current_user}/mlops_hiper"
xp_name = "hiperutilizacao_experiment"
w = WorkspaceClient()
w.workspace.mkdirs(path=xp_path)
client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Gerador da base bruta sintética (uma linha por atendimento, sem PII)
def gerar_atendimentos_sinteticos(n_beneficiarios: int, seed: int = 42) -> pd.DataFrame:
  """Uma linha por atendimento, jan/2021 a dez/2025.
  Cauda pesada: fragilidade gamma por beneficiário, crônicos/internados com taxa maior e episódios agudos (mês 5x)."""
  import numpy as np
  rng = np.random.default_rng(seed)
  meses = pd.period_range("2021-01", "2025-12", freq="M").to_timestamp()
  tipos = np.array(["CONSUL", "MEDIC", "MATERI", "DIARIA"])
  taxa_base = np.array([0.6, 0.25, 0.2, 0.03])         # atendimentos por mês
  custo_medio = np.array([120.0, 80.0, 60.0, 1500.0])  # R$ por atendimento
  especialidades = np.array(["CLINICA MEDICA", "CARDIOLOGIA", "ORTOPEDIA", "GINECOLOGIA", "PEDIATRIA", "DERMATOLOGIA",
                             "OFTALMOLOGIA", "PSIQUIATRIA", "ENDOCRINOLOGIA", "NEUROLOGIA", "UROLOGIA", "OTORRINOLARINGOLOGIA"])
  n = n_beneficiarios

  # Beneficiários: atributos constantes no tempo. Nascimento no dia 1 mantém a faixa etária estável dentro do mês.
  idade_2021 = rng.integers(0, 95, n)
  nascimento = pd.to_datetime([f"{2021 - a}-{m:02d}-01" for a, m in zip(idade_2021, rng.integers(1, 13, n))])
  benef = pd.DataFrame({
    "ID_PESSOA": np.arange(100_000, 100_000 + n),
    "ID_USR_CONTR": rng.choice(np.arange(9_000_000, 9_900_000), n, replace=False),
    "SEXO": rng.choice(["FEMININO", "MASCULINO"], n, p=[0.55, 0.45]),
    "DT_NASCIMENTO": nascimento.strftime("%Y-%m-%d"),
    "DS_STATUS_USUARIO": rng.choice(["ATIVO", "INATIVO"], n, p=[0.9, 0.1]),
    "GRUPO": rng.choice(np.array(["ADAPTADO", None], dtype=object), n, p=[0.25, 0.75]),  # nulo vira "POSTERIOR A LEI"
    "PCT_CRONICO": rng.choice(["S", "N"], n, p=[0.2, 0.8]),
    "PCT_INTERNADO": rng.choice(["S", "N"], n, p=[0.08, 0.92]),
  })
  idade_hoje = (pd.Timestamp("2026-01-01") - nascimento).days // 365.25
  faixas = ["00 a 10 Anos"] + [f"{i} a {i + 9} Anos" for i in range(11, 91, 10)] + ["91 a 99 Anos", "Acima de 99 anos"]
  benef["FAIXA_ETARIA"] = pd.cut(idade_hoje, [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 200], labels=faixas).astype(str)
  fator = rng.gamma(0.8, 1 / 0.8, n) * (1 + idade_2021 / 60) * np.where(benef.PCT_CRONICO == "S", 2.0, 1.0)
  internado = benef.PCT_INTERNADO.values == "S"

  # Grade beneficiário x mês x tipo -> nº de atendimentos (Poisson)
  b, m, t = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(len(meses)), np.arange(len(tipos)), indexing="ij"))
  episodio = rng.random(b.size) < 0.02
  lam = taxa_base[t] * fator[b] * np.where(episodio, 5.0, 1.0) * np.where(internado[b] & (t > 0), 1.8, 1.0)
  qtd = rng.poisson(lam)
  b, m, t = np.repeat(b, qtd), np.repeat(m, qtd), np.repeat(t, qtd)

  # Uma linha por atendimento: especialidade (80% entre as 3 favoritas), data (dias 5-25) e valores
  linhas = b.size
  favoritas = rng.integers(0, len(especialidades), (n, 3))
  esp = np.where(rng.random(linhas) < 0.8, favoritas[b, rng.integers(0, 3, linhas)], rng.integers(0, len(especialidades), linhas))
  dt = meses[m] + pd.to_timedelta(rng.integers(4, 25, linhas), unit="D") + pd.to_timedelta(rng.integers(0, 86_400, linhas), unit="s")
  vr_custo = np.round(custo_medio[t] * rng.lognormal(-0.4, 0.9, linhas), 2)

  df = benef.iloc[b].reset_index(drop=True)
  df["CD_TIPO_PRODUCAO_MEDICA"] = tipos[t]
  df["DS_ESPECIALIDADE"] = especialidades[esp]
  df["DT_ATENDIMENTO"] = dt.strftime("%Y-%m-%d %H:%M:%S")
  df["VR_CUSTO"] = vr_custo
  df["VR_TOTAL"] = np.round(vr_custo * rng.uniform(1.0, 1.3, linhas), 2)
  df["VR_INTERCAMBIO"] = np.where(rng.random(linhas) < 0.05, np.round(vr_custo * 0.3, 2), 0.0)
  return df

# COMMAND ----------

# DBTITLE 1,Cria a tabela bruta (bronze)
if reset_all_data or not spark.catalog.tableExists(bronze_table_name):
  try:
    print(f"Deleting existing monitors for {catalog}.{db}.{inference_table_name}")
    w.quality_monitors.delete(table_name=f"{catalog}.{db}.{inference_table_name}")
  except Exception as error:
    print(f"Error deleting monitor: {type(error).__name__}")
  experiment_details = client.get_experiment_by_name(f"{xp_path}/{xp_name}")
  if experiment_details:
    print(f"Deleting experiment: {experiment_details.experiment_id}")
    client.delete_experiment(experiment_details.experiment_id)

  print(f"creating `{bronze_table_name}` raw table")
  spark.createDataFrame(gerar_atendimentos_sinteticos(n_beneficiarios)) \
       .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_table_name)

# COMMAND ----------

# DBTITLE 1,Tabela de chaves para inferência (sem label)
if setup_adv_inference_data:
  if spark.catalog.tableExists("hiper_labels"):
    print("Creating table with beneficiary keys for inference...")
    spark.read.table("hiper_labels").select(*primary_keys, "split") \
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("hiper_ids_inferencia")
  else:
    print("Label table `hiper_labels` doesn't exist, please run the notebook '01_feature_engineering'")

# COMMAND ----------

# DBTITLE 1,Gera dados de inferência sintéticos com drift (notebook 08)
def generate_synthetic(inference_table):
  """Simula 30 dias de inferências com mais internados/crônicos e label deslocado para hiper_utilizacao."""
  import dbldatagen as dg
  import pyspark.sql.functions as F
  from datetime import datetime, timedelta

  model_version = client.get_model_version_by_alias(name=model_name, alias="Champion").version
  n_days_to_back_fill = 30
  inicio, fim = datetime.now() - timedelta(days=n_days_to_back_fill), datetime.now()
  faixas = ["00 a 10 Anos"] + [f"{i} a {i + 9} Anos" for i in range(11, 91, 10)] + ["91 a 99 Anos"]

  generation_spec = (
    dg.DataGenerator(sparkSession=spark, name="synthetic_data", rows=2000 * n_days_to_back_fill, random=True)
    .withColumn("ID_PESSOA", "long", minValue=100000, maxValue=999999)
    .withColumn("ID_USR_CONTR", "long", minValue=9000000, maxValue=9900000)
    .withColumn("CD_TIPO_PRODUCAO_MEDICA", "string", values=["CONSUL", "MEDIC", "MATERI", "DIARIA"], random=True, weights=[5, 2, 2, 1])
    .withColumn("dt_referencia", "timestamp", begin=inicio, end=fim, interval="1 day", random=True)
    .withColumn("FAIXA_ETARIA_ATENDIMENTO", "string", values=faixas, random=True)
    .withColumn("SEXO", "string", values=["FEMININO", "MASCULINO"], random=True)
    .withColumn("STATUS", "string", values=["ATIVO", "INATIVO"], random=True, weights=[9, 1])
    .withColumn("GRUPO", "string", values=["POSTERIOR A LEI", "ADAPTADO"], random=True, weights=[3, 1])
    .withColumn("PCT_INTERNADO", "string", values=["S", "N"], random=True, weights=[3, 7])
    .withColumn("PCT_CRONICO", "string", values=["S", "N"], random=True, weights=[4, 6])
    .withColumn("mes", "int", minValue=1, maxValue=12, random=True)
    .withColumn("custo_total_mes", "double", minValue=50.0, maxValue=20000.0, random=True)
    .withColumn("custo_assistencial_mes", "double", minValue=40.0, maxValue=16000.0, random=True)
    .withColumn("qtd_especialidades_mes", "long", minValue=1, maxValue=6, random=True)
    .withColumn("custo_por_especialidade", "double", expr="custo_total_mes / qtd_especialidades_mes",
                baseColumn=["custo_total_mes", "qtd_especialidades_mes"])
    .withColumn("classe_uso", "string", values=["uso_tipico", "uso_elevado", "hiper_utilizacao"], random=True, weights=[2, 3, 5])
    .withColumn("prediction", "string", values=["uso_tipico", "uso_elevado", "hiper_utilizacao"], random=True, weights=[6, 2, 2])
    .withColumn("inference_timestamp", "timestamp", begin=inicio, end=fim, interval="1 hour", random=True)
  )
  generation_spec.build().withColumn("model_version", F.lit(model_version)) \
    .write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.{inference_table}")

if generate_synthetic_data:
  generate_synthetic(inference_table=inference_table_name)
