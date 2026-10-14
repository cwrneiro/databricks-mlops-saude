# Databricks notebook source
# MAGIC %md
# MAGIC #  Aprovação do modelo `Challenger` (Model Deployment Job)
# MAGIC
# MAGIC Este notebook verifica as tags de aprovação do novo modelo candidato (__Challenger__).
# MAGIC
# MAGIC Permite ter um "Human-In-The-Loop" _OPCIONAL_ para aprovar uma nova versão do modelo como parte do seu [deployment job](https://docs.databricks.com/aws/en/mlflow/deployment-job)
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-4b-v2.png?raw=true" width="1200">
# MAGIC
# MAGIC *Observação: em um setup típico de MLOps, isto roda como parte do deployment job automatizado. Aqui também pode ser executado como notebook interativo.*
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet mlflow-skinny --upgrade --pre
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

dbutils.widgets.text("model_name", model_name, "Model Name") # Preenchido pelos parâmetros do Deployment Job
try:
  versao_padrao = client.get_model_version_by_alias(model_name, "Challenger").version # Padrão: versão @Challenger
except Exception:
  versao_padrao = "1" # alias ainda não existe (primeira execução)
dbutils.widgets.text("model_version", versao_padrao, "Model Version") # Preenchido pelos parâmetros do Deployment Job
dbutils.widgets.text("approval_tag_name", "Approval_Check", "Approval Tag to check")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Busca as informações do modelo
# MAGIC
# MAGIC Buscamos no Unity Catalog as informações da versão do modelo.

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient


client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Valida a nova versão (Challenger) para Champion
# MAGIC
# MAGIC Checagem das tags de verificação.

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
tag_name = dbutils.widgets.get("approval_tag_name") # Default = approval task name

# Tags da versão do modelo no UC
tags = client.get_model_version(model_name, model_version).tags

# Verifica se existe a tag de aprovação
if not any(tag == tag_name for tag in tags.keys()):
  raise Exception("Model version not approved for deployment")

else:
  # Se a tag existe, verifica se está aprovada
  if tags.get(tag_name).lower() == "approved":
    print("Model version approved for deployment")
    
    client.set_registered_model_alias(
      name=model_name,
      alias="Champion",
      version=model_version
    )

  else:
    raise Exception("Model version not approved for deployment")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parabéns, o modelo foi "Aprovado"
# MAGIC
# MAGIC Sabemos agora que o modelo está pronto para pipelines de inferência e endpoints de serving em tempo real, pois atende aos nossos padrões de validação.
# MAGIC
# MAGIC
# MAGIC Próximo: [Inferência batch com o modelo Champion]($./05_batch_inference)