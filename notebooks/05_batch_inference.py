# Databricks notebook source
# MAGIC %md
# MAGIC # Inferência batch - Classe de utilização
# MAGIC
# MAGIC ## Inferência com o modelo Champion
# MAGIC
# MAGIC Modelos no Unity Catalog podem ser carregados em pipelines de inferência batch. As predições geradas orientam a priorização da gestão de cuidado e alimentam análises. Usamos o modelo __@Champion__.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-5-v2.png?raw=true" width="1200">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC databricks-feature-engineering==0.13.0a8
# MAGIC mlflow==3.3.2
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow lightgbm --upgrade
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true $setup_adv_inference_data=true

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Usando o modelo para inferências batch
# MAGIC
# MAGIC Com o modelo disponível no Model Registry do Unity Catalog, podemos carregá-lo para calcular as inferências e salvá-las em uma tabela para análises e dashboards.
# MAGIC
# MAGIC Usamos o método `score_batch` do feature engineering client, que carrega automaticamente um PySpark UDF e distribui a inferência no cluster. Para volumes pequenos, também dá para carregar o modelo em Python puro e usar um DataFrame Pandas.
# MAGIC
# MAGIC Se não souber por onde começar, há código de exemplo na página __"Artifacts"__ do run do modelo no experimento.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inferência batch com o modelo Champion
# MAGIC
# MAGIC Usamos o `score_batch` do feature engineering client para gerar as predições dos beneficiários.
# MAGIC
# MAGIC Para simplificar, assumimos que as features já foram calculadas para todos os beneficiários e estão na feature table. Na prática isso é feito por pipelines de feature engineering separados.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reproduz o ambiente de inferência no notebook _(OPCIONAL)_
# MAGIC SOMENTE se for executar a inferência batch no ambiente padrão do Serverless (`env_manager="local"`). Caso contrário não é necessário: a inferência pode rodar em um ambiente virtual (`env_manager="virtualenv"` ou `"uv"`) criado a partir dos requirements do modelo, ou no mesmo MLR usado no treino.

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
# MAGIC
# MAGIC
# MAGIC requirements_path = ModelsArtifactRepository(f"models:/{model_name}@Champion").download_artifacts(artifact_path="requirements.txt") # baixa os requirements do registry
# MAGIC ```
# MAGIC -----------------------------------------------------------------
# MAGIC ```bash
# MAGIC %pip install --quiet -r $requirements_path
# MAGIC
# MAGIC
# MAGIC %restart_python
# MAGIC ```
# MAGIC ----------------------------------------------------------------
# MAGIC ```bash
# MAGIC %run ./_resources/00-setup $adv_mlops=true $setup_adv_inference_data=true
# MAGIC ```

# COMMAND ----------

env_manager = "local" # Mesmo ambiente serverless do treino; use "virtualenv" para recriar o ambiente do modelo em clusters clássicos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Usa o `fe.score_batch()` com a URI do modelo
# MAGIC Busca as chaves dos beneficiários a classificar e roda a inferência com Spark

# COMMAND ----------

# DBTITLE 1,Set model alias to use for batch inference
model_alias = "Champion" # "Challenger"

# URI do modelo
model_uri = f"models:/{model_name}@{model_alias}"
# model_uri = f"models:/{model_name}/{model_version}"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()

# Chaves de beneficiários do período de teste (2025) a serem classificadas
inference_df = spark.read.table("hiper_ids_inferencia").filter("split='test'").limit(5000)

# Batch score
preds_df = fe.score_batch(df=inference_df, model_uri=model_uri, result_type="string", env_manager=env_manager)
display(preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Pronto! As predições podem ser salvas em uma tabela e usadas pelo time de gestão de saúde para priorizar o acompanhamento dos beneficiários com hiperutilização.
# MAGIC
# MAGIC Os dados também ficam disponíveis no Genie para responder perguntas em linguagem natural!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Salva as predições para monitoramento
# MAGIC
# MAGIC Como queremos monitorar o modelo e suas predições ao longo do tempo, salvamos as predições junto com a informação do modelo que as gerou. Essa tabela pode ser monitorada para drift de features e de predição.
# MAGIC
# MAGIC Esta tabela não tem os labels (ground truth). Normalmente eles chegam com o tempo e, em muitos casos, podem nem estar disponíveis. Mesmo assim podemos monitorar o drift dos dados, que por si só pode indicar a necessidade de retreino.
# MAGIC
# MAGIC A tabela é salva como `hiper_inferencia_offline`, com a versão do modelo, as predições e o timestamp da inferência, sem labels.
# MAGIC

# COMMAND ----------

from mlflow import MlflowClient


client = MlflowClient()

model = client.get_registered_model(name=model_name)
model_version = client.get_model_version_by_alias(name=model_name, alias=model_alias).version

# COMMAND ----------

from datetime import datetime, timedelta
from pyspark.sql import functions as F


offline_inference_df = preds_df.drop("split") \
                              .withColumn("model_version", F.lit(model_version)) \
                              .withColumn("inference_timestamp", F.lit(datetime.now())) # - timedelta(days=1)))

offline_inference_df.write.mode("append") \
                    .option("overwriteSchema", True) \
                    .saveAsTable("hiper_inferencia_offline")

# display(offline_inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parabéns! O modelo foi usado em inferência batch.
# MAGIC
# MAGIC Agora vamos ver como implantar o modelo como endpoint REST para inferência em tempo real.
# MAGIC
# MAGIC Próximo:  [Servir as features e o modelo em tempo real]($./06_serve_features_and_model)
