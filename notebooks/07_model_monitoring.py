# Databricks notebook source
# MAGIC %md
# MAGIC # Monitorando o modelo com Lakehouse Monitoring
# MAGIC Nesta etapa usamos o Databricks Lakehouse Monitoring ([AWS](https://docs.databricks.com/en/lakehouse-monitoring/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/)) para monitorar a tabela de inferência.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-7-v2.png?raw=true" width="1200">
# MAGIC
# MAGIC O Lakehouse Monitoring anexa um monitor a qualquer tabela Delta e gera os pipelines necessários para perfilar os dados e calcular métricas de qualidade. Basta dizer com que frequência essas métricas devem ser coletadas.
# MAGIC
# MAGIC Use o Lakehouse Monitoring para monitorar drift de dados, de label e de predição, além de métricas de qualidade do modelo. Ele permite monitorar estatísticas (perfis de dados) e drift em tabelas com:
# MAGIC * inferências de scoring batch
# MAGIC * logs de requisições de um endpoint de Model Serving ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html) |[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/inference-tables))
# MAGIC
# MAGIC O Lakehouse Monitoring guarda as métricas de qualidade e drift em duas tabelas criadas automaticamente para cada tabela monitorada:
# MAGIC - Tabela de métricas de perfil (sufixo `_profile_metrics`)
# MAGIC   - Métricas como porcentagem de nulos, estatísticas descritivas, métricas do modelo como acurácia, F1, fairness e bias etc.
# MAGIC - Tabela de métricas de drift (sufixo `_drift_metrics`)
# MAGIC   - Métricas como a variação ("delta") da porcentagem de nulos, das médias e testes estatísticos para detectar drift.
# MAGIC
# MAGIC Para simplificar, usamos as inferências batch como tabela de inferência e anexamos um monitor à tabela `hiper_inferencia`.
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk mlflow-skinny --upgrade
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cria o monitor
# MAGIC Vamos criar um monitor sobre a tabela de inferência.
# MAGIC É um setup feito uma única vez.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cria/atualiza a tabela de inferência
# MAGIC
# MAGIC Ela pode unir as inferências offline e online.
# MAGIC Para simplificar, criamos a tabela de inferência como cópia da primeira tabela de predições batch, já com os labels.
# MAGIC
# MAGIC Em outro cenário, poderíamos processar a tabela de inferência online e armazená-la junto com a offline.

# COMMAND ----------

# DBTITLE 1,First Time (or Full Overwrite)
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE hiper_inferencia AS
# MAGIC   SELECT * EXCEPT (split, score_hiper, qtd_atendimentos_mes)
# MAGIC   FROM hiper_inferencia_offline LEFT JOIN hiper_labels USING (ID_PESSOA, ID_USR_CONTR, CD_TIPO_PRODUCAO_MEDICA, dt_referencia)
# MAGIC   ORDER BY inference_timestamp;
# MAGIC
# MAGIC ALTER TABLE hiper_inferencia SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# DBTITLE 1,Subsequent calls: Update/Upserts Labels when available
# MAGIC %sql
# MAGIC MERGE INTO hiper_inferencia AS i
# MAGIC   USING hiper_labels AS l
# MAGIC   ON i.ID_PESSOA = l.ID_PESSOA AND i.ID_USR_CONTR = l.ID_USR_CONTR
# MAGIC      AND i.CD_TIPO_PRODUCAO_MEDICA = l.CD_TIPO_PRODUCAO_MEDICA AND i.dt_referencia = l.dt_referencia
# MAGIC   WHEN MATCHED THEN UPDATE SET i.classe_uso = l.classe_uso

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cria/atualiza a tabela de baseline
# MAGIC
# MAGIC No Lakehouse Monitoring, a "baseline table" é a referência para medir drift e qualidade. É opcional, mas valiosa: fornece um padrão contra o qual os dados monitorados são comparados.
# MAGIC Características e usos principais:
# MAGIC
# MAGIC **Referência para o cálculo de drift:**
# MAGIC O principal objetivo é fornecer uma distribuição estável e esperada dos dados. O Lakehouse Monitoring calcula o drift comparando o estado atual da tabela monitorada com a baseline, identificando mudanças de distribuição, valores e outras características ao longo do tempo.
# MAGIC
# MAGIC **Reflete a qualidade esperada:**
# MAGIC A baseline deve representar um conjunto de dados com a qualidade desejada, por exemplo um snapshot de um período de qualidade conhecida.
# MAGIC
# MAGIC **Consistência de schema:**
# MAGIC A baseline deve ter idealmente o mesmo schema da tabela monitorada. O Lakehouse Monitoring usa heurísticas para lidar com diferenças, mas um schema consistente garante métricas precisas.
# MAGIC
# MAGIC **Uso no perfil de inferência:**
# MAGIC Em monitoramento de modelos, a baseline pode representar a distribuição esperada de features e predições, permitindo detectar concept drift. Recomendamos usar o conjunto de teste/hold-out do modelo baseline e, a cada nova versão, predizer o mesmo conjunto e adicioná-lo à baseline.
# MAGIC
# MAGIC Para simplificar, criamos a baseline a partir da tabela `hiper_inferencia_offline`, restrita ao período de teste.

# COMMAND ----------

# DBTITLE 1,One-time then do appends for every new model version
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE hiper_baseline AS
# MAGIC   SELECT * EXCEPT (ID_PESSOA, ID_USR_CONTR, dt_referencia, inference_timestamp, split, score_hiper, qtd_atendimentos_mes)
# MAGIC   FROM hiper_inferencia_offline LEFT JOIN hiper_labels USING (ID_PESSOA, ID_USR_CONTR, CD_TIPO_PRODUCAO_MEDICA, dt_referencia)
# MAGIC   WHERE split = 'test';

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cria uma métrica customizada
# MAGIC
# MAGIC Métricas customizadas são calculadas automaticamente pelo Lakehouse Monitoring e servem para capturar lógica de negócio ou um score de qualidade próprio.
# MAGIC
# MAGIC Aqui calculamos o impacto de negócio de um erro do modelo: o **custo mensal médio de hiperutilizadores não detectados** pelo modelo (que deixam de receber acompanhamento).

# COMMAND ----------

# DBTITLE 1,Define expected loss metric
from pyspark.sql.types import DoubleType, StructField
from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType


custo_hiper_nao_detectado_metric = [
  MonitorMetric(
    type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
    name="custo_hiper_nao_detectado",
    input_columns=[":table"],
    definition="""avg(CASE
    WHEN {{prediction_col}} != {{label_col}} AND {{label_col}} = 'hiper_utilizacao' THEN custo_total_mes
    ELSE 0 END
    )""",
    output_data_type=StructField("output", DoubleType()).json()
  )
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cria o monitor
# MAGIC
# MAGIC Como monitoramos uma tabela de inferência (com predições de um modelo de ML), usamos o perfil [Inference](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/create-monitor-api#inferencelog-profile).

# COMMAND ----------

# DBTITLE 1,Create Monitor
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorCronSchedule


print(f"Creating monitor for inference table {catalog}.{db}.{inference_table_name}")
w = WorkspaceClient()

try:
  info = w.quality_monitors.create(
    table_name=f"{catalog}.{db}.{inference_table_name}",
    inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            timestamp_col="inference_timestamp",
            granularities=["1 day"],
            model_id_col="model_version",
            label_col=label_col, # opcional
    ),
    schedule=MonitorCronSchedule(
        quartz_cron_expression="0 0 12 * * ?",  # Todo dia ao meio-dia
        timezone_id="America/Sao_Paulo"
    ),
    assets_dir=f"{os.getcwd()}/monitoring", # Altere para outra pasta se necessário
    output_schema_name=f"{catalog}.{db}",
    baseline_table_name=f"{catalog}.{db}.hiper_baseline",
    slicing_exprs=["FAIXA_ETARIA_ATENDIMENTO", "PCT_CRONICO='S'"], # Dimensões de recorte
    custom_metrics=custo_hiper_nao_detectado_metric)

except Exception as lhm_exception:
  if "already exist" in str(lhm_exception).lower():
    print(f"Monitor for {catalog}.{db}.{inference_table_name} already exists, retrieving monitor info:")
    info = w.quality_monitors.get(table_name=f"{catalog}.{db}.{inference_table_name}")
  else:
    raise lhm_exception

# COMMAND ----------

# MAGIC %md
# MAGIC Aguarda/verifica a criação do monitor

# COMMAND ----------

import time
from databricks.sdk.service.catalog import MonitorInfoStatus, MonitorRefreshInfoState


# Aguarda a criação do monitor
while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
  info = w.quality_monitors.get(table_name=f"{catalog}.{db}.{inference_table_name}")
  time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"

# COMMAND ----------

# MAGIC %md
# MAGIC A criação do monitor **dispara um refresh inicial**: buscamos/aguardamos esse refresh ou disparamos um e esperamos terminar

# COMMAND ----------

def get_refreshes():
  return w.quality_monitors.list_refreshes(table_name=f"{catalog}.{db}.{inference_table_name}").refreshes

refreshes = get_refreshes()
if len(refreshes) == 0:
  # Dispara um refresh se não houver nenhum
  w.quality_monitors.run_refresh(table_name=f"{catalog}.{db}.{inference_table_name}")
  time.sleep(5)
  refreshes = get_refreshes()

# Aguarda o refresh terminar
run_info = refreshes[0]
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  run_info = w.quality_monitors.get_refresh(table_name=f"{catalog}.{db}.{inference_table_name}", refresh_id=run_info.refresh_id)
  print(f"waiting for refresh to complete {run_info.state}...")
  time.sleep(180)

assert run_info.state == MonitorRefreshInfoState.SUCCESS, "Monitor refresh failed"

# COMMAND ----------

w.quality_monitors.get(table_name=f"{catalog}.{db}.{inference_table_name}")

# COMMAND ----------

# DBTITLE 1,Delete existing monitor [OPTIONAL]
# w.quality_monitors.delete(table_name=f"{catalog}.{db}.{inference_table_name}", purge_artifacts=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspecione o dashboard
# MAGIC
# MAGIC Agora você pode inspecionar o dashboard de monitoramento gerado automaticamente. Navegue até `hiper_inferencia` no __Catalog Explorer__, vá na aba __Quality__ e clique em __View dashboard__.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/07_view_dashboard_button.png?raw=true" width="480">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Você verá o número de inferências feitas antes do primeiro refresh do monitor (a primeira "janela"), além das métricas de performance do modelo.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/07_model_inferences.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Mais abaixo, na seção __Prediction drift__, aparecem a matriz de confusão e a distribuição das predições por classe.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/07_confusion_matrix.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Ainda não há drift, pois temos apenas a primeira janela. No próximo passo, simulamos dados com drift e atualizamos o monitor.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Próximo: detectar drift e disparar o retreino
# MAGIC
# MAGIC Vamos ver como detectar drift nos dados de inferência e definir regras de violação para disparar um fluxo de (re)treino do modelo.
# MAGIC
# MAGIC Próximo:
# MAGIC * [Detectar drift e disparar o retreino]($./08_drift_detection)
