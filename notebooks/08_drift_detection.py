# Databricks notebook source
# MAGIC %md
# MAGIC # Detecção de drift
# MAGIC
# MAGIC Nesta etapa definimos regras de detecção de drift que rodam periodicamente sobre os dados de inferência.
# MAGIC
# MAGIC **Detecção de drift** é o processo de identificar mudanças nas propriedades estatísticas dos dados de entrada, que podem degradar a performance do modelo ao longo do tempo. É essencial para manter a acurácia e a confiabilidade dos modelos em ambientes dinâmicos, permitindo intervenções no momento certo, como o retreino.
# MAGIC
# MAGIC Para simular drift usamos a biblioteca [_dbldatagen_](https://github.com/databrickslabs/dbldatagen), um projeto do Databricks Labs para gerar dados sintéticos com Spark.
# MAGIC
# MAGIC Simulamos um cenário em que a carteira passa a ter **mais beneficiários internados e crônicos** e a distribuição real das classes se desloca para `hiper_utilizacao`.
# MAGIC **Label drift** ocorre quando a distribuição do ground truth muda ao longo do tempo, seja por mudança real no perfil de utilização, seja por mudança no critério de classificação ou erros de rotulagem. _Criamos drift de label e de predição_.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-8-v2.png?raw=true" width="1200">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC DBR/MLR16.4LTS (Recomendado)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Install needed package
# MAGIC %pip install --quiet databricks-sdk mlflow-skinny --upgrade dbldatagen
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Define drift metrics
dbutils.widgets.dropdown("perf_metric", "f1_score.macro", ["accuracy_score", "precision.weighted", "recall.weighted", "f1_score.macro"])
dbutils.widgets.dropdown("drift_metric", "js_distance", ["chi_squared_test.statistic", "chi_squared_test.pvalue", "tv_distance", "l_infinity_distance", "js_distance"])
dbutils.widgets.text("model_id", "*", "Model Id")

# COMMAND ----------

# MAGIC %md
# MAGIC Roda o setup e gera os dados sintéticos com drift

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true $gen_synthetic_data=true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Atualiza o monitor
# MAGIC
# MAGIC O passo anterior gravou os dados sintéticos na tabela de inferência. Atualizamos o monitor para recalcular as métricas.
# MAGIC
# MAGIC **Obs.:** o refresh só é necessário se a tabela monitorada mudou

# COMMAND ----------

# DBTITLE 1,Refresh the monitor
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInfoStatus, MonitorRefreshInfoState


w = WorkspaceClient()
refresh_info = w.quality_monitors.run_refresh(table_name=f"{catalog}.{db}.{inference_table_name}")

while refresh_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  refresh_info = w.quality_monitors.get_refresh(table_name=f"{catalog}.{db}.{inference_table_name}", refresh_id=refresh_info.refresh_id)
  time.sleep(180)

# COMMAND ----------

# DBTITLE 1,Programmatically retrieve profile and drift table names from monitor info
monitor_info = w.quality_monitors.get(table_name=f"{catalog}.{db}.{inference_table_name}")
drift_table_name = monitor_info.drift_metrics_table_name
profile_table_name = monitor_info.profile_metrics_table_name

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Inspecione o dashboard
# MAGIC
# MAGIC Após o refresh, o dashboard de monitoramento mostra as métricas mais recentes do modelo. Avaliado contra os labels mais recentes, o modelo tem F1 macro e recall bem piores.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/08_model_kpis.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Vamos mostrar como buscar as métricas de drift programaticamente e disparar o retreino.
# MAGIC
# MAGIC Vale notar: pela matriz de confusão do dashboard, metade dos casos recentes passou a ser `hiper_utilizacao`. Antes de retreinar, vale questionar se houve uma mudança real no perfil da carteira (por exemplo, entrada de um contrato com muitos crônicos) ou um problema de qualidade de dados/rotulagem upstream. Problemas de rotulagem não exigem retreino.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/08_confusion_matrix.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Busca as métricas de drift
# MAGIC
# MAGIC Consultamos a tabela de métricas de drift do Lakehouse Monitoring para a tabela de inferência monitorada.
# MAGIC Testamos se as métricas ultrapassaram limites definidos pelo negócio:
# MAGIC 1. Drift de predição (distância de Jensen–Shannon) > 0.19
# MAGIC 2. Drift de label (distância de Jensen–Shannon) > 0.19
# MAGIC 3. Custo médio de hiperutilizador não detectado > R$ 50
# MAGIC 4. Performance (F1 macro) < 0.5

# COMMAND ----------

metric = dbutils.widgets.get("perf_metric")
drift = dbutils.widgets.get("drift_metric")
model_id = dbutils.widgets.get("model_id")

# COMMAND ----------

# MAGIC %md
# MAGIC Monta um dataframe para detectar degradação de performance a partir da tabela de métricas de perfil gerada pelo Lakehouse Monitoring

# COMMAND ----------

# DBTITLE 1,dataframe for performance metrics
performance_metrics_df = spark.sql(f"""
SELECT
  window.start as time,
  {metric} AS performance_metric,
  custo_hiper_nao_detectado,
  Model_Version AS `Model Id`
FROM {profile_table_name}
WHERE
  log_type = "INPUT"
  AND column_name = ":table"
  AND slice_key is null
  AND slice_value is null
  AND Model_Version = '{model_id}'
ORDER BY
  window.start
"""
)
display(performance_metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Monta um dataframe para detectar drift a partir da tabela de métricas de drift gerada pelo Lakehouse Monitoring.

# COMMAND ----------

drift_metrics_df = spark.sql(f"""
  SELECT
  window.start AS time,
  column_name,
  {drift} AS drift_metric,
  Model_Version AS `Model Id`
FROM {drift_table_name}
WHERE
  column_name IN ('prediction', 'classe_uso')
  AND slice_key is null
  AND slice_value is null
  AND Model_Version = '{model_id}'
  AND drift_type = "CONSECUTIVE"
ORDER BY
  window.start
"""
)
display(drift_metrics_df )

# COMMAND ----------

# DBTITLE 1,Unstack dataframe
from pyspark.sql.functions import first


# Se não há drift de label ou predição, pula
if not drift_metrics_df.isEmpty():
    unstacked_drift_metrics_df = (
        drift_metrics_df.groupBy("time", "`Model Id`")
        .pivot("column_name")
        .agg(first("drift_metric"))
        .orderBy("time")
    )
    display(unstacked_drift_metrics_df)

# COMMAND ----------

# DBTITLE 1,Join all metrics together
all_metrics_df = performance_metrics_df
if not drift_metrics_df.isEmpty():
    all_metrics_df = performance_metrics_df.join(
        unstacked_drift_metrics_df, on=["time", "Model Id"], how="inner"
    )

display(all_metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conta as violações e salva como task value
# MAGIC
# MAGIC Limites para caracterizar degradação:
# MAGIC - Métrica de performance (F1 macro) < 0.5
# MAGIC - Custo médio de hiperutilizador não detectado (métrica customizada ligada ao negócio) > R$ 50

# COMMAND ----------

# DBTITLE 1,count nr violations
from pyspark.sql.functions import col, abs


performance_violation_count = all_metrics_df.where(
    (col("performance_metric") < 0.5) & (abs(col("custo_hiper_nao_detectado")) > 50)
).count()

drift_violation_count = 0
if not drift_metrics_df.isEmpty():
    drift_violation_count = all_metrics_df.where(
        (col("classe_uso") > 0.19) & (col("prediction") > 0.19)
    ).count()

all_violations_count = drift_violation_count + performance_violation_count

# COMMAND ----------

print(f"Total number of joint violations: {all_violations_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo: disparar o retreino do modelo
# MAGIC
# MAGIC Com o número de violações, automatizamos ações como:
# MAGIC - Retreinar o modelo
# MAGIC - Enviar um alerta aos responsáveis via Slack ou e-mail
# MAGIC
# MAGIC Uma forma de fazer isso na Databricks é adicionar uma lógica condicional ao job com [a task If/else condition](https://docs.databricks.com/en/jobs/conditional-tasks.html#add-branching-logic-to-your-job-with-the-ifelse-condition-task).
# MAGIC
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/product/mlops/advanced/08_view_retraining_workflow.png?raw=true" width="1200">
# MAGIC
# MAGIC Para isso, salvamos o número de violações em um [task value](https://docs.databricks.com/en/jobs/share-task-context.html), consumido pela condição If/else.
# MAGIC
# MAGIC No nosso workflow, disparamos o treino do modelo como uma task do tipo job run, apontando para o job de treino.

# COMMAND ----------

# DBTITLE 1,Exit notebook by setting a task value
dbutils.jobs.taskValues.set(key = 'all_violations_count', value = all_violations_count)
