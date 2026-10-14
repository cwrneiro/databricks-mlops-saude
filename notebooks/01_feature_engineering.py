# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering - Hiperutilização assistencial
# MAGIC Nosso primeiro passo é analisar os dados, aplicar a metodologia de classificação e construir as features que usaremos para treinar o modelo.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-1-v2.png?raw=true" width="1200">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC databricks-feature-engineering==0.13.0a8
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 --upgrade
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análise exploratória
# MAGIC Para entender os dados, o que precisa de limpeza, pré-processamento etc.
# MAGIC - **Use as visualizações nativas da Databricks**
# MAGIC - Ou traga sua biblioteca de visualização preferida (seaborn, plotly...)

# COMMAND ----------

# DBTITLE 1,Lê a base bruta de atendimentos
atendimentosDF = spark.read.table(bronze_table_name)
display(atendimentosDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metodologia de classificação
# MAGIC
# MAGIC A classificação considera o **volume mensal de atendimentos** de cada beneficiário em relação à **mediana de utilização de grupos com características semelhantes (estratos)**. Dessa relação sai o `score_hiper`:
# MAGIC
# MAGIC * se o estrato mensal tem pelo menos 10 beneficiários, usa a mediana mensal do estrato; senão, a mediana anual
# MAGIC * os limites das classes são os percentis 90 e 95 da distribuição histórica dos scores (2021 a 2024)
# MAGIC
# MAGIC O código abaixo é a lógica PySpark de referência da metodologia, com dois ajustes: `F.count("*")` no lugar de `F.count("")` e uso consistente das colunas convertidas (`VR_TOTAL_D`, `VR_CUSTO_D`).
# MAGIC
# MAGIC Os percentis por estrato (`p75`…`p99`) da base de modelagem são descritivos e não entram no label nem no modelo, por isso não são recalculados aqui.

# COMMAND ----------

# DBTITLE 1,Tratamento da base (metodologia de classificação)
from pyspark.sql import functions as F

df = (atendimentosDF
  .withColumn("VR_TOTAL_D", F.col("VR_TOTAL").cast("double"))
  .withColumn("VR_CUSTO_D", F.col("VR_CUSTO").cast("double"))
  .withColumn("VR_INTERCAMBIO_D", F.col("VR_INTERCAMBIO").cast("double"))
  .withColumn("DT_ATENDIMENTO_TS", F.to_timestamp("DT_ATENDIMENTO", "yyyy-MM-dd HH:mm:ss"))
  .fillna({"GRUPO": "POSTERIOR A LEI"})
  .withColumn("IDADE_ATENDIMENTO", F.floor(F.datediff(F.col("DT_ATENDIMENTO_TS"), F.to_date("DT_NASCIMENTO")) / 365.25))
  .withColumn("FAIXA_ETARIA_ATENDIMENTO",
    F.when(F.col("IDADE_ATENDIMENTO") <= 10, "00 a 10 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 20, "11 a 20 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 30, "21 a 30 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 40, "31 a 40 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 50, "41 a 50 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 60, "51 a 60 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 70, "61 a 70 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 80, "71 a 80 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 90, "81 a 90 Anos")
     .when(F.col("IDADE_ATENDIMENTO") <= 99, "91 a 99 Anos")
     .otherwise("Acima de 99 anos"))
)

# COMMAND ----------

# DBTITLE 1,Agregação mensal por beneficiário
df_mes = (df
  .withColumn("ano", F.year("DT_ATENDIMENTO_TS"))
  .withColumn("mes", F.month("DT_ATENDIMENTO_TS"))
  .groupBy("ID_PESSOA", "ID_USR_CONTR", "DS_STATUS_USUARIO", "SEXO", "FAIXA_ETARIA", "FAIXA_ETARIA_ATENDIMENTO",
           "GRUPO", "PCT_INTERNADO", "PCT_CRONICO", "CD_TIPO_PRODUCAO_MEDICA", "ano", "mes")
  .agg(F.count("*").alias("qtd_atendimentos_mes"),
       F.countDistinct("DS_ESPECIALIDADE").alias("qtd_especialidades_mes"),
       F.sum("VR_TOTAL_D").alias("custo_total_mes"),
       F.sum("VR_CUSTO_D").alias("custo_assistencial_mes"))
  .withColumnRenamed("DS_STATUS_USUARIO", "STATUS")
  .select("ID_PESSOA", "ID_USR_CONTR", "GRUPO", "PCT_INTERNADO", "PCT_CRONICO", "CD_TIPO_PRODUCAO_MEDICA", "ano", "mes",
          "FAIXA_ETARIA_ATENDIMENTO", "SEXO", "STATUS", "custo_total_mes", "custo_assistencial_mes",
          "qtd_especialidades_mes", "qtd_atendimentos_mes")
)

# COMMAND ----------

# DBTITLE 1,Score de hiperutilização (mediana do estrato)
estrato_ano = ["FAIXA_ETARIA_ATENDIMENTO", "SEXO", "STATUS", "CD_TIPO_PRODUCAO_MEDICA", "ano", "GRUPO", "PCT_INTERNADO", "PCT_CRONICO"]
estrato_mes = estrato_ano + ["mes"]

ref_mes = df_mes.groupBy(estrato_mes).agg(F.percentile_approx("qtd_atendimentos_mes", 0.5).alias("E_Y_mes"), F.count("*").alias("n_estrato_mes"))
ref_ano = df_mes.groupBy(estrato_ano).agg(F.percentile_approx("qtd_atendimentos_mes", 0.5).alias("E_Y_ano"), F.count("*").alias("n_estrato_ano"))

usa_mensal = F.col("n_estrato_mes") >= 10
df_score = (df_mes
  .join(ref_mes, estrato_mes, "left")
  .join(ref_ano, estrato_ano, "left")
  .withColumn("score_hiper", F.when(usa_mensal, F.col("qtd_atendimentos_mes") / F.col("E_Y_mes"))
                              .otherwise(F.col("qtd_atendimentos_mes") / F.col("E_Y_ano")))
  .withColumn("tipo_referencia", F.when(usa_mensal, "mediana_mensal_estrato").otherwise("mediana_anual_estrato"))
)

# COMMAND ----------

# DBTITLE 1,Limites globais p90/p95 (2021-2024) e classe de uso
limites = df_score.filter("ano BETWEEN 2021 AND 2024") \
                  .agg(F.percentile_approx("score_hiper", 0.90).alias("p90"), F.percentile_approx("score_hiper", 0.95).alias("p95")) \
                  .first()
p90_global, p95_global = limites.p90, limites.p95

df_score = (df_score
  .withColumn(label_col, F.when(F.col("score_hiper") >= p95_global, "hiper_utilizacao")
                          .when(F.col("score_hiper") >= p90_global, "uso_elevado")
                          .otherwise("uso_tipico"))
  .withColumn("dt_referencia", F.make_date("ano", "mes", F.lit(1)).cast("timestamp"))
)
print(f"p90_global = {p90_global:.3f} | p95_global = {p95_global:.3f}")
display(df_score)

# COMMAND ----------

# DBTITLE 1,Checagem da metodologia
historico = df_score.filter("ano BETWEEN 2021 AND 2024")
total = historico.count()
dist = {r[label_col]: r["n"] / total for r in historico.groupBy(label_col).agg(F.count("*").alias("n")).collect()}
print(dist)
assert set(dist) == {"uso_tipico", "uso_elevado", "hiper_utilizacao"}, f"Classe ausente: {dist}"
assert dist["uso_tipico"] >= 0.70 and min(dist.values()) >= 0.01, f"Distribuição inesperada: {dist}"
assert df_score.filter("n_estrato_mes < 10 AND tipo_referencia != 'mediana_anual_estrato'").isEmpty(), "Fallback anual não aplicado"
assert df_score.count() == df_score.select(*primary_keys).distinct().count(), "Chave primária duplicada"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Labels em uma tabela separada para evitar vazamento
# MAGIC * Na prática, os labels (ground truth) ficam em uma tabela separada das features
# MAGIC * `score_hiper` e `qtd_atendimentos_mes` definem o label, por isso ficam junto dele e **não** entram na feature table
# MAGIC * Split temporal: treino = 2021–2024 (mesmo período dos limites), teste = 2025

# COMMAND ----------

# DBTITLE 1,Tabela de labels separada (evita vazamento) com split temporal
df_score.select(*primary_keys, label_col, "score_hiper", "qtd_atendimentos_mes") \
        .withColumn("split", F.when(F.col("dt_referencia") < "2025-01-01", "train").otherwise("test")) \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable("hiper_labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constraint de chave primária na tabela de labels _(OPCIONAL)_

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE hiper_labels DROP CONSTRAINT IF EXISTS hiper_labels_pk;
# MAGIC ALTER TABLE hiper_labels ALTER COLUMN ID_PESSOA SET NOT NULL;
# MAGIC ALTER TABLE hiper_labels ALTER COLUMN ID_USR_CONTR SET NOT NULL;
# MAGIC ALTER TABLE hiper_labels ALTER COLUMN CD_TIPO_PRODUCAO_MEDICA SET NOT NULL;
# MAGIC ALTER TABLE hiper_labels ALTER COLUMN dt_referencia SET NOT NULL;
# MAGIC ALTER TABLE hiper_labels ADD CONSTRAINT hiper_labels_pk PRIMARY KEY(ID_PESSOA, ID_USR_CONTR, CD_TIPO_PRODUCAO_MEDICA, dt_referencia);

# COMMAND ----------

# DBTITLE 1,Features (sem as colunas que definem o label)
featuresDF = df_score.select(*primary_keys, "FAIXA_ETARIA_ATENDIMENTO", "SEXO", "STATUS", "GRUPO", "PCT_INTERNADO", "PCT_CRONICO",
                             "mes", "custo_total_mes", "custo_assistencial_mes", "qtd_especialidades_mes")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Grava a feature table no Unity Catalog
# MAGIC
# MAGIC Com o Unity Catalog, qualquer tabela Delta com chave primária e colunas NOT NULL pode ser uma feature table offline.
# MAGIC
# MAGIC Isso traz governança, descoberta e reuso das features em toda a organização, aumentando a eficiência dos times.
# MAGIC
# MAGIC A linhagem do Unity Catalog dá rastreabilidade: sabemos qual modelo depende de quais feature tables.
# MAGIC
# MAGIC Feature tables de séries temporais têm uma chave primária adicional na coluna de tempo (aqui, `dt_referencia`, o mês de referência).
# MAGIC
# MAGIC Abaixo criamos a feature table com a API `FeatureEngineeringClient`. Também é possível criá-la via SQL:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```
# MAGIC CREATE TABLE {catalog}.{db}.{feature_table_name} (
# MAGIC   {primary_key} int NOT NULL,
# MAGIC   {timestamp_col} timestamp NOT NULL,
# MAGIC   feat1 long,
# MAGIC   feat2 varchar(100),
# MAGIC   CONSTRAINT features_pk PRIMARY KEY ({primary_key}, {timestamp_col} TIMESERIES)
# MAGIC );
# MAGIC ```
# MAGIC

# COMMAND ----------

# DBTITLE 1,Remove a feature table se já existir
# MAGIC %sql
# MAGIC -- Recriamos a feature table do zero para a demo ser repetível
# MAGIC DROP TABLE IF EXISTS hiper_features;

# COMMAND ----------

# DBTITLE 1,Import Feature Store Client
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Cria a feature table no UC
fe.create_table(
  name="hiper_features",
  primary_keys=primary_keys,
  schema=featuresDF.schema,
  timeseries_columns="dt_referencia",
  description=f"Features mensais por beneficiário e tipo de produção médica, derivadas de {catalog}.{db}.{bronze_table_name}. Não inclui as colunas usadas no cálculo do score de hiperutilização."
)

# COMMAND ----------

# DBTITLE 1,Grava as features
fe.write_table(name=f"{catalog}.{db}.hiper_features", df=featuresDF, mode="merge")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features calculadas sob demanda (on-demand)
# MAGIC
# MAGIC Algumas features são calculadas no momento da inferência. Funções no Unity Catalog podem ser usadas tanto no treino/inferência batch quanto no serving online, com governança sobre quem pode usá-las.
# MAGIC
# MAGIC A função `custo_por_especialidade` calcula o custo médio por especialidade atendida no mês: um beneficiário com custo alto concentrado em poucas especialidades tem um perfil diferente de quem distribui o custo entre muitas.
# MAGIC
# MAGIC Veja a documentação ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/on-demand-features.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/on-demand-features)).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION custo_por_especialidade(custo_total_mes_in DOUBLE, qtd_especialidades_mes_in BIGINT)
# MAGIC RETURNS DOUBLE
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "[Feature Function] Custo médio por especialidade atendida no mês"
# MAGIC AS $$
# MAGIC if qtd_especialidades_mes_in and qtd_especialidades_mes_in > 0:
# MAGIC   return custo_total_mes_in / qtd_especialidades_mes_in
# MAGIC return 0.0
# MAGIC $$

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE FUNCTION custo_por_especialidade;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Acelere a criação de modelos com o Data Science Agent do Databricks Assistant
# MAGIC O Data Science Agent transforma o Databricks Assistant de um copiloto em um parceiro autônomo para ciência de dados e analytics. Integrado aos Notebooks e ao SQL Editor, o [Data Science Agent](https://www.databricks.com/blog/introducing-databricks-assistant-data-science-agent) une inteligência, adaptabilidade e execução em uma única experiência.
# MAGIC <img src="https://www.databricks.com/sites/default/files/2025-09/AgentModeOG1Border.png?v=1756901406" width="500px" style="float: right"/>
# MAGIC - Transforma o Databricks Assistant em um parceiro autônomo para tarefas de ciência de dados e analytics em Notebooks e no SQL Editor.
# MAGIC
# MAGIC - Explora dados, gera e executa código e corrige erros a partir de um único prompt, reduzindo horas de trabalho para minutos.
# MAGIC
# MAGIC - Feito para tarefas comuns de ciência de dados e baseado no Unity Catalog, com acesso governado aos dados.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vamos treinar um modelo com otimização de hiperparâmetros
# MAGIC
# MAGIC Nesta demo usamos a integração nativa do MLflow com o `optuna`
# MAGIC
# MAGIC Próximo: [Treinar um modelo com HPO]($./02_model_training_hpo_optuna)
