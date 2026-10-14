# Databricks notebook source
# MAGIC %md
# MAGIC #  Validação do modelo `Challenger`
# MAGIC
# MAGIC Este notebook valida o novo modelo candidato (__Challenger__) e é a primeira task do **Model Deployment Job do MLflow 3**.
# MAGIC
# MAGIC [Deployment jobs](https://docs.databricks.com/aws/en/mlflow/deployment-job) se integram ao privilégio [CREATE MODEL VERSION](https://docs.databricks.com/aws/en/data-governance/unity-catalog/manage-privileges/privileges#create-model-version) do Unity Catalog para completar a história de MLOps: um usuário pode receber o privilégio **`CREATE MODEL VERSION`** para registrar novas versões do modelo. Quando considera uma versão boa o suficiente para produção, ele a registra no modelo do Unity Catalog, o que dispara este deployment job para avaliá-la automaticamente.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-4a-v2.png?raw=true" width="1200">
# MAGIC
# MAGIC *Observação: em um setup típico de MLOps, isto roda como parte do deployment job automatizado. Aqui também pode ser executado como notebook interativo.*
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Checagens gerais de validação
# MAGIC
# MAGIC Em MLOps, há mais testes do que apenas a acurácia do modelo. Para garantir a estabilidade do sistema de ML e a conformidade com requisitos regulatórios, cada modelo adicionado ao registry passa por uma série de checagens, incluindo:
# MAGIC <br>
# MAGIC * __Documentação do modelo__
# MAGIC * __Inferência em dados de produção__
# MAGIC * __Teste Champion-Challenger para garantir que os KPIs de negócio são aceitáveis__
# MAGIC
# MAGIC Neste notebook exploramos algumas formas de fazer esses testes e de adicionar metadados ao modelo via tags indicando se passou em cada teste.
# MAGIC
# MAGIC Esta parte normalmente é específica de cada negócio e de seus requisitos de qualidade.
# MAGIC
# MAGIC Para cada teste, adicionamos tags para saber o que foi validado no modelo. Também é possível adicionar comentários.

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC databricks-feature-engineering==0.13.0a8
# MAGIC mlflow==3.3.2
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow lightgbm plotly --upgrade
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Busca as informações do modelo
# MAGIC
# MAGIC Buscamos no Unity Catalog as informações da versão do modelo a ser validada.

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# Nome completo do modelo
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
model_alias = "Challenger" # Alias a ser testado/validado

client = MlflowClient()
model_details = client.get_model_version(model_name, model_version)

run_info = client.get_run(run_id=model_details.run_id)

print(f"Validating {model_alias} model for {model_name} on model version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checagens do modelo

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checagem de descrição
# MAGIC
# MAGIC O cientista de dados descreveu o modelo submetido?

# COMMAND ----------

# Sem descrição ou com descrição curta demais: marca a tag como False
if not model_details.description:
  has_description = False
  print("Please add model description")

elif not len(model_details.description) > 20:
  has_description = False
  print("Please add detailed model description (40 char min).")

else:
  has_description = True

print(f'Model {model_name} version {model_details.version} has description: {has_description}')
client.set_model_version_tag(name=model_name, version=str(model_details.version), key="has_description", value=has_description)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Valida a predição
# MAGIC
# MAGIC Queremos testar se o modelo consegue predizer em dados de produção. Carregamos o modelo e as features mais recentes da feature store e fazemos algumas predições.
# MAGIC
# MAGIC **OBSERVAÇÃO: usamos `env_manager="local"`: treino e validação rodam no mesmo ambiente serverless, com as bibliotecas do modelo instaladas acima.**

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.types import StructType
import pandas as pd


fe = FeatureEngineeringClient()

# Carrega o modelo como Spark UDF
model_uri = f"models:/{model_name}/{model_version}"

# Predição em um DataFrame Spark
try:
  # Lê labels e chaves
  labelsDF = spark.read.table("hiper_labels").filter("split='test'").select(*primary_keys, label_col).limit(10)

  # Batch score
  features_w_preds = fe.score_batch(
    df=labelsDF,
    model_uri=model_uri,
    result_type=labelsDF.schema[label_col].dataType,
    env_manager="local")

  display(features_w_preds)
  predicts_check = True

except Exception as e:
  print(e)
  features_w_preds = spark.createDataFrame([], StructType([]))
  print("Unable to predict on features.")
  predicts_check = False

  pass

client.set_model_version_tag(name=model_name, version=str(model_version), key="predicts", value=predicts_check)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checagem de artefatos
# MAGIC O cientista de dados registrou artefatos complementares junto com o modelo?

# COMMAND ----------

import os


# Cria o diretório local
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Baixa os artefatos do tracking server
local_path = mlflow.artifacts.download_artifacts(run_id=run_info.info.run_id, dst_path=local_dir)

# Marca a versão conforme possua artefatos ou não
if not os.listdir(local_path):
  has_artifacts = False
  print("There are no artifacts associated with this model.  Please include some data/metrics visualization or data profiling.  MLflow supports HTML, .png, and more.")

else:
  has_artifacts = True
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

client.set_model_version_tag(name=model_name, version=model_version, key="has_artifacts", value=has_artifacts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Métrica de performance do modelo
# MAGIC
# MAGIC Validamos a métrica de performance comparando o `@Challenger` com o `@Champion`. Se ainda não há `@Champion`, apenas registramos a métrica do `@Challenger`.
# MAGIC
# MAGIC O modelo registrado guarda a referência ao run do experimento MLflow onde as métricas foram registradas no treino, dando rastreabilidade do modelo implantado até o treino.
# MAGIC
# MAGIC Usamos o **F1 macro** no conjunto de teste (2025), separado no momento do treino.

# COMMAND ----------

model_run_id = model_details.run_id
f1_macro = mlflow.get_run(model_run_id).data.metrics['test_f1_macro']

try:
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
except Exception:
    champion_model = None

if champion_model:
    # Compara o F1 macro do Challenger com o do Champion atual
    champion_f1 = mlflow.get_run(champion_model.run_id).data.metrics['test_f1_macro']
    print(f'Champion F1 macro: {champion_f1}. Challenger F1 macro: {f1_macro}.')
    metric_f1_passed = f1_macro >= champion_f1
else:
    print("No Champion found. Accept the model as it's the first one.")
    metric_f1_passed = True

print(f'Model {model_name} version {model_details.version} metric_f1_passed: {metric_f1_passed}')

# Tag indicando se passou na checagem de F1
client.set_model_version_tag(name=model_name, version=model_details.version, key="metric_f1_passed", value=metric_f1_passed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Métrica de negócio: valor da gestão de cuidado
# MAGIC
# MAGIC Usamos o conjunto de validação para estimar o impacto do novo modelo no negócio.
# MAGIC
# MAGIC Cada beneficiário sinalizado como `hiper_utilizacao` é acionado pela gestão de cuidado, com um custo por intervenção. Quando o beneficiário é de fato um hiperutilizador, o acompanhamento evita parte do seu custo mensal.
# MAGIC
# MAGIC O valor pode sair negativo: com `class_weight="balanced"` o modelo sinaliza mais beneficiários do que os ~5% de hiperutilizadores reais, e cada intervenção tem custo. A métrica serve para comparar Challenger e Champion; ajustar o limiar de decisão e as constantes com o time de gestão de saúde é o próximo passo.
# MAGIC
# MAGIC ***Observação: isto avalia os modelos, não é um teste A/B**. O teste A/B é feito online, dividindo o tráfego entre 2 modelos, e exige um loop de feedback para avaliar o efeito da predição (por exemplo: o acompanhamento reduziu de fato a utilização?). Veremos o teste A/B no notebook de serving em tempo real.*

# COMMAND ----------

# Base de validação: amostra do período de teste (2025)
validation_df = spark.table("hiper_labels").filter("split='test'").select(*primary_keys, label_col).limit(20000)

# Chama o modelo com o alias dado e retorna a predição
def predict_classe(validation_df, model_alias):
    return fe.score_batch(df=validation_df, model_uri=f"models:/{model_name}@{model_alias}",
                          result_type=validation_df.schema[label_col].dataType)

# COMMAND ----------

import plotly.express as px

# Constantes ilustrativas - ajuste com o time de gestão de saúde
economia_gestao = 0.20     # fração do custo mensal evitada ao acompanhar um hiperutilizador
custo_intervencao = 150.0  # R$ por beneficiário acionado pela gestão de cuidado

def valor_gestao(preds):
    """Economia com os hiperutilizadores corretamente sinalizados menos o custo das intervenções, por beneficiário-mês.
    As predições são por tipo de produção médica: um beneficiário sinalizado em vários tipos no mês é acionado uma vez."""
    benef = preds.assign(acionado=preds["prediction"] == "hiper_utilizacao", hiper=preds[label_col] == "hiper_utilizacao") \
                 .groupby(["ID_PESSOA", "dt_referencia"]) \
                 .agg(acionado=("acionado", "any"), hiper=("hiper", "any"), custo=("custo_total_mes", "sum"))
    acionados = benef[benef["acionado"]]
    return economia_gestao * acionados.loc[acionados["hiper"], "custo"].sum() - custo_intervencao * len(acionados)

# COMMAND ----------

def valor_gestao_cuidado(model_alias):
    return valor_gestao(predict_classe(validation_df, model_alias).toPandas())

if champion_model:
    data = {'Model Alias': ['Challenger', 'Champion'],
            'Valor da gestão de cuidado (R$)': [valor_gestao_cuidado("Challenger"), valor_gestao_cuidado("Champion")]}
    business_metric_passed = data['Valor da gestão de cuidado (R$)'][0] >= data['Valor da gestão de cuidado (R$)'][1]
else:
    print("No Champion found. Skipping business metrics evaluation.")
    print("Você pode voltar e rodar esta célula de novo depois de promover o Challenger a Champion no restante deste notebook.")
    data = {'Model Alias': ['Challenger', 'Champion'], 'Valor da gestão de cuidado (R$)': [0, 0]}
    business_metric_passed = True

# Tag indicando se passou na métrica de negócio
client.set_model_version_tag(name=model_name, version=model_details.version, key="business_metric_passed", value=business_metric_passed)

px.bar(data, x='Model Alias', y='Valor da gestão de cuidado (R$)', color='Model Alias',
    title='Métrica de negócio - valor da gestão de cuidado')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resultados da validação
# MAGIC
# MAGIC Pronto! Fizemos algumas checagens simples no modelo. Vamos ver os resultados.

# COMMAND ----------

results = client.get_model_version(model_name, model_version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promove a nova versão (Challenger) a Champion
# MAGIC
# MAGIC Quando os resultados são satisfatórios, promovemos o modelo a Champion definindo o alias `@Champion`. Pipelines de inferência que carregam o modelo pelo alias `@Champion` passam a usar esta versão. O alias sai automaticamente da versão Champion anterior, se houver. O modelo mantém o alias `@Challenger` até que um novo Challenger o substitua.

# COMMAND ----------

if metric_f1_passed and has_artifacts and has_description and predicts_check and business_metric_passed:
  print(f"Registering model {model_name} Version {model_version} as Champion!")
  client.set_registered_model_alias(
    name=model_name,
    alias="Champion",
    version=model_version
  )

  # Atualiza a tag de aprovação
  client.set_model_version_tag(name=model_name, version=model_details.version, key="Approval_Check", value="Approved")

else:
  client.set_model_version_tag(name=model_name, version=model_details.version, key="Approval_Check", value="Failed")

# COMMAND ----------

# MAGIC %md
# MAGIC Nesta demo promovemos o modelo mantendo-o no mesmo catálogo e schema, para que a demo seja autocontida.
# MAGIC
# MAGIC Na prática, recomenda-se manter catálogos separados para Dev, QA e Prod para dados e ativos de IA, inclusive modelos. Nesse caso, o modelo de produção seria registrado em um catálogo de produção, com o `@alias` adequado. Isso pode ser feito programaticamente e disparado quando o modelo estiver pronto para promoção.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parabéns, o modelo foi validado e promovido
# MAGIC
# MAGIC Sabemos agora que o modelo está pronto para pipelines de inferência e endpoints de serving em tempo real, pois atende aos nossos padrões de validação.
# MAGIC
# MAGIC
# MAGIC Próximo: [Inferência batch com o modelo Champion]($./05_batch_inference)
