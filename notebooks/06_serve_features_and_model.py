# Databricks notebook source
# MAGIC %md
# MAGIC # Inferência em tempo real - Hiperutilização
# MAGIC
# MAGIC Já vimos como obter predições em batch. Agora vamos publicar as features e o modelo para predições em tempo real via API REST. Aplicações de atendimento e de gestão de cuidado podem consultar a classe de utilização do beneficiário no momento em que interagem com ele.
# MAGIC
# MAGIC Como as predições são feitas durante a interação, precisam ser retornadas com baixa latência.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-6-v2.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Para servir as features e o modelo, vamos:
# MAGIC
# MAGIC - Disponibilizar as features para consulta de baixa latência pelo modelo via Online Feature Store (Lakebase)
# MAGIC - Implantar o modelo registrado no Unity Catalog em um endpoint de Model Serving
# MAGIC
# MAGIC Essas tarefas podem ser feitas pela UI ou automatizadas com o Databricks Python SDK ([AWS](https://docs.databricks.com/en/dev-tools/sdk-python.html#)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/sdk-python)|[GCP](https://docs.gcp.databricks.com/dev-tools/sdk-python.html)), disponível no Databricks Runtime 13.3LTS+

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC databricks-feature-engineering==0.13.0a8
# MAGIC mlflow==3.3.2
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

dbutils.widgets.text("model_name", model_name, "Model Name") # Preenchido pelos parâmetros do Deployment Job
try:
  versao_padrao = client.get_model_version_by_alias(model_name, "Champion").version # Padrão: versão @Champion
except Exception:
  versao_padrao = "1" # alias ainda não existe (primeira execução)
dbutils.widgets.text("model_version", versao_padrao, "Model Version") # Preenchido pelos parâmetros do Deployment Job
dbutils.widgets.text("online_store_name", online_store_name, "Online Store Name")
dbutils.widgets.dropdown("drop_online_store", "False", ["True", "False"], "Reset Online Table(s)")
dbutils.widgets.dropdown("smoke_test", "False", ["True", "False"], "Smoke Test Flag")

# COMMAND ----------

# MAGIC %md
# MAGIC # Servindo features com as Online Tables do Lakebase
# MAGIC
# MAGIC Para responder consultas de predição com baixa latência, publicamos as features em online tables e as servimos em tempo real ao modelo.
# MAGIC
# MAGIC No feature engineering criamos uma tabela Delta como feature table offline. Lembre que qualquer tabela Delta com chave primária pode ser uma feature table na Databricks.

# COMMAND ----------

is_smoke_test = dbutils.widgets.get("smoke_test").lower() == "true"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Cria a Online Feature Store
# MAGIC
# MAGIC A online table pode ser criada pela UI do Catalog Explorer ou pela API. Os passos estão abaixo. Mais detalhes na documentação ([AWS](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-store)). Permissões necessárias ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#user-permissions)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#user-permissions)).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Habilita o Change Data Feed na feature table
# MAGIC
# MAGIC Uma online table é uma cópia somente leitura de uma tabela Delta, armazenada em formato orientado a linhas e otimizada para acesso online.
# MAGIC
# MAGIC A Databricks atualiza as online tables de forma eficiente sempre que a feature table muda, graças ao Change Data Feed do Delta Lake. Vamos habilitá-lo na feature table `hiper_features`.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE hiper_features SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### OPÇÃO 1: pela UI do Catalog Explorer
# MAGIC No Catalog Explorer, navegue até a tabela de origem a ser sincronizada. No menu **Create**, selecione **Synced table**.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/06_create_online_table.gif?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Preencha os campos:
# MAGIC
# MAGIC * **Name**: `hiper_features_online`
# MAGIC   * Nome da online table no Unity Catalog.
# MAGIC * **Primary Key**: `ID_PESSOA`, `ID_USR_CONTR`, `CD_TIPO_PRODUCAO_MEDICA`
# MAGIC   * Colunas da tabela de origem usadas como chave primária na online table.
# MAGIC * **Timeseries Key**: `dt_referencia`
# MAGIC   * Coluna da tabela de origem usada como chave de série temporal.
# MAGIC
# MAGIC Mantenha o **Sync mode** como **Snapshot**: a sincronização atualiza o pipeline a partir da feature table de origem. Veja a documentação ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-tables)).
# MAGIC
# MAGIC Clique em Confirm e aguarde a sincronização terminar na página da online table.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/06_online_table.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC *A online table é criada no catálogo, schema e nome especificados, e aparece com o ícone de online table no Catalog Explorer.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### OPÇÃO 2: pelo Databricks Feature Engineering SDK
# MAGIC
# MAGIC A alternativa é o [Databricks Feature Engineering SDK](https://api-docs.databricks.com/python/feature-engineering/latest/index.html). Primeiro definimos a online store, depois publicamos a tabela.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient


# Inicializa o client
fe = FeatureEngineeringClient()

# Nome da Online Store (altere para evitar conflito se quiser recriar)
online_store_name = dbutils.widgets.get("online_store_name")
print(f'Using online store: {online_store_name}')

endpoint_name = ("hiperutilizacao_" + current_user_az)[:50]
print(f'Using endoint name: {endpoint_name}')

# Verifica se já existe
online_store = fe.get_online_store(name=online_store_name)

# COMMAND ----------

import time


if online_store:
  print(f"Online store exists:")
  print(f"Store: {online_store.name}, State: {online_store.state}, Capacity: {online_store.capacity}")

  # Para aumentar a capacidade da online store
  # updated_store = fe.update_online_store(
  #     name=online_store_name,
  #     capacity="CU_2"
  # )

  if dbutils.widgets.get("drop_online_store") == "True" and not is_smoke_test:
    # Remove e aguarda
    fe.delete_online_store(name=online_store_name)
    time.sleep(60)

    print(f"Dropping/Recreating it.")
    online_store = fe.create_online_store(
      name=online_store_name,
      capacity="CU_1"
    )

elif not online_store and not is_smoke_test:
  # Cria uma online store com a capacidade especificada
  print(f"Creating Online store: {online_store_name}")
  online_store = fe.create_online_store(
      name=online_store_name,
      capacity="CU_1"  # Opções: "CU_1", "CU_2", "CU_4", "CU_8"
  )

# COMMAND ----------

if not is_smoke_test:
    print(f"Publishing feature table to online store...")
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            publish_state = fe.publish_table(
                online_store=online_store,
                source_table_name=f"{catalog}.{db}.hiper_features",
                online_table_name=f"{catalog}.{db}.hiper_features_online",
                # streaming=True
            )
            break
        except Exception as e:
            if "feature sync is currently in progress" in str(e):
                print("Feature sync in progress, retrying...")
                retry_count += 1
                time.sleep(10)  # Aguarda 10 segundos antes de tentar de novo
            else:
                raise e
    else:
        print("Failed to publish after multiple retries.")

    # Aguarda a primeira sincronização: um endpoint criado antes da online table ficar online falha
    while not (estado := w.database.get_synced_database_table(name=f"{catalog}.{db}.hiper_features_online").data_synchronization_status.detailed_state.value).startswith("SYNCED_TABLE_ONLINE"):
        assert "FAILED" not in estado, f"Sincronização da online table falhou: {estado}"
        print(f"Online table: {estado}, aguardando...")
        time.sleep(30)
    print(f"Online table: {estado}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Revisando a feature calculada sob demanda
# MAGIC
# MAGIC A online table está publicada e as features estão disponíveis para o modelo com baixa latência.
# MAGIC
# MAGIC Lembre que também definimos a função `custo_por_especialidade`, calculada sob demanda. Vamos revisá-la.
# MAGIC
# MAGIC Ela foi especificada como feature function ao criar o conjunto de treino, e essa informação foi registrada com o modelo no MLflow. Assim, no serving, o modelo sabe tanto buscar as features na online table quanto calcular `custo_por_especialidade` sob demanda.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE FUNCTION EXTENDED custo_por_especialidade;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Implantando o modelo para inferência em tempo real
# MAGIC
# MAGIC Para disponibilizar o modelo via API REST, vamos implantá-lo em um endpoint de Model Serving.
# MAGIC
# MAGIC As aplicações de atendimento e gestão de cuidado podem apontar para esse endpoint, com muitas requisições concorrentes. O Model Serving cuida de toda a infraestrutura, implantação e escala: basta implantar o modelo!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promovendo o modelo para serving em tempo real
# MAGIC
# MAGIC Vimos como usar os aliases `@Champion` e `@Challenger` para promover modelos no pipeline batch.
# MAGIC
# MAGIC Como a inferência em tempo real é outra forma de consumo do modelo, precisamos de um fluxo que garanta uma implantação segura. Há várias estratégias para implantar uma nova versão; vamos ver o teste A/B como exemplo. Outras incluem canary, shadow testing etc.
# MAGIC
# MAGIC No teste A/B há um modelo 'em produção'. Quando temos uma nova versão, desviamos uma porcentagem do tráfego para ela, comparamos os resultados com o modelo em produção e monitoramos. Se os resultados forem aceitáveis, todo o tráfego passa para a nova versão, que se torna a nova versão em produção.
# MAGIC
# MAGIC Introduzimos um terceiro alias, `@Production`, para esse fluxo:
# MAGIC
# MAGIC - `@Production`: modelo em produção
# MAGIC - `@Champion`: nova versão promovida após o teste Champion-Challenger
# MAGIC - `@Challenger`: modelo que desafia o Champion. Nunca vai para serving em tempo real sem antes ser promovido a Champion.
# MAGIC
# MAGIC O Model Serving permite servir várias versões em um endpoint e dividir o tráfego entre elas. Por exemplo, para testar um novo Champion, 20% do tráfego pode ir para o Champion e 80% para o modelo em produção.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/06_traffic_percent.png?raw=true" width="450">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Como esta é a primeira implantação do endpoint, a versão Champion recebe 100% do tráfego.
# MAGIC

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Cria o endpoint de Model Serving
# MAGIC
# MAGIC Com o modelo registrado no Unity Catalog, criar o endpoint é simples, pela UI ou pela API.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### OPÇÃO 1: pela UI
# MAGIC
# MAGIC Vá em **Serving** na seção **Machine Learning** e clique em **Create serving endpoint**.
# MAGIC
# MAGIC Ou abra a página do modelo e clique em "Serving": o modelo sobe atrás de um endpoint REST e você já pode enviar requisições HTTP!
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/06_create_serving_endpoint.gif?raw=true" width="854">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Preencha os campos:
# MAGIC
# MAGIC * **Name**: `hiperutilizacao_<usuario>`
# MAGIC   * Nome do endpoint
# MAGIC * **Entity**: digite `modelo_hiperutilizacao` e escolha o modelo registrado nos notebooks anteriores.
# MAGIC   * Modelo registrado no Unity Catalog a ser servido.
# MAGIC * **Compute type**: mantenha **CPU**
# MAGIC * **Compute scale out**: escolha **Small**
# MAGIC   * Define quantas requisições concorrentes o endpoint suporta.
# MAGIC * **Scale to zero**: mantenha marcado
# MAGIC   * Permite que o endpoint escale para zero quando não há requisições
# MAGIC
# MAGIC Clique em **Create** e aguarde o provisionamento, que pode levar bastante tempo.
# MAGIC
# MAGIC Quando o endpoint estiver pronto, o status será **Ready**.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/06_served_model.png?raw=true" width="854">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Veja a documentação sobre criação e gestão de endpoints. ([AWS](https://docs.databricks.com/machine-learning/model-inference/serverless/create-manage-serverless-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-inference/serverless/create-manage-serverless-endpoints))

# COMMAND ----------

# MAGIC %md
# MAGIC ### OPÇÃO 2: via API
# MAGIC
# MAGIC O mesmo pode ser feito programaticamente. O código abaixo cria o endpoint de serving.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cria/atualiza o endpoint

# COMMAND ----------

# Nome do modelo sem o namespace do UC
served_model_name =  model_name.split('.')[-1]

# COMMAND ----------

from databricks.sdk.service.serving import AiGatewayConfig, EndpointCoreConfigInput


endpoint_config_dict = {
    "served_entities": [
        # Modelos a serem servidos
        {
            "entity_name": model_name,
            "entity_version": model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
        },
        # {
        #     "entity_name": model_name,
        #     "entity_version": model_version_b,
        #     "scale_to_zero_enabled": True,
        #     "workload_size": "Small",
        # },
    ],
    "traffic_config": {
        "routes": [
            # Versões servidas; traffic_percentage deve somar 100
            # Convenção de nome do served_model_name: <registered_model_name>-<model_version>
            {"served_model_name": f"{served_model_name}-{model_version}", "traffic_percentage": 100},
            # {"served_model_name": f"{served_model_name}-{model_version_b}", "traffic_percentage": 10},
        ]
    },
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# Inference table do AI Gateway: registra requisições e respostas do endpoint em uma tabela Delta
ai_gateway_config = AiGatewayConfig.from_dict({
    "inference_table_config": {
        "catalog_name": catalog,
        "schema_name": db,
        "table_name_prefix": "hiper_served",
        "enabled": True,
    }
})

# COMMAND ----------

from databricks.sdk.service.serving import EndpointTag
from databricks.sdk.errors import ResourceDoesNotExist


if not is_smoke_test:
  try:
    # Atualiza com a nova versão do modelo
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=endpoint_config.served_entities,
        traffic_config=endpoint_config.traffic_config
      )

    print(f"Updating endpoint {endpoint_name} with models {model_name} version {model_version}")

  except ResourceDoesNotExist:
    w.serving_endpoints.create(
      name=endpoint_name,
      config=endpoint_config,
      ai_gateway=ai_gateway_config,
      tags=[EndpointTag.from_dict({"key": "demo", "value": "hiperutilizacao"})]
    )

    print(f"Creating endpoint {endpoint_name} with models {model_name} version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Aguarda o endpoint ficar pronto
# MAGIC
# MAGIC Deixe a célula abaixo rodando: o endpoint pode levar um bom tempo para ficar pronto.

# COMMAND ----------

from datetime import timedelta


if not is_smoke_test:
  # Aguarda o endpoint ficar pronto ou terminar de atualizar
  endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name, timeout=timedelta(minutes=30))

  assert endpoint.state.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY" , "Endpoint not ready or failed"

# COMMAND ----------

# MAGIC %md
# MAGIC # Envia requisições via REST
# MAGIC
# MAGIC Você pode testar o endpoint pela UI, colando este JSON:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {"ID_PESSOA": 100001, "ID_USR_CONTR": 9123456, "CD_TIPO_PRODUCAO_MEDICA": "CONSUL", "dt_referencia": "2025-06-01"}
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC (use chaves existentes na tabela `hiper_labels`, como as geradas na célula abaixo)
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/06_online_scoring.gif?raw=true" width="950">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Rode as próximas células para chamar o endpoint programaticamente.
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Exemplo de payload: chaves de beneficiários do período de teste
import json

amostra = spark.table("hiper_labels").filter("split='test'").select(*primary_keys).limit(2).toPandas()
amostra["dt_referencia"] = amostra["dt_referencia"].dt.strftime("%Y-%m-%d")
dataframe_records = json.loads(amostra.to_json(orient="records"))  # tipos nativos (numpy int64 não serializa)
dataframe_records

# COMMAND ----------

# DBTITLE 1,Query endpoint
print("Classe de uso:")
response = w.serving_endpoints.query(name=f"{endpoint_name}", dataframe_records=dataframe_records)
print(response.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parabéns! As features e o modelo estão disponíveis em tempo real.
# MAGIC
# MAGIC Agora podemos classificar beneficiários em batch e em tempo real. A seguir, veremos como monitorar a performance do modelo.
# MAGIC
# MAGIC Com as tabelas de inferência, criamos um monitor para acompanhar o comportamento do sistema de ML ao longo do tempo (drift de features, de predição e de label, métricas do modelo etc.)
# MAGIC
# MAGIC Próximos passos:
# MAGIC * [Criar o monitor de performance do modelo]($./07_model_monitoring)
# MAGIC * [Detectar drift e disparar o retreino]($./08_drift_detection)
