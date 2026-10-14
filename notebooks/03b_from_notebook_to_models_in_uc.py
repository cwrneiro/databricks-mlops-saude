# Databricks notebook source
# MAGIC %md
# MAGIC # Gerenciando o ciclo de vida do modelo no Unity Catalog
# MAGIC
# MAGIC Um dos principais desafios de cientistas de dados e engenheiros de ML é a falta de um repositório central de modelos, suas versões e de meios para gerenciá-los ao longo do ciclo de vida.
# MAGIC
# MAGIC [Models in Unity Catalog](https://docs.databricks.com/en/mlflow/models-in-uc.html) resolve esse desafio e permite que o time de dados:
# MAGIC <br>
# MAGIC * **Descubra** modelos registrados, aliases atuais, runs de experimentos e o código associado a um modelo
# MAGIC * **Promova** modelos para diferentes fases do ciclo de vida usando aliases
# MAGIC * **Marque** modelos com tags para registrar metadados do processo de MLOps
# MAGIC * **Implante** diferentes versões de um modelo, permitindo que engenheiros de MLOps testem versões distintas
# MAGIC * **Teste** modelos de forma automatizada
# MAGIC * **Documente** modelos ao longo do ciclo de vida
# MAGIC * **Proteja** o acesso e as permissões para registrar, executar ou modificar modelos
# MAGIC
# MAGIC Veremos como testar e promover um novo modelo __Challenger__ como candidato a substituir o __Champion__ atual.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-3b-v2.png?raw=true" width="1200">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC mlflow==3.3.2
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Como usar Models in Unity Catalog
# MAGIC Normalmente, cientistas de dados que usam MLflow conduzem vários experimentos, cada um com runs que registram métricas e parâmetros. Durante o desenvolvimento, eles escolhem o melhor run de um experimento e registram seu modelo no Unity Catalog. Pense nisso como fazer um **commit** do modelo no Unity Catalog, assim como se faz commit de código em um controle de versão.
# MAGIC
# MAGIC O Unity Catalog oferece aliases de texto livre, como `Baseline`, `Challenger`, `Champion`, além de tags.
# MAGIC
# MAGIC Usuários com as permissões adequadas podem criar modelos, alterar aliases e tags e criar novas versões (**CREATE MODEL VERSION**), o que dispara os Deployment Jobs.

# COMMAND ----------

# MAGIC %pip install --quiet mlflow --upgrade
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encontra o melhor run e registra o modelo no Unity Catalog para validação
# MAGIC
# MAGIC Concluímos os runs de treino para encontrar um modelo candidato a __Challenger__. Vamos selecionar programaticamente o melhor modelo do experimento e registrá-lo no Unity Catalog, usando a API `search_runs` do MLflow:

# COMMAND ----------

import mlflow


print(f"Finding best run from {xp_name} and pushing new model version to {model_name}")
mlflow.set_experiment(f"{xp_path}/{xp_name}")

# COMMAND ----------

# Melhor run (para a demo; na prática a busca seria entre vários runs)
best_run_name = "hiper-hpo-best-run"

best_model = mlflow.search_runs(
  order_by=["metrics.test_f1_macro DESC"],
  max_results=1,
  filter_string=f"status = 'FINISHED' and run_name='{best_run_name}'"
)
# Opcional: carregar o experimento como DataFrame Spark com todos os runs
# df = spark.read.format("mlflow-experiment").load(experiment_id)
best_model

# COMMAND ----------

# MAGIC %md
# MAGIC Com o melhor modelo em mãos, registramos no Model Registry do Unity Catalog usando o run ID

# COMMAND ----------

# Run id do melhor modelo
run_id = best_model.iloc[0]['run_id']

# OU
# run_id = "" Para promover seu próprio run na demo: cole aqui o run_id

# COMMAND ----------

# Registra o melhor modelo no Model Registry
print(f"Registering model to {model_name}")
model_details = mlflow.register_model(f"runs:/{run_id}/model", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Neste momento o modelo ainda não tem aliases nem descrição sobre seu ciclo de vida e metadados. Vamos atualizar essas informações.
# MAGIC
# MAGIC **OBSERVAÇÃO: o registro de uma nova versão dispara um run do Model Deployment Job criado anteriormente.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adiciona uma descrição ao modelo registrado
# MAGIC
# MAGIC Feito uma vez, para o modelo como um todo.

# COMMAND ----------

from mlflow import MlflowClient


client = MlflowClient(registry_uri="databricks-uc")

# Descrição principal do modelo, normalmente feita uma vez
client.update_registered_model(
  name=model_details.name,
  description="Classifica a utilização assistencial mensal de cada beneficiário (uso_tipico, uso_elevado, hiper_utilizacao) a partir das features da tabela hiper_features. Apoia o time de gestão de saúde na priorização de ações de cuidado.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC E mais detalhes sobre a versão que acabamos de registrar

# COMMAND ----------

# Detalhes desta versão específica
best_score = best_model['metrics.test_f1_macro'].values[0]
run_name = best_model['tags.mlflow.runName'].values[0]
version_desc = f"Esta versão tem F1 macro de {round(best_score*100, 2)}% no conjunto de teste (2025). Veja o run de treino para mais detalhes."

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description=version_desc
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define a nova versão como Challenger
# MAGIC
# MAGIC A nova versão registrada recebe o alias __Challenger__. Modelos Challenger são candidatos a substituir o modelo Champion, que é o modelo em uso.
# MAGIC
# MAGIC O alias indica em que fase do ciclo de vida o modelo está.

# COMMAND ----------

# Define esta versão como Challenger via alias
client.set_registered_model_alias(
  name=model_name,
  alias="Challenger",
  version=model_details.version
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Agora inspecione as versões do modelo no Catalog Explorer: você verá a descrição da versão e o alias `Challenger`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Linhagem do modelo no Unity Catalog
# MAGIC
# MAGIC O Unity Catalog rastreia a linhagem do modelo. Selecione o modelo no Catalog Explorer, clique na última versão e em **See lineage graph** na aba **Lineage**.
# MAGIC
# MAGIC O Unity Catalog captura a lógica de featurização upstream: a feature table usada no treino (`hiper_features`) e a feature function (`custo_por_especialidade`). Também rastreia o notebook onde o modelo foi treinado. Isso dá rastreabilidade sobre a origem das features e permite análise de impacto.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/03_model_lineage.png?raw=true" width="1200">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo: validação do modelo Challenger
# MAGIC
# MAGIC Com o modelo __Challenger__ registrado, queremos validá-lo. As etapas de validação estão em um notebook e o processo foi automatizado como parte de um Lakeflow job (o "Deployment Job").
# MAGIC
# MAGIC Se o modelo passar em todos os testes, será promovido a `Champion`.
# MAGIC
# MAGIC Próximo: veja como o modelo é testado antes de ser promovido a `Champion` **via código** [no notebook de `Evaluation`]($./04a_challenger_validation)
