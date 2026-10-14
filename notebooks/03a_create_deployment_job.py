# Databricks notebook source
# MAGIC %md
# MAGIC # Cria o Model Deployment Job do MLflow 3.0
# MAGIC Vamos criar o job que avalia, aprova e implanta automaticamente cada nova versão do modelo de hiperutilização.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-3a-v2.png?raw=true" width="1200">
# MAGIC
# MAGIC <!-- [metadata={"description":"MLOps end2end workflow: Feature engineering",
# MAGIC  "authors":["quentin.ambard@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["feature store"]},
# MAGIC                  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define as tasks e cria o job programaticamente
# MAGIC Este notebook cria um Databricks Job no formato de Deployment Job do MLflow 3.0, com três tasks: Evaluation, Approval_Check e Deployment. A task Evaluation avalia o modelo em um conjunto de dados; a Approval_Check verifica se o modelo foi aprovado para implantação via tags do UC e o botão de aprovação na UI do modelo; a Deployment publica as features na Online Store e o modelo em um endpoint de serving.
# MAGIC
# MAGIC <img src="https://docs.databricks.com/aws/en/assets/images/deployment-job-create-ui-c43b64da503e3f0babcb9ff81a78610d.png" width="1200">
# MAGIC
# MAGIC Template adaptado da documentação ([AWS](https://docs.databricks.com/aws/mlflow/deployment-job#example-template-notebooks) | [Azure](https://learn.microsoft.com/azure/databricks/mlflow/deployment-job#example-template-notebooks) | [GCP](https://docs.databricks.com/gcp/mlflow/deployment-job#example-template-notebooks)).
# MAGIC
# MAGIC Passos:
# MAGIC 1. Crie um modelo no UC ou use um existente. Veja, por exemplo, os exemplos do MLflow 3 ([AWS](https://docs.databricks.com/aws/mlflow/mlflow-3-install#example-notebooks) | [Azure](https://learn.microsoft.com/azure/databricks/mlflow/mlflow-3-install#example-notebooks) | [GCP](https://docs.databricks.com/gcp/mlflow/mlflow-3-install#example-notebooks)).
# MAGIC 2. Crie as tasks independentes como notebooks Python no workspace ou repositório.
# MAGIC 3. Crie o DAG/job principal de deployment
# MAGIC 4. _OPCIONAL_ Após executar o notebook, o job criado não estará ligado a nenhum modelo do UC. **Ainda é preciso conectar o job a um modelo do UC** na UI do modelo, como indicado na documentação, ou via MLflow como na última célula ([AWS](https://docs.databricks.com/aws/mlflow/deployment-job#connect) | [Azure](https://learn.microsoft.com/azure/databricks/mlflow/deployment-job#connect) | [GCP](https://docs.databricks.com/gcp/mlflow/deployment-job#connect)), caso o modelo ainda não exista.

# COMMAND ----------

# MAGIC %md
# MAGIC Último ambiente testado:
# MAGIC ```
# MAGIC mlflow>=3.3.0
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet mlflow --upgrade
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aponta para os notebooks das tasks

# COMMAND ----------

import os


current_directory = os.getcwd()

# COMMAND ----------

# REQUIRED
model_version = "1" # The version of the already created UC Model
job_name = "Hiper-Model-Deployment-Job" # The desired name of the deployment job

# REQUIRED: Create notebooks for each task
evaluation_notebook_path = f"{current_directory}/04a_challenger_validation"
approval_notebook_path = f"{current_directory}/04b_challenger_approval" # OPTIONNAL only if Human in the Loop is required
deployment_notebook_path = f"{current_directory}/06_serve_features_and_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cria o Lakeflow Job
# MAGIC **OBSERVAÇÃO:**
# MAGIC - A flag `smoke_test` em `True` evita implantar/servir o modelo (para testes).

# COMMAND ----------

# Create job with necessary configuration to connect to model as deployment job
from databricks.sdk.service import jobs


job_settings = jobs.JobSettings(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="Evaluation",
            notebook_task=jobs.NotebookTask(notebook_path=evaluation_notebook_path),
            max_retries=0,
        ),
        jobs.Task(
            task_key="Approval_Check",
            notebook_task=jobs.NotebookTask(
                notebook_path=approval_notebook_path,
                base_parameters={"approval_tag_name": "{{task.name}}"}
            ),
            depends_on=[jobs.TaskDependency(task_key="Evaluation")],
            max_retries=0,
        ),
        jobs.Task(
            task_key="Deployment",
            notebook_task=jobs.NotebookTask(
                notebook_path=deployment_notebook_path,
                base_parameters={"smoke_test": False},
            ),
            depends_on=[jobs.TaskDependency(task_key="Approval_Check")],
            max_retries=0,
        ),
    ],
    parameters=[
        jobs.JobParameter(name="model_name", default=model_name),
        jobs.JobParameter(name="model_version", default=model_version),
    ],
    queue=jobs.QueueSettings(enabled=True),
    max_concurrent_runs=1,
)

# COMMAND ----------

from databricks.sdk import WorkspaceClient


w = WorkspaceClient()

# Search for the job by name (in case it exists)
existing_jobs = w.jobs.list(name=job_name)
job_id = None
for created_job in existing_jobs:
  if created_job.settings.name == job_name and created_job.creator_user_name == current_user:
      job_id = created_job.job_id
      break

if job_id:
  # Update existing job
  print("Updating existing...")
  w.jobs.update(job_id=job_id, new_settings=job_settings)

else:
  # Create new job
  print("Creating new...")
  created_job = w.jobs.create(**job_settings.__dict__)
  job_id = created_job.job_id

print(f"Job ID: {job_id}")

# COMMAND ----------

# DBTITLE 1,ONE-TIME Operation
print("Use the job name " + job_name + " to connect the deployment job to the UC model " + model_name + " as indicated in the UC Model UI.")
print("\nFor your reference, the job ID is: " + str(job_id))
print("\nDocumentation: \nAWS: https://docs.databricks.com/aws/mlflow/deployment-job#connect \nAzure: https://learn.microsoft.com/azure/databricks/mlflow/deployment-job#connect \nGCP: https://docs.databricks.com/gcp/mlflow/deployment-job#connect")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Liga o job ao modelo no UC
# MAGIC Liga programaticamente o deployment job ao modelo no UC.
# MAGIC Se o modelo não existir, ele é criado sem versões.

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient


client = MlflowClient(registry_uri="databricks-uc")

try:
  model_info = client.get_registered_model(model_name)
  if model_info:
    # Model exists - Link job
    if model_info.deployment_job_id == job_id:
      print("Model exists with existing job - Pass")
      pass

    else:
      print("Model exists - Updating job")
      client.update_registered_model(model_name, deployment_job_id="") # Unlink current job
      client.update_registered_model(model_name, deployment_job_id=job_id) # Link new one

except mlflow.exceptions.RestException as e:
  if "PERMISSION_DENIED" in str(e):
        print(f"Permission denied on model `{model_name}` - Deployment Job NOT UPDATED.")
        # Optionally, handle or re-raise
        pass

  else:
    # Create Empty Model placeholder and Link job
    print("Model does not exist - Creating model and linking job")
    client.create_registered_model(model_name, deployment_job_id=job_id)