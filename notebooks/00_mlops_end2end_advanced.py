# Databricks notebook source
# MAGIC %md
# MAGIC # Demo MLOps end-to-end com MLflow e Models in Unity Catalog
# MAGIC
# MAGIC ## Desafios para levar um projeto de ML para produção
# MAGIC
# MAGIC Levar um projeto de ML de um notebook isolado para um pipeline de produção é complexo e exige múltiplas competências.
# MAGIC
# MAGIC Ter um modelo funcionando em um notebook não é suficiente. Precisamos cobrir o ciclo de vida completo do projeto de ML e resolver os seguintes desafios:
# MAGIC
# MAGIC * Atualizar os dados ao longo do tempo (pipeline de ingestão de produção)
# MAGIC * Salvar, compartilhar e reutilizar features de ML na organização
# MAGIC * Garantir que uma nova versão do modelo respeite os padrões de qualidade e não quebre o pipeline
# MAGIC * Governança de modelos: o que está implantado, como foi treinado, por quem e com quais dados?
# MAGIC * Como monitorar e retreinar o modelo...
# MAGIC
# MAGIC Além disso, esses projetos normalmente envolvem vários times, gerando atrito e silos:
# MAGIC
# MAGIC * Engenheiros de dados, responsáveis por ingerir, preparar e expor os dados
# MAGIC * Cientistas de dados, especialistas em análise de dados e construção de modelos
# MAGIC * Engenheiros de ML, que montam a infraestrutura e os pipelines de ML (similar a DevOps)
# MAGIC
# MAGIC Isso tem impacto real no negócio: atrasa projetos, impede que cheguem à produção e que gerem retorno.
# MAGIC
# MAGIC ## O que é MLOps?
# MAGIC
# MAGIC MLOps é um conjunto de padrões, ferramentas, processos e metodologia que busca otimizar tempo, eficiência e qualidade, garantindo governança em projetos de ML.
# MAGIC
# MAGIC MLOps orquestra o ciclo de vida do projeto entre os times para implementar esses pipelines de ML de forma fluida.
# MAGIC
# MAGIC A Databricks está posicionada de forma única para resolver esse desafio com o padrão Lakehouse: além de reunir engenheiros de dados, cientistas de dados e engenheiros de ML em uma única plataforma, oferece ferramentas para orquestrar projetos de ML e acelerar a ida para produção.
# MAGIC
# MAGIC ## Fluxo do processo de MLOps
# MAGIC
# MAGIC Nesta demo percorremos um fluxo de MLOps completo, de ponta a ponta. O resultado final é um modelo que alimenta as análises do time de gestão de saúde e um endpoint REST que faz predições em tempo real.
# MAGIC
# MAGIC Este fluxo envolve:
# MAGIC
# MAGIC * preparar features e funções, calculando-as e persistindo-as em uma feature store offline
# MAGIC * treinar um modelo para implantação
# MAGIC * registrar o modelo para que seu uso seja governado
# MAGIC * validar o modelo em uma análise champion-challenger
# MAGIC * usar o modelo treinado em um job de inferência em batch
# MAGIC * publicar features para consulta em tempo real
# MAGIC * implantar o modelo em um endpoint de serving em tempo real
# MAGIC * monitorar drift de dados e do modelo
# MAGIC * detectar drift e disparar o retreino do modelo
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-0-v2.png?raw=true" width="1200">
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet mlflow --upgrade
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $adv_mlops=true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hiperutilização assistencial
# MAGIC
# MAGIC Vamos implementar um modelo de classificação de utilização assistencial para ilustrar os conceitos avançados de MLOps.
# MAGIC
# MAGIC O time de gestão de saúde de uma operadora de saúde precisa identificar, a cada mês, em qual classe de utilização cada beneficiário se encontra:
# MAGIC
# MAGIC | Classe | Definição |
# MAGIC |---|---|
# MAGIC | `uso_tipico` | Utilização dentro dos padrões esperados |
# MAGIC | `uso_elevado` | Utilização acima do padrão habitual |
# MAGIC | `hiper_utilizacao` | Utilização significativamente superior ao padrão de referência |
# MAGIC
# MAGIC O label vem da própria metodologia do time de inteligência: o **score de hiperutilização** é o volume mensal de atendimentos do beneficiário dividido pela mediana de utilização de grupos com características semelhantes (estratos). Os limites das classes são os percentis 90 e 95 da distribuição histórica dos scores (2021 a 2024).
# MAGIC
# MAGIC O modelo aprende a reconhecer essas classes a partir do perfil do beneficiário e do seu custo no mês, permitindo que a gestão de cuidado priorize ações. Ele também será usado em tempo real, por meio de uma API.
# MAGIC
# MAGIC Nesta demo usamos uma base **sintética** no formato de uma base de atendimentos de operadora (uma linha por atendimento), sem dados pessoais.

# COMMAND ----------

display(spark.table(bronze_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC Nosso primeiro passo é analisar os dados e preparar um conjunto de features.
# MAGIC
# MAGIC
# MAGIC Próximo: [Analisar os dados e preparar as features]($./01_feature_engineering)
