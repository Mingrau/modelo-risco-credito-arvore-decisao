# Modelo de Risco de Crédito com Árvore de Decisão

Este projeto utiliza **Machine Learning** para classificar proponentes de crédito entre **bons pagadores** e **maus pagadores**, usando um modelo de **Árvore de Decisão** em Python.

O objetivo é simular uma situação real de análise de crédito, em que uma instituição precisa decidir se uma proposta deve ser **aprovada** ou **reprovada** com base em dados cadastrais e histórico de crédito.

---

## Objetivo do projeto

Construir um modelo capaz de prever se um cliente pode ser classificado como:

- **Aprovado**: cliente com menor risco de inadimplência
- **Reprovado**: cliente com maior risco de inadimplência

A variável resposta utilizada foi:
mau


Onde: 

0 = bom pagador / aprovado
1 = mau pagador / reprovado

--

## Base de dados

A base utilizada foi o dataset público Credit Card Approval Prediction, disponível no Kaggle.

O projeto utiliza informações como:

dados cadastrais dos clientes
renda
gênero
posse de veículo
posse de imóvel
tipo de renda
escolaridade
estado civil
tipo de moradia
ocupação
histórico de crédito
Etapas do projeto

O projeto foi desenvolvido seguindo as seguintes etapas:

Carregamento da base de dados
Tratamento de valores nulos
Análise dos tipos de variáveis
Transformação de variáveis categóricas com get_dummies
Separação das variáveis explicativas X e variável resposta y
Divisão da base em treino e teste
Treinamento de modelo de Árvore de Decisão
Avaliação com matriz de confusão
Análise de acurácia, precision, recall e f1-score
Ajuste do modelo com class_weight e ccp_alpha
Comparação com uma nova árvore de decisão
Discussão sobre o impacto da acurácia em bases desbalanceadas
Tecnologias utilizadas
Python
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook / Google Colab
Git e GitHub
Tratamento dos dados

O Scikit-learn não aceita variáveis categóricas em formato de texto diretamente. Por isso, as variáveis categóricas foram transformadas em variáveis dummy.

Exemplo de variáveis tratadas:

CODE_GENDER
FLAG_OWN_CAR
FLAG_OWN_REALTY
NAME_INCOME_TYPE
NAME_EDUCATION_TYPE
NAME_FAMILY_STATUS
NAME_HOUSING_TYPE
OCCUPATION_TYPE

Também foi realizado o tratamento de valores nulos na variável:

OCCUPATION_TYPE

Os valores ausentes foram preenchidos com:

Sem informacao
Separação treino e teste

A base foi dividida da seguinte forma:

70% para treinamento
30% para teste

Foi utilizado o parâmetro stratify=y para manter a mesma proporção da variável resposta nas bases de treino e teste.

Desbalanceamento da base

A base apresentou forte desbalanceamento entre as classes:

Bons pagadores: aproximadamente 98,31%
Maus pagadores: aproximadamente 1,69%

Por esse motivo, a acurácia isolada pode ser enganosa.

Um modelo que classificasse todos os clientes como bons pagadores teria uma acurácia alta, mas não seria útil para identificar clientes de maior risco.

--

## Modelos testados

Foram avaliadas diferentes versões de Árvore de Decisão.

1. **Árvore inicial**

Modelo simples de Árvore de Decisão.

Apesar de apresentar alta acurácia, o modelo não foi eficiente para identificar maus pagadores.

2. **Árvore com class_weight='balanced'**

Foi utilizado o parâmetro class_weight='balanced' para dar mais peso à classe minoritária.

Essa abordagem aumentou a capacidade do modelo de identificar maus pagadores, mas também elevou a quantidade de bons clientes classificados como maus.

3. **Árvore com ccp_alpha**

Foi aplicado o parâmetro ccp_alpha, responsável pela poda da árvore.

O objetivo foi encontrar um equilíbrio melhor entre:

identificar maus pagadores
reduzir a reprovação incorreta de bons clientes
manter boa acurácia geral
Resultado do modelo final

O modelo final utilizou:

DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42,
    ccp_alpha=1.9591914612592648e-05
)

Na base de teste, o modelo apresentou aproximadamente:

Métrica	Resultado
Acurácia	96,16%
Precision - Reprovado	19%
Recall - Reprovado	38%
F1-score - Reprovado	25%
Matriz de confusão do modelo final

A matriz de confusão do modelo final foi:

[[10447   306]
 [  114    71]]

Interpretação:

Situação	Quantidade
Bons clientes aprovados corretamente	10.447
Bons clientes reprovados por engano	306
Maus pagadores aprovados por engano	114
Maus pagadores reprovados corretamente	71
Interpretação de negócio

No contexto de crédito, existem dois tipos principais de erro.

Aprovar um mau pagador

Esse erro representa risco de inadimplência.

Cliente mau → modelo aprova
Reprovar um bom pagador

Esse erro representa perda de oportunidade comercial.

Cliente bom → modelo reprova

A escolha do melhor modelo depende da estratégia da empresa.

Uma política mais conservadora pode buscar reduzir a aprovação de maus pagadores, mesmo que isso aumente a reprovação de bons clientes.

Uma política mais flexível pode aprovar mais clientes, aceitando maior risco de inadimplência.

Principal aprendizado

O principal aprendizado do projeto foi que acurácia alta não significa necessariamente um bom modelo, especialmente em bases desbalanceadas.

Neste caso, um modelo que classificasse todos os contratos como bons teria acurácia próxima de 98,31%, mas não identificaria nenhum mau pagador.

Por isso, em problemas de crédito, é fundamental analisar também:

matriz de confusão
precision
recall
f1-score
proporção de clientes classificados como maus
impacto de cada tipo de erro no negócio
Como executar o projeto

*Clone o repositório:*

git clone https://github.com/Mingrau/modelo-risco-credito-arvore-decisao.git

*Acesse a pasta:*

cd modelo-risco-credito-arvore-decisao

*Instale as dependências:*

pip install -r requirements.txt

*Execute o script ou abra o notebook:*

python modelo_arvore_decisao_credito.py
Estrutura sugerida do repositório
modelo-risco-credito-arvore-decisao/
│
├── README.md
├── modelo_arvore_decisao_credito.py
├── ML_MAU_PAGADOR.ipynb
├── requirements.txt
└── data/
    └── df_modelo.csv

--

## Autor

***Alexandre Carvalho Ramos***

Projeto desenvolvido como parte dos estudos em Ciência de Dados, com foco em Machine Learning aplicado à análise de risco de crédito.

