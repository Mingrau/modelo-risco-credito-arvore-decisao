# -*- coding: utf-8 -*-

"""
Modelo de Risco de Crédito com Árvore de Decisão

Objetivo:
Classificar proponentes de crédito entre bons e maus pagadores utilizando
Árvore de Decisão com Scikit-learn.

Dataset base:
Credit Card Approval Prediction - Kaggle

Autor:
Alexandre Carvalho Ramos
"""

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 2. CARREGAMENTO DA BASE
# ============================================================

# A base df_modelo.csv deve conter os dados cadastrais já consolidados
# com a variável resposta "mau".
df_modelo = pd.read_csv('df_modelo.csv')

print("Formato inicial da base:", df_modelo.shape)
print("\nTipos de dados:")
print(df_modelo.dtypes)

print("\nValores nulos por coluna:")
print(df_modelo.isna().sum())


# ============================================================
# 3. TRATAMENTO DE VALORES NULOS
# ============================================================

# A variável OCCUPATION_TYPE possui valores ausentes.
# Vamos preencher com uma categoria específica.
df_modelo['OCCUPATION_TYPE'] = df_modelo['OCCUPATION_TYPE'].fillna('Sem informacao')

print("\nTotal de valores nulos após tratamento:")
print(df_modelo.isna().sum().sum())


# ============================================================
# 4. TRATAMENTO DE VARIÁVEIS CATEGÓRICAS
# ============================================================

# O Scikit-learn não aceita variáveis em texto diretamente.
# Por isso, transformamos as variáveis categóricas em dummies.
variaveis_dummy = [
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE'
]

df_modelo_dummies = pd.get_dummies(
    df_modelo,
    columns=variaveis_dummy,
    drop_first=True,
    dtype=int
)


# ============================================================
# 5. SELEÇÃO DAS COLUNAS PARA O MODELO
# ============================================================

# Colunas numéricas originais que serão usadas no modelo.
colunas_numericas_modelo = [
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'FLAG_MOBIL',
    'FLAG_WORK_PHONE',
    'FLAG_PHONE',
    'FLAG_EMAIL',
    'CNT_FAM_MEMBERS'
]

# Identificar automaticamente as colunas dummies criadas.
colunas_dummies_modelo = [
    coluna for coluna in df_modelo_dummies.columns
    if any(coluna.startswith(var + '_') for var in variaveis_dummy)
]

# Criar a base final para Machine Learning.
df_colunas_modelo = df_modelo_dummies[
    colunas_numericas_modelo + colunas_dummies_modelo + ['mau']
].copy()

print("\nFormato da base final para ML:", df_colunas_modelo.shape)


# ============================================================
# 6. SEPARAÇÃO ENTRE X E y
# ============================================================

# X: variáveis explicativas
# y: variável resposta
X = df_colunas_modelo.drop(columns=['mau'])
y = df_colunas_modelo['mau']

print("\nFormato de X:", X.shape)
print("Formato de y:", y.shape)

print("\nColunas object em X:")
print(X.select_dtypes(include='object').columns)

print("\nTotal de nulos em X:")
print(X.isna().sum().sum())


# ============================================================
# 7. SEPARAÇÃO ENTRE TREINO E TESTE
# ============================================================

# 70% treino e 30% teste.
# stratify=y mantém a proporção de bons e maus nas duas bases.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTamanho das bases:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nDistribuição da variável resposta na base total:")
print(y.value_counts(normalize=True).mul(100).round(2))

print("\nDistribuição da variável resposta no treino:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nDistribuição da variável resposta no teste:")
print(y_test.value_counts(normalize=True).mul(100).round(2))


# ============================================================
# 8. MODELO FINAL COM ÁRVORE DE DECISÃO
# ============================================================

# Melhor alpha encontrado nos testes anteriores.
melhor_alpha = 1.9591914612592648e-05

arvore_final = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42,
    ccp_alpha=melhor_alpha
)

# Treinar o modelo.
arvore_final.fit(X_train, y_train)

# Classificar a base de teste.
y_pred_final = arvore_final.predict(X_test)


# ============================================================
# 9. AVALIAÇÃO DO MODELO FINAL
# ============================================================

acuracia_test_final = accuracy_score(y_test, y_pred_final)

print("\n==============================")
print("MODELO FINAL")
print("==============================")

print("Melhor ccp_alpha:", melhor_alpha)
print(f"Acurácia na base de teste: {acuracia_test_final:.2%}")

print("\nMatriz de confusão - Modelo Final:")
cm_final = confusion_matrix(y_test, y_pred_final)
print(cm_final)

print("\nRelatório de classificação - Modelo Final:")
print(classification_report(y_test, y_pred_final, zero_division=0))


# ============================================================
# 10. VISUALIZAÇÃO DA MATRIZ DE CONFUSÃO - MODELO FINAL
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_final,
    display_labels=['Aprovados', 'Reprovados']
)

disp.plot(
    ax=ax,
    cmap='Blues',
    values_format='d',
    colorbar=False
)

plt.title('Matriz de Confusão - Modelo Final')
plt.xlabel('Classe predita pelo modelo')
plt.ylabel('Classe real')
plt.show()


# ============================================================
# 11. VISUALIZAÇÃO DA ÁRVORE
# ============================================================

# Como a árvore final é grande, visualizamos apenas os primeiros níveis.
plt.figure(figsize=(30, 15))

plot_tree(
    arvore_final,
    feature_names=X.columns,
    class_names=['Aprovado', 'Reprovado'],
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3
)

plt.title('Árvore de Decisão - Primeiros Níveis')
plt.show()


# ============================================================
# 12. ACURÁCIA TREINO X TESTE - MODELO FINAL
# ============================================================

y_pred_train_final = arvore_final.predict(X_train)
y_pred_test_final = arvore_final.predict(X_test)

acuracia_train_final = accuracy_score(y_train, y_pred_train_final)
acuracia_test_final = accuracy_score(y_test, y_pred_test_final)

print("\nAcurácia do Modelo Final:")
print(f"Acurácia na base de treinamento: {acuracia_train_final:.2%}")
print(f"Acurácia na base de teste: {acuracia_test_final:.2%}")


# ============================================================
# 13. NOVA ÁRVORE SOLICITADA NO EXERCÍCIO
# ============================================================

# Nova árvore:
# - mínimo de observações por folha = 5
# - profundidade máxima = 10
# - random_state = 123

arvore_nova = DecisionTreeClassifier(
    min_samples_leaf=5,
    max_depth=10,
    random_state=123
)

arvore_nova.fit(X_train, y_train)

y_pred_nova = arvore_nova.predict(X_test)


# ============================================================
# 14. AVALIAÇÃO DA NOVA ÁRVORE
# ============================================================

print("\n==============================")
print("NOVA ÁRVORE")
print("==============================")

acuracia_test_nova = accuracy_score(y_test, y_pred_nova)

print(f"Acurácia na base de teste - Nova Árvore: {acuracia_test_nova:.2%}")

cm_nova = confusion_matrix(y_test, y_pred_nova)

print("\nMatriz de confusão - Nova Árvore:")
print(cm_nova)

print("\nRelatório de classificação - Nova Árvore:")
print(classification_report(y_test, y_pred_nova, zero_division=0))


# ============================================================
# 15. VISUALIZAÇÃO DA MATRIZ DE CONFUSÃO - NOVA ÁRVORE
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_nova,
    display_labels=['Aprovados', 'Reprovados']
)

disp.plot(
    ax=ax,
    cmap='Blues',
    values_format='d',
    colorbar=False
)

plt.title('Matriz de Confusão - Nova Árvore')
plt.xlabel('Classe predita pelo modelo')
plt.ylabel('Classe real')
plt.show()


# ============================================================
# 16. DISTRIBUIÇÃO DAS PREDIÇÕES DA NOVA ÁRVORE
# ============================================================

print("\nDistribuição absoluta das predições - Nova Árvore:")
print(pd.Series(y_pred_nova).value_counts())

print("\nDistribuição percentual das predições - Nova Árvore:")
print(pd.Series(y_pred_nova).value_counts(normalize=True).mul(100).round(2))

proporcao_maus_preditos = (y_pred_nova == 1).mean() * 100

print(f"\nProporção de proponentes classificados como maus: {proporcao_maus_preditos:.2f}%")


# ============================================================
# 17. ACURÁCIA TREINO X TESTE - NOVA ÁRVORE
# ============================================================

y_pred_train_nova = arvore_nova.predict(X_train)
y_pred_test_nova = arvore_nova.predict(X_test)

acuracia_train_nova = accuracy_score(y_train, y_pred_train_nova)
acuracia_test_nova = accuracy_score(y_test, y_pred_test_nova)

print("\nAcurácia da Nova Árvore:")
print(f"Acurácia na base de treinamento: {acuracia_train_nova:.2%}")
print(f"Acurácia na base de teste: {acuracia_test_nova:.2%}")


# ============================================================
# 18. ACURÁCIA CLASSIFICANDO TODOS COMO BONS
# ============================================================

# Como a base é desbalanceada, classificar todos como bons pode gerar
# uma acurácia alta, mesmo sem identificar nenhum mau pagador.
acuracia_todos_bons = (y_test == 0).mean()

print("\nAcurácia classificando todos os contratos como bons:")
print(f"Acurácia: {acuracia_todos_bons:.4f}")
print(f"Acurácia: {acuracia_todos_bons:.2%}")


# ============================================================
# 19. COMPARAÇÃO ENTRE MODELOS
# ============================================================

comparacao_modelos = pd.DataFrame([
    {
        'modelo': 'Árvore Final com ccp_alpha',
        'accuracy': accuracy_score(y_test, y_pred_final),
        'precision_reprovado': precision_score(y_test, y_pred_final, pos_label=1, zero_division=0),
        'recall_reprovado': recall_score(y_test, y_pred_final, pos_label=1, zero_division=0),
        'f1_reprovado': f1_score(y_test, y_pred_final, pos_label=1, zero_division=0)
    },
    {
        'modelo': 'Nova Árvore max_depth=10',
        'accuracy': accuracy_score(y_test, y_pred_nova),
        'precision_reprovado': precision_score(y_test, y_pred_nova, pos_label=1, zero_division=0),
        'recall_reprovado': recall_score(y_test, y_pred_nova, pos_label=1, zero_division=0),
        'f1_reprovado': f1_score(y_test, y_pred_nova, pos_label=1, zero_division=0)
    },
    {
        'modelo': 'Todos classificados como bons',
        'accuracy': acuracia_todos_bons,
        'precision_reprovado': 0,
        'recall_reprovado': 0,
        'f1_reprovado': 0
    }
])

print("\nComparação entre modelos:")
print(comparacao_modelos)


# ============================================================
# 20. IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

importancias = pd.DataFrame({
    'variavel': X.columns,
    'importancia': arvore_final.feature_importances_
})

importancias = importancias.sort_values(by='importancia', ascending=False)

print("\nTop 15 variáveis mais importantes:")
print(importancias.head(15))


# Visualização das 15 variáveis mais importantes
top15 = importancias.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top15['variavel'], top15['importancia'])
plt.gca().invert_yaxis()
plt.title('Top 15 Variáveis Mais Importantes - Árvore Final')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.show()