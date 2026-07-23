# Bank Churn Prediction (Evasão Bancária) 🏦

Repository for a notebook from a Machine Learning class (IBMEC).

Projeto de classificação que prevê a **evasão de clientes (churn)** em um banco, usando o [Churn Dataset](https://github.com/cassiusf/datasets/raw/refs/heads/main/desafios/Churn_Dataset.csv), com foco em regressão logística como modelo benchmark para um problema de classificação desbalanceado.

## Arquivo

### `Cópia_de_01_Evasão_bancária.ipynb`

Notebook único, organizado em três grandes etapas:

**1. EDA (Análise Exploratória)**
- Carregamento do dataset (~10 mil registros de clientes bancários).
- Limpeza: remoção de 2 linhas duplicadas e 4 registros com valores ausentes; exclusão de `RowNumber` e `CustomerId` (identificadores sem valor preditivo).
- **Análise exploratória de clusters**: aplicação de K-Means (com método do cotovelo, Silhouette Score e Davies-Bouldin) para investigar segmentações naturais nos dados antes da modelagem supervisionada.
- **Análise univariada** de cada variável (`CreditScore`, `Surname`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`), incluindo boxplots, histogramas e verificação de outliers via IQR.
- Decisão de **remover a variável `Surname`** por alta cardinalidade (~3 mil valores únicos), pouco útil para a maioria dos modelos.
- Confirmação de que o dataset é **fortemente desbalanceado** em relação à variável alvo `Exited`.

**2. Pré-processamento**
- Split treino/teste (80/20), estratificado pelo cluster identificado na EDA.
- Pipeline com `ColumnTransformer`: `RobustScaler` para variáveis numéricas (mais robusto a outliers) e `OneHotEncoder` (drop first) para as variáveis categóricas `Geography` e `Gender`.
- Análise da matriz de correlação entre as features processadas e a variável alvo.

**3. Modelagem**
- **Modelo**: Regressão Logística com `class_weight="balanced"` (pesos calculados automaticamente pelo scikit-learn: ~0.628 para a classe majoritária e ~2.440 para a classe minoritária, dado o desbalanceamento).
- **Threshold customizado** (0.3 em vez do padrão 0.5): ajustado para priorizar a captura de clientes que realmente vão sair (reduzindo falsos negativos), aceitando mais falsos positivos em troca.
- **Avaliação**: Classification Report (precision, recall, F1) e ROC AUC, tanto no treino quanto no teste.
- **Resultado**: recall de ~93% na classe de churn — o modelo identifica quase todos os clientes que sairiam, ao custo de mais falsos positivos (erro tipo I).

## Dataset

- **Fonte**: [Churn_Dataset.csv](https://github.com/cassiusf/datasets/raw/refs/heads/main/desafios/Churn_Dataset.csv)
- **Tamanho**: ~10.000 registros de clientes bancários
- **Variável alvo**: `Exited` (1 = cliente saiu do banco, 0 = permaneceu)
- **Principais features**: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

## Como rodar

Desenvolvido no Google Colab. Principais dependências:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
```

## Principais decisões e aprendizados

- Em datasets fortemente desbalanceados, **ajustar o threshold de decisão** é uma alavanca simples e eficaz para priorizar recall na classe minoritária, sem precisar recorrer a técnicas de reamostragem.
- `class_weight="balanced"` já ajuda o modelo a não ignorar a classe minoritária durante o treino, complementando o ajuste de threshold.
- Variáveis de altíssima cardinalidade (como `Surname`) tendem a atrapalhar mais do que ajudar em modelos lineares simples, e foram excluídas por essa razão.
