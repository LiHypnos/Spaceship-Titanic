# Spaceship Titanic — Mineração de Dados

Trabalho desenvolvido para a disciplina de Mineração de Dados, em junho de 2026, utilizando a competição Spaceship Titanic do Kaggle.

O projeto aborda um problema de classificação binária: prever, a partir das características dos passageiros, se eles foram transportados para outra dimensão.

Competição: https://www.kaggle.com/competitions/spaceship-titanic

Kaggle: Lianxx  
Score: 0.80360

## Sobre o problema

Em 2912, a nave espacial Spaceship Titanic sofreu uma colisão com uma anomalia espaço-temporal, fazendo com que aproximadamente metade dos passageiros desaparecesse.

O conjunto de dados contém informações sobre os passageiros, suas cabines, origem, destino, estado de criogenia e gastos realizados durante a viagem. O objetivo da competição é prever a variável `Transported`.

## Objetivos

O trabalho foi desenvolvido seguindo as principais etapas de um processo de mineração de dados e aprendizado de máquina:

- análise exploratória dos dados;
- tratamento de valores ausentes;
- análise estatística;
- engenharia de atributos;
- pré-processamento;
- treinamento e comparação de modelos;
- validação cruzada;
- otimização de hiperparâmetros;
- avaliação dos resultados;
- geração da submissão para o Kaggle.

## Dataset

Foram utilizados os arquivos disponibilizados pela competição:

| Arquivo | Descrição |
|---|---|
| `train.csv` | Dados utilizados para treinamento e validação |
| `test.csv` | Dados utilizados para gerar as previsões finais |
| `sample_submission.csv` | Modelo de submissão disponibilizado pelo Kaggle |

O conjunto de treino possui 8.693 passageiros e o conjunto de teste possui 4.277.

A variável `Transported` possui distribuição aproximadamente balanceada:

- `True`: 50,4%
- `False`: 49,6%

## Análise exploratória

Inicialmente foram analisados os tipos das variáveis, estatísticas descritivas, distribuição da variável-alvo, valores ausentes e relação entre os atributos e `Transported`.

Também foi analisado o comportamento das variáveis relacionadas aos gastos dos passageiros, como `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa` e `VRDeck`.

### Valores ausentes

Os valores ausentes foram tratados de acordo com o tipo da variável:

| Tipo de variável | Tratamento |
|---|---|
| Numéricas | Mediana |
| Categóricas | Moda |
| Variáveis de gasto | Ausência interpretada como gasto 0 |

## Análise estatística

Foi utilizado o teste Qui-Quadrado de independência para investigar a associação entre as variáveis categóricas e `Transported`.

Os resultados encontrados foram:

| Variável | χ² | p-valor |
|---|---:|---:|
| `HomePlanet` | 325,0 | 3,92 × 10⁻⁷⁰ |
| `CryoSleep` | 1861,7 | < 0,001 |
| `Destination` | 106,4 | 6,55 × 10⁻²³ |
| `VIP` | 12,1 | 2,36 × 10⁻³ |

No conjunto analisado, `CryoSleep` apresentou a maior estatística do teste entre as variáveis categóricas avaliadas.

## Engenharia de atributos

Foram criados novos atributos a partir das informações presentes no dataset.

| Atributo | Origem | Descrição |
|---|---|---|
| `Deck` | `Cabin` | Identifica o deck da cabine |
| `Cabin_num` | `Cabin` | Número da cabine |
| `Side` | `Cabin` | Lado da nave |
| `Group` | `PassengerId` | Identificador do grupo de viagem |
| `GroupSize` | `PassengerId` | Quantidade de passageiros no grupo |
| `IsAlone` | `GroupSize` | Indica se o passageiro viaja sozinho |
| `TotalSpend` | Variáveis de gasto | Soma dos gastos do passageiro |
| `HasSpending` | `TotalSpend` | Indica se houve algum gasto |

Por exemplo, a cabine:

```text
B/175/S
```

é transformada em:

```text
Deck      = B
Cabin_num = 175
Side      = S
```

Também foi criado o atributo `TotalSpend`:

```python
TotalSpend = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck
```

e o indicador:

```python
HasSpending = TotalSpend > 0
```

## Pré-processamento

O pré-processamento foi implementado com `Pipeline` e `ColumnTransformer` do Scikit-learn.

Para as variáveis numéricas foi utilizado:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

Para as variáveis categóricas:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OrdinalEncoder()
```

Foram utilizadas 6 variáveis categóricas:

```text
HomePlanet
CryoSleep
Destination
VIP
Deck
Side
```

e 11 variáveis numéricas:

```text
Age
RoomService
FoodCourt
ShoppingMall
Spa
VRDeck
Cabin_num
GroupSize
IsAlone
TotalSpend
HasSpending
```

Os dados foram separados de forma estratificada em 80% para treinamento e 20% para validação.

## Modelos avaliados

Foram comparados seis algoritmos de classificação:

- Regressão Logística
- Naive Bayes Gaussiano
- Árvore de Decisão
- Random Forest
- KNN
- Gradient Boosting

A avaliação inicial foi feita utilizando Stratified K-Fold Cross Validation com 5 folds.

| Modelo | Acurácia média | Desvio padrão |
|---|---:|---:|
| Regressão Logística | 0,7854 | 0,0141 |
| Naive Bayes | 0,7307 | 0,0140 |
| Árvore de Decisão | 0,7542 | 0,0046 |
| Random Forest | 0,7982 | 0,0081 |
| KNN (K=5) | 0,7844 | 0,0054 |
| Gradient Boosting | 0,8000 | 0,0081 |

## Otimização de hiperparâmetros

Após a comparação inicial, Random Forest e Gradient Boosting foram submetidos a uma busca de hiperparâmetros utilizando `RandomizedSearchCV`.

Foram avaliadas 25 combinações de parâmetros com 5 folds de validação cruzada.

### Random Forest

Melhores parâmetros encontrados:

```text
n_estimators      = 500
max_depth         = 20
max_features      = sqrt
min_samples_split = 2
min_samples_leaf  = 4
```

Acurácia média na validação cruzada:

```text
0.8034
```

### Gradient Boosting

Melhores parâmetros encontrados:

```text
n_estimators      = 300
learning_rate     = 0.05
max_depth         = 4
min_samples_split = 10
subsample         = 0.9
```

Acurácia média na validação cruzada:

```text
0.8079
```

## Avaliação final

Os modelos foram avaliados no conjunto de validação separado do treinamento.

| Modelo | Acurácia | AUC-ROC |
|---|---:|---:|
| Regressão Logística | 0,8028 | 0,8814 |
| Naive Bayes | 0,7504 | 0,8507 |
| Árvore de Decisão | 0,7545 | 0,7548 |
| Random Forest | 0,8028 | 0,8906 |
| KNN (K=5) | 0,7780 | 0,8511 |
| Gradient Boosting | 0,8108 | 0,9029 |
| Random Forest otimizado | 0,7993 | 0,8996 |
| Gradient Boosting otimizado | 0,8177 | 0,9100 |

O Gradient Boosting otimizado foi utilizado para a submissão final.

Resultados na validação:

```text
Accuracy: 81,77%
AUC-ROC:  0,9100
```

O relatório de classificação apresentou valores próximos de 0,82 para precisão, recall e F1-score nas duas classes.

## Importância das variáveis

As principais variáveis segundo a importância das features do Gradient Boosting foram:

| Variável | Importância |
|---|---:|
| `TotalSpend` | 20,06% |
| `HasSpending` | 16,20% |
| `FoodCourt` | 8,77% |
| `Cabin_num` | 7,34% |
| `Spa` | 7,16% |

As variáveis relacionadas aos gastos apresentaram participação relevante nas decisões do modelo.

## Submissão

Depois da etapa de avaliação e seleção do modelo, o pipeline final foi treinado utilizando os dados rotulados disponíveis e aplicado ao conjunto de teste.

As previsões foram salvas no arquivo:

```text
submission.csv
```

Formato:

```text
PassengerId,Transported
0013_01,True
0018_01,False
0019_01,True
...
```

Resultado obtido na competição:

```text
Kaggle Score: 0.80360
Kaggle User: Lianxx
```

## Estrutura do projeto

O principal arquivo deste repositório é:

```text
spaceship_titanic_v3.ipynb
```

O notebook contém todo o processo utilizado no trabalho:

```text
Carregamento dos dados
        ↓
Análise exploratória
        ↓
Engenharia de atributos
        ↓
Pré-processamento
        ↓
Treinamento dos modelos
        ↓
Validação cruzada
        ↓
Otimização de hiperparâmetros
        ↓
Avaliação
        ↓
Submissão no Kaggle
```

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Google Colab
- Kaggle

## Contexto acadêmico

Disciplina: Mineração de Dados  
Período: junho de 2026  
Competição: Spaceship Titanic  
Plataforma: Kaggle

## Referências

- Kaggle: https://www.kaggle.com/competitions/spaceship-titanic
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- SciPy: https://scipy.org/
