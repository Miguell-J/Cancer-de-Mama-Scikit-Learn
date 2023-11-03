# Classificação de Câncer de Mama usando Naive Bayes

<img src="https://www.nexofin.com/archivos/2020/08/doctor-with-a-pink-ribbon_1508929535-1536x722.jpg"/>

## Visão Geral

Este repositório contém um script em Python para construir e avaliar um modelo de classificação para o diagnóstico de câncer de mama usando o algoritmo Gaussian Naive Bayes. O código utiliza a biblioteca scikit-learn para carregar o conjunto de dados de câncer de mama, dividi-lo em conjuntos de treinamento e teste, treinar um modelo Gaussian Naive Bayes, fazer previsões e calcular a precisão do modelo.

## Dependências
Antes de executar o script, certifique-se de ter as seguintes bibliotecas Python instaladas:

- scikit-learn: Fornece ferramentas de aprendizado de máquina para classificação, regressão e muito mais.
- pandas: Uma biblioteca de manipulação e análise de dados.
- Você pode instalar essas bibliotecas usando pip:

```bash
pip install scikit-learn pandas
```

## Uso
Clone o repositório ou baixe o script para a sua máquina local.

### Execute o script:

```bash
python classificacao_cancer_mama.py
```

## Explicação do Código
O script pode ser dividido nas seguintes seções:

### Importar Bibliotecas

```python
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
```

Esta seção importa as bibliotecas Python necessárias para a manipulação de dados e aprendizado de máquina.

Carregar o Conjunto de Dados de Câncer de Mama:

```python
dados = load_breast_cancer()
```

Este trecho de código carrega o conjunto de dados de câncer de mama usando a função load_breast_cancer do scikit-learn.

### Dividir os Dados

```python
y = dados['target']
x = dados['data']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

Ele divide o conjunto de dados em recursos (x) e rótulos alvo (y) e, em seguida, realiza uma divisão em conjuntos de treinamento e teste usando a função train_test_split. Os dados são divididos em um conjunto de treinamento com 70% dos dados e um conjunto de teste com 30%.

Criar e Treinar o Modelo Naive Bayes:

```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
modelo = gnb.fit(x_train, y_train)
```

Nesta seção, importamos o classificador Gaussian Naive Bayes do scikit-learn, criamos uma instância do modelo e o treinamos com os dados de treinamento.

### Fazer Previsões

```python
previsoes = modelo.predict(x_test)
```

Ele utiliza o modelo treinado para fazer previsões nos dados de teste.

### Calcular a Precisão

```python
from sklearn.metrics import accuracy_score
print("A precisão do modelo é de: {:.2%}".format(accuracy_score(y_test, previsoes)))
>>> A precisão do modelo é de: 94.15%
```

Por fim, importamos a função accuracy_score do scikit-learn e calculamos a precisão das previsões do modelo nos dados de teste. O resultado é impresso no console como uma porcentagem.

## Resultados
O script irá mostrar a precisão do modelo Gaussian Naive Bayes no diagnóstico de câncer de mama com base no conjunto de dados fornecido.

Para quaisquer dúvidas ou análises adicionais, consulte o script e a documentação dentro do repositório.
