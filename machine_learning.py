# Importando as bibliotecas
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Lendo o Excel (use o caminho correto do seu arquivo)
dados = pd.read_excel("filmes.xlsx")

# Verificando se as colunas existem
colunas_necessarias = ["duracao", "tem_acao", "classificacao"]
assert all(col in dados.columns for col in colunas_necessarias), "Coluna faltando no Excel!"

# Definindo features (X) e rótulo (y)
X = dados[["duracao", "tem_acao"]]
y = dados["classificacao"]

# Dividindo em treino (80%) e teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criando e treinando o modelo
modelo = DecisionTreeClassifier(random_state=42)  # random_state para reprodutibilidade
modelo.fit(X_treino, y_treino)

# Avaliando o modelo
previsoes = modelo.predict(X_teste)
precisao = accuracy_score(y_teste, previsoes)
print(f"Precisão do modelo: {precisao:.2%}")  # Ex: 85.00%

# Prevendo a classificação de um novo filme
novo_filme = pd.DataFrame([[101, 1]], columns=["duracao", "tem_acao"])
resultado = modelo.predict(novo_filme)
print(f"Classificação prevista para o novo filme: {resultado[0]}")