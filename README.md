# Agrupamento de Jogadores FIFA com K-Means

Aplicação de aprendizado não supervisionado para identificação de perfis táticos de jogadores de futebol a partir do dataset FIFA, utilizando o algoritmo K-Means.

---

## Sobre o Projeto

Este projeto aplica o algoritmo **K-Means** sobre atributos de desempenho do dataset FIFA para agrupar jogadores em perfis táticos distintos. O sistema permite filtrar por liga específica (Premier League, Brasileirão, etc.) diretamente no terminal.

Os quatro clusters identificados na Premier League foram:

| Cluster | Perfil | Exemplos |
|---|---|---|
| Cluster 0 | Atacantes Velozes | Bryan Gil, T. Lamptey, P. Shalulile |
| Cluster 1 | Defensores Físicos | Gabriel, S. Botman, J. Evans, L. Dunk |
| Cluster 2 | Craques Ofensivos | De Bruyne, Salah, Kane, Haaland |
| Cluster 3 | Meio-campistas Completos | Casemiro, Van Dijk, Cancelo, Kanté |

---

## Pré-requisitos

- Python 3.10+
- pip

---

## Instalação

**1. Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

**2. Instale as dependências:**
```bash
pip install pandas scikit-learn matplotlib numpy
```

**3. Baixe o dataset FIFA** ⚠️

O arquivo `male_players.csv` **não está incluído** no repositório por ser muito grande (+500MB).

Faça o download gratuito no Kaggle:

👉 [kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)

Após baixar, coloque o arquivo `male_players.csv` na **raiz do projeto** (mesma pasta do `codigo.py`).

---

## Como Usar

Execute o script:
```bash
python codigo.py
```

O programa vai listar todas as ligas disponíveis e pedir para você escolher:
```
[0] Brazilian Série A
[1] English Premier League
[2] Germany 1. Bundesliga
...

Digite o número ou o nome da liga: premier
Liga selecionada: English Premier League
```

Após a execução, dois arquivos serão gerados:
- `clusters_futebol.png` — gráfico de dispersão (PCA)
- `tabela_clusters.png` — tabela de perfil médio por cluster

---

## Documentação Técnica

### Dependências

| Biblioteca | Uso |
|---|---|
| `pandas` | Carregamento e manipulação dos dados |
| `scikit-learn` | Normalização (StandardScaler), K-Means e PCA |
| `matplotlib` | Geração dos gráficos e tabela de resultados |
| `numpy` | Operações numéricas auxiliares |

---

### Etapa 1 — Carregamento dos Dados

```python
colunas_necessarias = [
    "short_name", "player_positions", "league_name",
    "pace", "shooting", "passing", "dribbling", "defending", "physic"
]

chunks = []
for chunk in pd.read_csv(
    "male_players.csv",
    usecols=colunas_necessarias,
    chunksize=10000,
    engine="python",
    on_bad_lines="skip",
):
    chunks.append(chunk.dropna())

df = pd.concat(chunks, ignore_index=True)
```

O arquivo `male_players.csv` possui quase **9 milhões de linhas**. Carregar tudo de uma vez consumiria vários GB de RAM. A solução é ler o arquivo em **chunks de 10.000 linhas**, carregando apenas as colunas necessárias.

| Parâmetro | Valor | Significado |
|---|---|---|
| `usecols` | lista de colunas | Carrega só o necessário, ignorando as demais |
| `chunksize` | 10000 | Lê o arquivo em blocos de 10.000 linhas |
| `engine` | python | Motor mais tolerante a inconsistências |
| `on_bad_lines` | skip | Ignora linhas corrompidas ou mal formatadas |

---

### Etapa 2 — Seleção Interativa de Liga

```python
ligas = sorted(df["league_name"].unique())

while True:
    escolha = input("Digite o número ou o nome da liga: ").strip()

    if escolha.isdigit() and int(escolha) < len(ligas):
        LIGA = ligas[int(escolha)]
        break

    matches = [l for l in ligas if escolha.lower() in l.lower()]
    if len(matches) == 1:
        LIGA = matches[0]
        break
    elif len(matches) > 1:
        print("Mais de uma liga encontrada, seja mais específico:")
    else:
        print("Liga não encontrada. Tente novamente.")
```

O sistema aceita dois tipos de entrada:
- **Número** — índice da liga na lista (ex: `1`)
- **Nome parcial** — parte do nome em qualquer capitalização (ex: `premier` encontra `English Premier League`)

O loop `while True` garante que o programa só avance quando uma liga válida for selecionada.

---

### Etapa 3 — Atributos de Desempenho

```python
atributos = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

for col in atributos:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)
```

| Atributo | Tradução | Descrição |
|---|---|---|
| `pace` | Velocidade | Capacidade de sprint e aceleração |
| `shooting` | Finalização | Precisão e potência nos chutes |
| `passing` | Passe | Qualidade e alcance dos passes |
| `dribbling` | Drible | Habilidade de superar adversários |
| `defending` | Defesa | Capacidade de marcar e interceptar |
| `physic` | Físico | Força física e resistência |

Todos os atributos são expressos em escala de **0 a 99**. O `pd.to_numeric(errors="coerce")` converte valores inválidos para `NaN`, que são removidos pelo `dropna()`.

---

### Etapa 4 — Normalização

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

O **StandardScaler** transforma cada atributo para ter **média zero** e **desvio padrão unitário**:

```
z = (x - média) / desvio_padrão
```

O K-Means baseia-se em distâncias euclidianas. Sem normalização, atributos com valores maiores dominariam o cálculo. A normalização garante que todos contribuam de forma equânime.

---

### Etapa 5 — K-Means

```python
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
```

| Parâmetro | Valor | Significado |
|---|---|---|
| `n_clusters` | 4 | Número de clusters |
| `random_state` | 42 | Seed para reprodutibilidade |
| `n_init` | 10 | Inicializações independentes — seleciona a de menor inércia |

**Por que k=4?** Corresponde aos quatro perfis táticos mais comuns no futebol moderno: atacantes velozes, defensores físicos, craques ofensivos e meio-campistas completos.

**Por que n_init=10?** O K-Means pode convergir para mínimos locais. Com 10 inicializações, o algoritmo escolhe automaticamente a melhor configuração.

---

### Etapa 6 — Visualização (PCA)

```python
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
```

O **PCA** reduz os dados de 6 dimensões para 2, permitindo plotar os clusters em um gráfico 2D.

> ⚠️ O PCA é usado **apenas para visualização**. Os clusters foram calculados pelo K-Means sobre os dados originais com 6 atributos — a redução não altera os grupos.

---

### Etapa 7 — Tabela de Perfil por Cluster

```python
perfil = df.groupby("cluster")[atributos].mean().round(1)
```

Calcula a média de cada atributo por cluster. A tabela gerada é estilizada com:
- Cabeçalho escuro com texto branco
- Cor distinta por cluster nos rótulos de linha
- **Maior valor de cada coluna em negrito**
- Salva em alta resolução (`dpi=180`) para uso em artigos

---

### Fluxo Completo

```
male_players.csv
      │
      ▼
 Carregamento em chunks (10k linhas)
      │
      ▼
 Seleção interativa de liga
      │
      ▼
 Filtragem + limpeza dos atributos
      │
      ▼
 Normalização (StandardScaler)
      │
      ▼
 K-Means (k=4, n_init=10)
      │
      ├──▶ Gráfico de dispersão PCA → clusters_futebol.png
      │
      └──▶ Tabela de perfil médio  → tabela_clusters.png
```

---

## Referência

AKHANLI, Serhat Emre; HENNIG, Christian. *Clustering of football players based on performance data and aggregated clustering validity indexes.* Journal of Quantitative Analysis in Sports, 2023.

---

## Autor

Ygor Luckas Sousa Ramos  
Disciplina: Mineração de Dados  
Orientador: Walass Froes de Oliveira
