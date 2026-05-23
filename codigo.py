import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ── 1. CARREGAR OS DADOS ──────────────────────────────────────────────────────
print("Carregando dados...")

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
print(f"Total de jogadores carregados: {len(df)}")

# ── 2. SELECIONAR LIGA ────────────────────────────────────────────────────────
ligas = sorted(df["league_name"].unique())

print("\nLigas disponíveis:")
for idx, liga in enumerate(ligas):
    print(f"  [{idx}] {liga}")

while True:
    escolha = input("\nDigite o número ou o nome da liga: ").strip()

    if escolha.isdigit() and int(escolha) < len(ligas):
        LIGA = ligas[int(escolha)]
        break

    # Busca parcial (case-insensitive)
    matches = [l for l in ligas if escolha.lower() in l.lower()]
    if len(matches) == 1:
        LIGA = matches[0]
        break
    elif len(matches) > 1:
        print("Mais de uma liga encontrada, seja mais específico:")
        for m in matches:
            print(f"  - {m}")
    else:
        print("Liga não encontrada. Tente novamente.")

print(f"\nLiga selecionada: {LIGA}")

# ── 3. FILTRAR ────────────────────────────────────────────────────────────────
df = df[df["league_name"] == LIGA].reset_index(drop=True)
print(f"Jogadores na liga: {len(df)}")

# ── 4. SELECIONAR ATRIBUTOS DE DESEMPENHO ─────────────────────────────────────
atributos = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

for col in atributos:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)
print(f"Jogadores após limpeza: {len(df)}")

X = df[atributos]

# ── 5. NORMALIZAR ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 6. K-MEANS ────────────────────────────────────────────────────────────────
print("Aplicando K-Means...")

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# ── 7. EXEMPLOS DE JOGADORES POR CLUSTER ──────────────────────────────────────
print("\nExemplos de jogadores por cluster:")
for i in range(k):
    jogadores = df[df["cluster"] == i]["short_name"].head(5).values
    print(f"  Cluster {i}: {', '.join(jogadores)}")

# ── 8. GRÁFICO DE DISPERSÃO ───────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

colors = ["#1D9E75", "#534AB7", "#BA7517", "#A32D2D"]

plt.figure(figsize=(10, 6))
for i in range(k):
    subset = df[df["cluster"] == i]
    plt.scatter(subset["x"], subset["y"], label=f"Cluster {i}",
                alpha=0.5, s=15, color=colors[i])

plt.title(f"Agrupamento de jogadores FIFA — K-Means\n{LIGA}")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.tight_layout()
plt.savefig("clusters_futebol.png", dpi=150)
plt.show()
print("Salvo: clusters_futebol.png")

# ── 9. TABELA DE PERFIL POR CLUSTER ───────────────────────────────────────────
perfil = df.groupby("cluster")[atributos].mean().round(1)

nomes_cluster = {i: f"Cluster {i}" for i in range(k)}
cabecalhos = ["Velocidade", "Finalização", "Passe", "Drible", "Defesa", "Físico"]

fig, ax = plt.subplots(figsize=(12, 2.8))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.axis("off")

linhas = perfil.values.tolist()
rotulos_linha = [nomes_cluster[i] for i in perfil.index]

tabela = ax.table(
    cellText=linhas,
    rowLabels=rotulos_linha,
    colLabels=cabecalhos,
    cellLoc="center",
    rowLoc="center",
    loc="center",
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.scale(1.3, 2.4)

n_cols = len(cabecalhos)
for j in range(n_cols):
    cell = tabela[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#AAAAAA")

for i in range(k):
    cor_hex = colors[i]

    label_cell = tabela[i + 1, -1]
    label_cell.set_facecolor(cor_hex)
    label_cell.set_text_props(color="white", fontweight="bold")
    label_cell.set_edgecolor("#AAAAAA")

    for j in range(n_cols):
        cell = tabela[i + 1, j]
        cell.set_facecolor(cor_hex + "22")
        cell.set_edgecolor("#CCCCCC")

        col_vals = perfil.iloc[:, j]
        if perfil.iloc[i, j] == col_vals.max():
            cell.set_text_props(fontweight="bold")

ax.set_title(
    f"Perfil médio por cluster — {LIGA}",
    fontsize=13,
    fontweight="bold",
    pad=16,
    color="#2C3E50"
)

plt.tight_layout()
plt.savefig("tabela_clusters.png", dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.show()
print("Salvo: tabela_clusters.png")

print("\nPronto! Arquivos salvos: clusters_futebol.png e tabela_clusters.png")