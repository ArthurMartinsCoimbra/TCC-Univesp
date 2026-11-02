
# ============================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# Caminhos do projeto
PASTA = Path(__file__).resolve().parent
PASTA_FIGURAS = PASTA / "figuras"

# Criar pasta automaticamente, caso não exista
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)


# Nomes dos modelos que aparecerão nos gráficos
NOMES = {
    "Naive": "Naive",
    "ETS_AUTO": "ETS",
    "ARIMA_AUTO": "ARIMA/SARIMA",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "SVR": "SVR"
}


# Configurações visuais gerais
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300
})


# ============================================================
# 2. FUNÇÃO PARA SALVAR AS FIGURAS
# ============================================================

def salvar_figura(fig, nome):
    """
    Salva uma figura em PNG (300 DPI) e PDF vetorial.
    """

    caminho_png = PASTA_FIGURAS / f"{nome}.png"
    caminho_pdf = PASTA_FIGURAS / f"{nome}.pdf"

    fig.savefig(
        caminho_png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        caminho_pdf,
        bbox_inches="tight"
    )

    print(f"Figura salva: {nome}")

    plt.close(fig)


# ============================================================
# 3. CARREGAR OS RESULTADOS DO EXPERIMENTO
# ============================================================

metricas = pd.read_csv(
    PASTA / "metricas_consolidadas.csv"
)

familias = pd.read_csv(
    PASTA / "comparacao_familias.csv"
)


# ============================================================
# 4. VALIDAR OS DADOS
# ============================================================

modelos_esperados = set(NOMES.keys())

if metricas["id_tsdl"].nunique() != 16:
    raise ValueError(
        "Esperadas 16 séries na tabela de métricas."
    )

if set(metricas["modelo"]) != modelos_esperados:
    raise ValueError(
        "Os modelos encontrados não correspondem aos seis esperados."
    )

if len(metricas) != 96:
    raise ValueError(
        "Esperadas 96 combinações série-modelo."
    )

if metricas.duplicated(["id_tsdl", "modelo"]).any():
    raise ValueError(
        "Existem combinações série-modelo duplicadas."
    )

if not np.isfinite(metricas["MASE"]).all():
    raise ValueError(
        "Existem valores inválidos de MASE."
    )

if len(familias) != 16:
    raise ValueError(
        "Esperadas 16 séries na comparação entre famílias."
    )

if familias["id_tsdl"].duplicated().any():
    raise ValueError(
        "Existem séries duplicadas na comparação entre famílias."
    )

if not np.isfinite(
    familias["razao_ml_estatistico"]
).all():
    raise ValueError(
        "Existem razões inválidas na comparação entre famílias."
    )

if (familias["razao_ml_estatistico"] <= 0).any():
    raise ValueError(
        "As razões entre famílias devem ser positivas."
    )

if set(metricas["id_tsdl"]) != set(familias["id_tsdl"]):
    raise ValueError(
        "Os identificadores de séries não coincidem entre os arquivos."
    )

print("Arquivos carregados e validados corretamente.")


# ============================================================
# 5. FIGURA 1 - MASE MÉDIO E MEDIANO
# ============================================================

resumo = (
    metricas
    .groupby("modelo")["MASE"]
    .agg(["mean", "median"])
    .sort_values("mean")
)

nomes_modelos = [
    NOMES.get(modelo, modelo)
    for modelo in resumo.index
]

posicoes = np.arange(len(resumo))
largura = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))


# Barras da média
barras_media = ax.bar(
    posicoes - largura / 2,
    resumo["mean"],
    largura,
    label="Média",
    color="tab:blue"
)


# Barras da mediana
barras_mediana = ax.bar(
    posicoes + largura / 2,
    resumo["median"],
    largura,
    label="Mediana",
    color="tab:orange"
)


# Linha de referência do MASE
ax.axhline(
    y=1,
    color="gray",
    linestyle="--",
    linewidth=1,
    alpha=0.8,
    label="MASE = 1"
)


# Identificação dos eixos
ax.set_xticks(posicoes)

ax.set_xticklabels(
    nomes_modelos,
    rotation=15,
    ha="right"
)

ax.set_ylabel("MASE")

ax.set_title(
    "MASE médio e mediano dos métodos de previsão"
)


# Legenda e grade
ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

ax.set_axisbelow(True)


# Mostrar valores acima das barras
ax.bar_label(
    barras_media,
    fmt="%.2f",
    padding=3,
    fontsize=8
)

ax.bar_label(
    barras_mediana,
    fmt="%.2f",
    padding=3,
    fontsize=8
)


# Ajustar espaço vertical para os rótulos
ax.set_ylim(
    0,
    max(resumo["mean"].max(), resumo["median"].max()) * 1.15
)


plt.tight_layout()

salvar_figura(
    fig,
    "figura_1_mase_modelos"
)


# ============================================================
# 6. FIGURA 2 - RANKING MÉDIO DOS MODELOS
# ============================================================

# Calcular a posição de cada modelo em cada série.
# Menor MASE recebe a melhor posição.
metricas["posicao"] = (
    metricas
    .groupby("id_tsdl")["MASE"]
    .rank(
        method="average",
        ascending=True
    )
)


# Calcular o ranking médio dos modelos
ranking_medio = (
    metricas
    .groupby("modelo")["posicao"]
    .mean()
    .sort_values()
)


nomes_ranking = [
    NOMES.get(modelo, modelo)
    for modelo in ranking_medio.index
]


fig, ax = plt.subplots(figsize=(9, 5))


barras = ax.barh(
    nomes_ranking,
    ranking_medio.values,
    color="tab:blue"
)


# Colocar a melhor posição no topo
ax.invert_yaxis()


ax.set_xlabel(
    "Posição média (menor é melhor)"
)

ax.set_title(
    "Ranking médio dos métodos nas 16 séries temporais"
)


# Como temos seis modelos, as posições variam de 1 a 6
ax.set_xlim(0, 6)


ax.grid(
    axis="x",
    alpha=0.25
)

ax.set_axisbelow(True)


# Mostrar valores ao lado das barras
ax.bar_label(
    barras,
    fmt="%.2f",
    padding=4
)


plt.tight_layout()

salvar_figura(
    fig,
    "figura_2_ranking_medio"
)


# ============================================================
# 7. FIGURA 3 - VANTAGEM RELATIVA ENTRE FAMÍLIAS
# ============================================================

familias = familias.sort_values(
    "id_tsdl"
).copy()


ids = familias["id_tsdl"].astype(str)

razoes = familias["razao_ml_estatistico"]


# Definir cor conforme a família vencedora
#
# Razão > 1: vantagem dos modelos estatísticos
# Razão < 1: vantagem dos modelos de ML
# Razão = 1: igualdade de desempenho

cores = [
    (
        "tab:blue" if razao > 1
        else "tab:orange" if razao < 1
        else "gray"
    )
    for razao in razoes
]


fig, ax = plt.subplots(figsize=(11, 5.5))


barras = ax.bar(
    ids,
    razoes,
    color=cores
)


# Linha horizontal indicando igualdade de desempenho
ax.axhline(
    y=1,
    linestyle="--",
    color="black",
    linewidth=1
)


# Identificação dos eixos
ax.set_xlabel(
    "Identificador da série"
)

ax.set_ylabel(
    "Razão MAE (ML / Estatísticos)"
)

ax.set_title(
    "Vantagem relativa entre famílias de métodos"
)


# Legenda explicativa
elementos_legenda = [
    Patch(
        facecolor="tab:blue",
        label="Vantagem estatística"
    ),
    Patch(
        facecolor="tab:orange",
        label="Vantagem de Machine Learning"
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        label="Igualdade de desempenho"
    )
]

ax.legend(
    handles=elementos_legenda,
    loc="upper left"
)


# Grade horizontal
ax.grid(
    axis="y",
    alpha=0.25
)

ax.set_axisbelow(True)


# Ajustar identificadores
ax.tick_params(
    axis="x",
    labelrotation=45
)


# Garantir espaço vertical para as barras
ax.set_ylim(
    0,
    max(razoes.max() * 1.12, 1.1)
)


plt.tight_layout()

salvar_figura(
    fig,
    "figura_3_comparacao_familias"
)


# ============================================================
# 8. FINALIZAÇÃO
# ============================================================

print("\nTodas as figuras foram geradas com sucesso!")

print(
    "Pasta de saída:",
    PASTA_FIGURAS
)
