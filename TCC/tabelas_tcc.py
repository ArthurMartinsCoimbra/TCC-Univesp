
# ============================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PASTA = Path(__file__).resolve().parent
PASTA_TABELAS = PASTA / "tabelas"

PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300
})


# ============================================================
# 2. CARREGAR OS RESULTADOS
# ============================================================

arquivo = PASTA / "robustez_caracteristicas_treino.csv"

dados = pd.read_csv(arquivo)

associacoes_esperadas = {
    ("seasonal_strength", "R_ARIMA_AUTO"),
    ("seasonal_strength", "R_ETS_AUTO"),
    ("entropy", "R_XGBoost")
}

pares_encontrados = set(
    zip(dados["caracteristica"], dados["modelo"])
)

if pares_encontrados != associacoes_esperadas:
    raise ValueError(
        "As associações encontradas não correspondem "
        "às três associações esperadas."
    )

if len(dados) != 3:
    raise ValueError(
        "A tabela deve conter exatamente três associações."
    )

colunas_numericas = [
    "n_original",
    "rho_original",
    "n_treino",
    "rho_treino",
    "p_treino"
]

if not np.isfinite(
    dados[colunas_numericas].to_numpy()
).all():
    raise ValueError(
        "Foram encontrados valores numéricos inválidos."
    )


# ============================================================
# 3. PREPARAR TABELA ACADÊMICA
# ============================================================

nomes_caracteristicas = {
    "seasonal_strength": "Sazonalidade",
    "entropy": "Entropia"
}

nomes_modelos = {
    "R_ARIMA_AUTO": "ARIMA/SARIMA",
    "R_ETS_AUTO": "ETS",
    "R_XGBoost": "XGBoost"
}

ordem_modelos = {
    "R_ARIMA_AUTO": 1,
    "R_ETS_AUTO": 2,
    "R_XGBoost": 3
}

dados = dados.copy()

dados["ordem"] = dados["modelo"].map(ordem_modelos)

dados = dados.sort_values("ordem")

tabela = pd.DataFrame({
    "Característica": dados["caracteristica"].map(
        nomes_caracteristicas
    ),
    "Modelo": dados["modelo"].map(
        nomes_modelos
    ),
    "n": dados["n_treino"].astype(int),
    "ρ original": dados["rho_original"],
    "ρ treino": dados["rho_treino"],
    "Valor-p (treino)": dados["p_treino"]
})


# ============================================================
# 4. EXPORTAR CSV DA TABELA
# ============================================================

tabela.to_csv(
    PASTA_TABELAS / "tabela_correlacoes_tcc.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. FORMATAR PARA APRESENTAÇÃO
# ============================================================

tabela_formatada = tabela.copy()

for coluna in ["ρ original", "ρ treino", "Valor-p (treino)"]:
    tabela_formatada[coluna] = (
        tabela_formatada[coluna]
        .map(lambda valor: f"{valor:.4f}".replace(".", ","))
    )


# ============================================================
# 6. GERAR TABELA EM PNG E PDF
# ============================================================

fig, ax = plt.subplots(figsize=(11, 3.2))

ax.axis("off")

colunas = tabela_formatada.columns.tolist()

conteudo = tabela_formatada.values.tolist()

tabela_grafica = ax.table(
    cellText=conteudo,
    colLabels=colunas,
    cellLoc="center",
    colLoc="center",
    loc="center",
    colWidths=[0.19, 0.22, 0.08, 0.17, 0.17, 0.17]
)

tabela_grafica.auto_set_font_size(False)
tabela_grafica.set_fontsize(10)
tabela_grafica.scale(1, 1.9)


# Estilo acadêmico: poucas linhas, sem grade completa
for (linha, coluna), celula in tabela_grafica.get_celld().items():

    celula.set_facecolor("white")
    celula.set_edgecolor("black")
    celula.set_linewidth(0)

    if linha == 0:
        celula.set_text_props(weight="bold")
        celula.visible_edges = "TB"
        celula.set_linewidth(1)

    elif linha == len(tabela_formatada):
        celula.visible_edges = "B"
        celula.set_linewidth(1)
        
    else:
        celula.visible_edges = ""


# Título acima da tabela
fig.text(
    0.5,
    0.95,
    "Correlações de Spearman entre características "
    "estruturais e desempenho relativo",
    ha="center",
    fontsize=12,
    weight="bold"
)


# Nota abaixo da tabela
fig.text(
    0.08,
    0.13,
    "Nota: razão de desempenho = MAE do modelo / MAE do Naive. "
    "ρ original: série completa; ρ treino: treinamento inicial.",
    ha="left",
    fontsize=8.5
)

fig.text(
    0.08,
    0.07,
    "Fonte: Elaboração própria (2026).",
    ha="left",
    fontsize=8.5
)


plt.savefig(
    PASTA_TABELAS / "tabela_correlacoes_tcc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    PASTA_TABELAS / "tabela_correlacoes_tcc.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 7. FINALIZAÇÃO
# ============================================================

print("\nTabela acadêmica gerada com sucesso!")

print(
    tabela_formatada.to_string(index=False)
)

print("\nArquivos salvos em:")
print(PASTA_TABELAS)
