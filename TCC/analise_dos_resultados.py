# ============================================================
# 1. IMPORTAÇÕES
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 2. LOCALIZAR ARQUIVOS
# ============================================================

PASTA = Path(__file__).resolve().parent

arquivos = [
    "metricas_estatisticos.csv",
    "metricas_random_forest.csv",
    "metricas_xgboost.csv",
    "metricas_svr.csv"
]


# ============================================================
# 3. CARREGAR E UNIFICAR MÉTRICAS
# ============================================================

tabelas = []

for nome_arquivo in arquivos:

    caminho = PASTA / nome_arquivo

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {nome_arquivo}"
        )

    tabela = pd.read_csv(caminho)

    tabelas.append(tabela)


metricas = pd.concat(
    tabelas,
    ignore_index=True
)


# ============================================================
# 4. VALIDAR RESULTADOS
# ============================================================

modelos_esperados = {
    "Naive",
    "ETS_AUTO",
    "ARIMA_AUTO",
    "RandomForest",
    "XGBoost",
    "SVR"
}

ids_esperados = set(
    pd.read_csv(PASTA / "amostra_tcc.csv")["id_tsdl"]
)

combinacoes = pd.MultiIndex.from_product(
    [sorted(ids_esperados), sorted(modelos_esperados)],
    names=["id_tsdl", "modelo"]
)

if len(metricas) != 96:
    raise ValueError(
        f"Esperadas 96 linhas, encontradas {len(metricas)}."
    )

if metricas.duplicated(["id_tsdl", "modelo"]).any():
    raise ValueError(
        "Existem combinações série-modelo duplicadas."
    )

pares_observados = pd.MultiIndex.from_frame(
    metricas[["id_tsdl", "modelo"]]
)

if set(pares_observados) != set(combinacoes):
    raise ValueError(
        "Existem combinações série-modelo ausentes ou inesperadas."
    )

if not np.isfinite(
    metricas[["MAE", "RMSE", "MASE"]].to_numpy()
).all():
    raise ValueError(
        "Foram encontradas métricas inválidas."
    )

print("Validação concluída!")
print("Total de séries:", metricas["id_tsdl"].nunique())
print("Total de modelos:", metricas["modelo"].nunique())
print("Total de combinações:", len(metricas))


# ============================================================
# 5. SALVAR TABELA CONSOLIDADA
# ============================================================

metricas = metricas.sort_values(
    ["id_tsdl", "modelo"]
)

metricas.to_csv(
    PASTA / "metricas_consolidadas.csv",
    index=False
)

print("\nTabela consolidada salva com sucesso.")


# ============================================================
# 6. RESUMO INICIAL DOS MODELOS
# ============================================================

resumo = (
    metricas
    .groupby("modelo")
    .agg(
        MASE_medio=("MASE", "mean"),
        MASE_mediano=("MASE", "median"),
        MAE_medio=("MAE", "mean")
    )
    .sort_values("MASE_medio")
)

print("\nResumo geral:")
print(resumo.to_string())

# ============================================================
# 7. RANKING DOS MODELOS POR SÉRIE
# ============================================================

metricas["ranking"] = (
    metricas
    .groupby("id_tsdl")["MASE"]
    .rank(method="average", ascending=True)
)

# Ranking médio: menor é melhor
ranking_medio = (
    metricas
    .groupby("modelo")["ranking"]
    .mean()
    .sort_values()
)

print("\nRanking médio dos modelos:")
print(ranking_medio.to_string())


# ============================================================
# 8. VITÓRIAS POR MODELO
# ============================================================

vencedores = metricas.loc[
    metricas.groupby("id_tsdl")["MASE"].idxmin()
]

print("\nVitórias por modelo:")
print(vencedores["modelo"].value_counts())


# ============================================================
# 9. TABELA DE MASE POR SÉRIE
# ============================================================

tabela_mase = metricas.pivot(
    index="id_tsdl",
    columns="modelo",
    values="MASE"
)

print("\nMASE por série e modelo:")
print(tabela_mase.round(3).to_string())


# ============================================================
# 10. SALVAR RESULTADOS
# ============================================================

ranking_medio.to_csv(
    PASTA / "ranking_modelos.csv"
)

tabela_mase.to_csv(
    PASTA / "comparacao_mase_por_serie.csv"
)

print("\nRankings e comparações salvos em CSV.")

# ============================================================
# 11. CARREGAR CARACTERÍSTICAS ESTRUTURAIS
# ============================================================

caracteristicas = pd.read_csv(
    PASTA / "caracteristicas_series.csv"
)

print("\nDimensões das características:")
print(caracteristicas.shape)

print("\nValores ausentes:")
print(caracteristicas.isna().sum())


# ============================================================
# 12. JUNTAR CARACTERÍSTICAS E MASE
# ============================================================

# tabela_mase foi criada anteriormente na seção 9
tabela_mase = tabela_mase.reset_index()

analise_estrutural = caracteristicas.merge(
    tabela_mase,
    on="id_tsdl",
    how="inner",
    validate="one_to_one"
)

# Conferir se nenhuma série desapareceu
if len(analise_estrutural) != 16:
    raise ValueError(
        "A junção não preservou as 16 séries."
    )

# Conferir se todos os modelos estão presentes
colunas_modelos = [
    "Naive",
    "ETS_AUTO",
    "ARIMA_AUTO",
    "RandomForest",
    "XGBoost",
    "SVR"
]

if analise_estrutural[colunas_modelos].isna().any().any():
    raise ValueError(
        "Existem resultados de modelos ausentes."
    )


# ============================================================
# 13. SALVAR BASE DA ANÁLISE ESTRUTURAL
# ============================================================

analise_estrutural.to_csv(
    PASTA / "analise_estrutural.csv",
    index=False
)

print("\nBase estrutural criada com sucesso!")
print("Dimensões:", analise_estrutural.shape)

print("\nPrimeiras séries:")
print(
    analise_estrutural.head().round(3).to_string(index=False)
)

# ============================================================
# 14. COMPARAR AS FAMÍLIAS DE MÉTODOS
# ============================================================

modelos_estatisticos = [
    "Naive",
    "ETS_AUTO",
    "ARIMA_AUTO"
]

modelos_ml = [
    "RandomForest",
    "XGBoost",
    "SVR"
]


# Melhor MASE observado em cada família
analise_estrutural["melhor_estatistico"] = (
    analise_estrutural[modelos_estatisticos].min(axis=1)
)

analise_estrutural["melhor_ml"] = (
    analise_estrutural[modelos_ml].min(axis=1)
)


# Identificar qual modelo venceu dentro de cada família
analise_estrutural["modelo_estatistico"] = (
    analise_estrutural[modelos_estatisticos].idxmin(axis=1)
)

analise_estrutural["modelo_ml"] = (
    analise_estrutural[modelos_ml].idxmin(axis=1)
)


# Razão entre os melhores MASEs observados
analise_estrutural["razao_ml_estatistico"] = (
    analise_estrutural["melhor_ml"] /
    analise_estrutural["melhor_estatistico"]
)


# Família vencedora em cada série
analise_estrutural["familia_vencedora"] = np.select(
    [
        analise_estrutural["razao_ml_estatistico"] > 1,
        analise_estrutural["razao_ml_estatistico"] < 1
    ],
    [
        "Estatístico",
        "Machine Learning"
    ],
    default="Empate"
)


# ============================================================
# 15. EXIBIR RESULTADOS
# ============================================================

colunas = [
    "id_tsdl",
    "subject",
    "modelo_estatistico",
    "melhor_estatistico",
    "modelo_ml",
    "melhor_ml",
    "razao_ml_estatistico",
    "familia_vencedora"
]

print("\nComparação entre famílias:")

print(
    analise_estrutural[colunas]
    .round(3)
    .to_string(index=False)
)

print("\nVitórias por família:")

print(
    analise_estrutural["familia_vencedora"]
    .value_counts()
)


# ============================================================
# 16. SALVAR COMPARAÇÃO
# ============================================================

analise_estrutural.to_csv(
    PASTA / "comparacao_familias.csv",
    index=False
)

print("\nComparação entre famílias salva em CSV.")

# ============================================================
# 17. CORRELAÇÕES DE SPEARMAN
# ============================================================

from scipy.stats import spearmanr

caracteristicas_analisadas = [
    "trend",
    "seasonal_strength",
    "x_acf1",
    "entropy"
]

resultados_correlacao = []

for caracteristica in caracteristicas_analisadas:

    # Excluir somente linhas com valores ausentes
    base = analise_estrutural[
        [caracteristica, "razao_ml_estatistico"]
    ].dropna()

    rho, p_valor = spearmanr(
        base[caracteristica],
        base["razao_ml_estatistico"]
    )

    resultados_correlacao.append({
        "caracteristica": caracteristica,
        "n_series": len(base),
        "spearman_rho": rho,
        "p_valor": p_valor
    })


correlacoes = pd.DataFrame(resultados_correlacao)

print("\nCorrelações entre características e vantagem relativa:")
print(correlacoes.round(4).to_string(index=False))


# Salvar resultados
correlacoes.to_csv(
    PASTA / "correlacoes_estruturais.csv",
    index=False
)

# ============================================================
# 18. GRÁFICO - SAZONALIDADE X VANTAGEM RELATIVA
# ============================================================

import matplotlib.pyplot as plt

# Recuperar as frequências das séries
frequencias = (
    pd.read_csv(PASTA / "amostra_tcc.csv")
    [["id_tsdl", "frequency"]]
    .drop_duplicates()
)

base_grafico = analise_estrutural.merge(
    frequencias,
    on="id_tsdl",
    how="left",
    validate="one_to_one"
)

base_grafico = base_grafico.dropna(
    subset=["seasonal_strength", "razao_ml_estatistico"]
)

nomes_frequencia = {
    1: "Anual",
    4: "Trimestral",
    12: "Mensal",
    52: "Semanal"
}

fig, ax = plt.subplots(figsize=(10, 6))

for frequencia, grupo in base_grafico.groupby("frequency"):

    ax.scatter(
        grupo["seasonal_strength"],
        grupo["razao_ml_estatistico"],
        s=75,
        label=nomes_frequencia.get(frequencia, str(frequencia))
    )

    for _, linha in grupo.iterrows():

        ax.annotate(
            str(int(linha["id_tsdl"])),
            (
                linha["seasonal_strength"],
                linha["razao_ml_estatistico"]
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

# Linha de empate entre as famílias
ax.axhline(
    y=1,
    color="gray",
    linestyle="--",
    label="Empate entre famílias"
)

ax.set_xlabel("Intensidade da sazonalidade")
ax.set_ylabel("Razão MASE ML / Estatístico")

ax.set_title(
    "Sazonalidade e desempenho relativo entre famílias"
)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    PASTA / "sazonalidade_vs_desempenho.png",
    dpi=300
)

plt.show()

# ============================================================
# 19. ANÁLISE DE SENSIBILIDADE DA SAZONALIDADE
# ============================================================

from scipy.stats import spearmanr

base_sazonal = analise_estrutural[
    [
        "id_tsdl",
        "seasonal_strength",
        "razao_ml_estatistico"
    ]
].dropna()

# Correlação considerando todas as séries disponíveis
rho_total, p_total = spearmanr(
    base_sazonal["seasonal_strength"],
    base_sazonal["razao_ml_estatistico"]
)

# Remover temporariamente as três séries
# com maior sazonalidade, apenas para sensibilidade
ids_alta_sazonalidade = [12, 122, 639]

base_reduzida = base_sazonal[
    ~base_sazonal["id_tsdl"].isin(ids_alta_sazonalidade)
]

rho_reduzido, p_reduzido = spearmanr(
    base_reduzida["seasonal_strength"],
    base_reduzida["razao_ml_estatistico"]
)

print("\nAnálise de sensibilidade - Sazonalidade")

print(
    f"Amostra completa: n={len(base_sazonal)}, "
    f"rho={rho_total:.4f}, p={p_total:.4f}"
)

print(
    f"Sem as três séries mais sazonais: "
    f"n={len(base_reduzida)}, "
    f"rho={rho_reduzido:.4f}, "
    f"p={p_reduzido:.4f}"
)

# ============================================================
# 20. CORRELAÇÕES POR MODELO INDIVIDUAL
# ============================================================

from scipy.stats import spearmanr

caracteristicas_analisadas = [
    "trend",
    "seasonal_strength",
    "x_acf1",
    "entropy"
]

modelos = [
    "Naive",
    "ETS_AUTO",
    "ARIMA_AUTO",
    "RandomForest",
    "XGBoost",
    "SVR"
]

resultados_individuais = []

for caracteristica in caracteristicas_analisadas:

    for modelo in modelos:

        base = analise_estrutural[
            [caracteristica, modelo]
        ].dropna()

        rho, p_valor = spearmanr(
            base[caracteristica],
            base[modelo]
        )

        resultados_individuais.append({
            "caracteristica": caracteristica,
            "modelo": modelo,
            "n_series": len(base),
            "rho": rho,
            "p_valor": p_valor
        })


correlacoes_modelos = pd.DataFrame(
    resultados_individuais
)


# ============================================================
# 21. EXIBIR MATRIZ DE CORRELAÇÕES
# ============================================================

matriz_correlacoes = correlacoes_modelos.pivot(
    index="caracteristica",
    columns="modelo",
    values="rho"
)

print("\nCorrelações de Spearman por modelo:")
print(matriz_correlacoes.round(3).to_string())


# ============================================================
# 22. SALVAR RESULTADOS
# ============================================================

correlacoes_modelos.to_csv(
    PASTA / "correlacoes_por_modelo.csv",
    index=False
)

print("\nCorrelações individuais salvas em CSV.")

# ============================================================
# 23. SENSIBILIDADE DAS CORRELAÇÕES INDIVIDUAIS
# ============================================================

from scipy.stats import spearmanr

modelos = [
    "Naive",
    "ETS_AUTO",
    "ARIMA_AUTO",
    "RandomForest",
    "XGBoost",
    "SVR"
]

ids_alta_sazonalidade = [12, 122, 639]

resultados_sensibilidade = []

for modelo in modelos:

    base = analise_estrutural[
        ["id_tsdl", "seasonal_strength", modelo]
    ].dropna()

    base_reduzida = base[
        ~base["id_tsdl"].isin(ids_alta_sazonalidade)
    ]

    rho_total, _ = spearmanr(
        base["seasonal_strength"],
        base[modelo]
    )

    rho_reduzido, _ = spearmanr(
        base_reduzida["seasonal_strength"],
        base_reduzida[modelo]
    )

    resultados_sensibilidade.append({
        "modelo": modelo,
        "rho_completo": rho_total,
        "rho_sem_tres_series": rho_reduzido,
        "diferenca": rho_reduzido - rho_total
    })

sensibilidade_modelos = pd.DataFrame(
    resultados_sensibilidade
)

print("\nSensibilidade das correlações por modelo:")
print(
    sensibilidade_modelos
    .round(4)
    .to_string(index=False)
)

sensibilidade_modelos.to_csv(
    PASTA / "sensibilidade_modelos.csv",
    index=False
)

# ============================================================
# 24. CALCULAR ESCALAS DO MASE SAZONAL
# ============================================================

dados_originais = pd.read_csv(
    PASTA / "amostra_tcc.csv"
)

K = 10

escalas_sazonais = {}

for id_tsdl, grupo in dados_originais.groupby("id_tsdl"):

    grupo = grupo.sort_values("indice")

    serie = grupo["valor"].to_numpy()

    frequencia = int(grupo["frequency"].iloc[0])

    # Utilizar somente a primeira janela de treinamento
    treino_inicial = serie[:-K]

    # Período da referência sazonal
    m = frequencia

    if len(treino_inicial) <= m:
        raise ValueError(
            f"Série {id_tsdl}: treinamento insuficiente "
            f"para escala sazonal m={m}"
        )

    # Diferenças entre observações separadas por m períodos
    diferencas = np.abs(
        treino_inicial[m:] - treino_inicial[:-m]
    )

    escala_sazonal = np.mean(diferencas)

    if (
        not np.isfinite(escala_sazonal)
        or escala_sazonal <= 0
    ):
        raise ValueError(
            f"Escala sazonal inválida na série {id_tsdl}"
        )

    escalas_sazonais[id_tsdl] = escala_sazonal


# ============================================================
# 25. CALCULAR MASE SAZONAL DOS SEIS MODELOS
# ============================================================

metricas["escala_mase_sazonal"] = (
    metricas["id_tsdl"].map(escalas_sazonais)
)

metricas["MASE_sazonal"] = (
    metricas["MAE"] /
    metricas["escala_mase_sazonal"]
)

print("\nComparação dos dois tipos de MASE:")

print(
    metricas[
        [
            "id_tsdl",
            "modelo",
            "MASE",
            "MASE_sazonal"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 26. COMPARAR O MASE MÉDIO DOS MODELOS
# ============================================================

resumo_robustez = (
    metricas
    .groupby("modelo")
    .agg(
        MASE_original=("MASE", "mean"),
        MASE_sazonal=("MASE_sazonal", "mean")
    )
    .sort_values("MASE_original")
)

print("\nComparação das médias por modelo:")

print(
    resumo_robustez
    .round(4)
    .to_string()
)


# ============================================================
# 27. SALVAR RESULTADOS
# ============================================================

metricas.to_csv(
    PASTA / "metricas_com_mase_sazonal.csv",
    index=False
)

resumo_robustez.to_csv(
    PASTA / "resumo_robustez_mase.csv"
)

print("\nAnálise de robustez salva com sucesso.")

# ============================================================
# 28. CORRELAÇÕES COM MASE SAZONAL
# ============================================================

from scipy.stats import spearmanr

# Trazer as características para a tabela atualizada
base_robustez = metricas.merge(
    caracteristicas[
        ["id_tsdl", "seasonal_strength"]
    ],
    on="id_tsdl",
    how="left",
    validate="many_to_one"
)

resultados_robustez = []

for modelo, grupo in base_robustez.groupby("modelo"):

    # Considerar apenas séries com sazonalidade disponível
    grupo = grupo.dropna(
        subset=["seasonal_strength", "MASE", "MASE_sazonal"]
    )

    # Correlação com MASE original
    rho_original, p_original = spearmanr(
        grupo["seasonal_strength"],
        grupo["MASE"]
    )

    # Correlação com MASE sazonal
    rho_sazonal, p_sazonal = spearmanr(
        grupo["seasonal_strength"],
        grupo["MASE_sazonal"]
    )

    resultados_robustez.append({
        "modelo": modelo,
        "n_series": len(grupo),
        "rho_original": rho_original,
        "p_original": p_original,
        "rho_sazonal": rho_sazonal,
        "p_sazonal": p_sazonal
    })


tabela_robustez = pd.DataFrame(
    resultados_robustez
).sort_values("modelo")

print("\nSazonalidade x MASE: comparação das referências")

print(
    tabela_robustez
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 29. SALVAR RESULTADOS
# ============================================================

tabela_robustez.to_csv(
    PASTA / "correlacoes_mase_sazonal.csv",
    index=False
)

print("\nCorrelações de robustez salvas.")

# ============================================================
# 30. COMPARAÇÃO DOS MODELOS COM O NAIVE
# ============================================================

# Criar tabela de MAE por série e modelo
tabela_mae = metricas.pivot(
    index="id_tsdl",
    columns="modelo",
    values="MAE"
)

# Verificar se os MAEs do Naive são válidos
if (
    not np.isfinite(tabela_mae["Naive"]).all()
    or (tabela_mae["Naive"] <= 0).any()
):
    raise ValueError(
        "Existem valores inválidos de MAE para o Naive."
    )

# Calcular razões em relação ao Naive
razoes_naive = tabela_mae.div(
    tabela_mae["Naive"],
    axis=0
)

# Remover o próprio Naive (razão sempre igual a 1)
razoes_naive = razoes_naive.drop(
    columns="Naive"
)

# Renomear colunas
razoes_naive = razoes_naive.add_prefix("R_")

# Transformar id_tsdl em coluna
razoes_naive = razoes_naive.reset_index()


# ============================================================
# 31. JUNTAR COM AS CARACTERÍSTICAS ESTRUTURAIS
# ============================================================

analise_relativa = caracteristicas.merge(
    razoes_naive,
    on="id_tsdl",
    how="inner",
    validate="one_to_one"
)

if len(analise_relativa) != 16:
    raise ValueError(
        "A análise relativa não contém as 16 séries."
    )

print("\nRazões de MAE em relação ao Naive:")

print(
    analise_relativa.round(3).to_string(index=False)
)


# ============================================================
# 32. SALVAR RESULTADOS
# ============================================================

analise_relativa.to_csv(
    PASTA / "analise_relativa_naive.csv",
    index=False
)

print("\nAnálise relativa salva com sucesso.")

# ============================================================
# 33. CORRELAÇÕES DAS VANTAGENS RELATIVAS AO NAIVE
# ============================================================

from scipy.stats import spearmanr

caracteristicas_relativas = [
    "trend",
    "seasonal_strength",
    "x_acf1",
    "entropy"
]

modelos_relativos = [
    "R_ARIMA_AUTO",
    "R_ETS_AUTO",
    "R_RandomForest",
    "R_SVR",
    "R_XGBoost"
]

resultados = []

for caracteristica in caracteristicas_relativas:

    for modelo in modelos_relativos:

        base = analise_relativa[
            [caracteristica, modelo]
        ].dropna()

        rho, p_valor = spearmanr(
            base[caracteristica],
            base[modelo]
        )

        resultados.append({
            "caracteristica": caracteristica,
            "modelo": modelo,
            "n_series": len(base),
            "rho": rho,
            "p_valor": p_valor
        })


correlacoes_relativas = pd.DataFrame(resultados)

print("\nCorrelações relativas ao Naive:")
print(
    correlacoes_relativas
    .round(4)
    .to_string(index=False)
)

correlacoes_relativas.to_csv(
    PASTA / "correlacoes_relativas_naive.csv",
    index=False
)

# ============================================================
# 34. SENSIBILIDADE LEAVE-ONE-OUT
# ============================================================

associacoes = [
    ("seasonal_strength", "R_ARIMA_AUTO"),
    ("seasonal_strength", "R_ETS_AUTO"),
    ("entropy", "R_XGBoost")
]

resultados_loo = []

for caracteristica, modelo in associacoes:

    base = analise_relativa[
        ["id_tsdl", caracteristica, modelo]
    ].dropna()

    rho_original, _ = spearmanr(
        base[caracteristica],
        base[modelo]
    )

    for id_removido in base["id_tsdl"]:

        reduzida = base[
            base["id_tsdl"] != id_removido
        ]

        rho, p_valor = spearmanr(
            reduzida[caracteristica],
            reduzida[modelo]
        )

        resultados_loo.append({
            "caracteristica": caracteristica,
            "modelo": modelo,
            "id_removido": id_removido,
            "n_series": len(reduzida),
            "rho_original": rho_original,
            "rho_reduzido": rho,
            "p_valor": p_valor
        })

sensibilidade_loo = pd.DataFrame(resultados_loo)

# Resumir a variação dos coeficientes
resumo_loo = (
    sensibilidade_loo
    .groupby(["caracteristica", "modelo"])
    .agg(
        rho_original=("rho_original", "first"),
        menor_rho=("rho_reduzido", "min"),
        maior_rho=("rho_reduzido", "max")
    )
    .reset_index()
)

print("\nSensibilidade leave-one-out:")
print(resumo_loo.round(4).to_string(index=False))

# Salvar resultados detalhados e resumo
sensibilidade_loo.to_csv(
    PASTA / "sensibilidade_leave_one_out.csv",
    index=False
)

resumo_loo.to_csv(
    PASTA / "resumo_leave_one_out.csv",
    index=False
)

# ============================================================
# 35. DESEMPENHO RELATIVO POR FREQUÊNCIA
# ============================================================

frequencias = (
    pd.read_csv(PASTA / "amostra_tcc.csv")
    [["id_tsdl", "frequency"]]
    .drop_duplicates()
)

base_frequencias = analise_relativa.merge(
    frequencias,
    on="id_tsdl",
    how="left",
    validate="one_to_one"
)

nomes_frequencias = {
    1: "Anual",
    4: "Trimestral",
    12: "Mensal",
    52: "Semanal"
}

base_frequencias["frequencia_nome"] = (
    base_frequencias["frequency"]
    .map(nomes_frequencias)
)

colunas_razoes = [
    "R_ARIMA_AUTO",
    "R_ETS_AUTO",
    "R_RandomForest",
    "R_SVR",
    "R_XGBoost"
]

resumo_frequencias = (
    base_frequencias
    .groupby("frequencia_nome")[colunas_razoes]
    .median()
    .reindex(["Anual", "Trimestral", "Mensal", "Semanal"])
)

print("\nMediana da razão MAE/Naive por frequência:")
print(resumo_frequencias.round(3).to_string())

resumo_frequencias.to_csv(
    PASTA / "desempenho_por_frequencia.csv"
)

# ============================================================
# 36. TABELA-SÍNTESE DOS PRINCIPAIS ACHADOS
# ============================================================

sintese = pd.DataFrame([
    {
        "aspecto": "Desempenho geral",
        "resultado": "ARIMA 1,017; ETS 1,026 (MASE médio)",
        "interpretacao": "Melhores médias dos métodos estatísticos",
        "limitacao": "Resultados condicionados ao protocolo experimental"
    },
    {
        "aspecto": "Ranking médio",
        "resultado": "ARIMA 2,563; ETS 2,813",
        "interpretacao": "Liderança também nas posições médias",
        "limitacao": "Diferenças não implicam significância estatística"
    },
    {
        "aspecto": "Comparação entre famílias",
        "resultado": "Estatísticos 10 vitórias; ML 6",
        "interpretacao": "Predominância descritiva dos estatísticos",
        "limitacao": "Melhores modelos escolhidos retrospectivamente"
    },
    {
        "aspecto": "Sazonalidade e famílias",
        "resultado": "rho=0,612; p=0,060; n=10",
        "interpretacao": "Associação positiva sugestiva",
        "limitacao": "Sensível às três séries mais sazonais"
    },
    {
        "aspecto": "Sazonalidade versus Naive",
        "resultado": "ARIMA rho=-0,903; ETS rho=-0,867",
        "interpretacao": "Maior vantagem relativa em séries sazonais",
        "limitacao": "Associação exploratória; não causal"
    },
    {
        "aspecto": "Entropia e XGBoost",
        "resultado": "rho=0,703; p=0,0024",
        "interpretacao": "Menor vantagem relativa com maior entropia",
        "limitacao": "Dependência da referência Naive e da amostra"
    },
    {
        "aspecto": "Leave-one-out",
        "resultado": "Três associações conservaram o sinal",
        "interpretacao": "Estabilidade à exclusão individual",
        "limitacao": "Não comprova generalização"
    },
    {
        "aspecto": "MASE sazonal",
        "resultado": "ETS 0,544; ARIMA 0,577",
        "interpretacao": "Mudança na classificação agregada",
        "limitacao": "Normalização altera médias, não previsões"
    },
    {
        "aspecto": "Frequência temporal",
        "resultado": "Vantagens medianas trimestrais; dificuldades semanais",
        "interpretacao": "Possível heterogeneidade por frequência",
        "limitacao": "Apenas quatro séries por frequência"
    }
])

sintese.to_csv(
    PASTA / "tabela_sintese_tcc.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTabela-síntese dos resultados:")
print(sintese.to_string(index=False))

print("\nTabela-síntese salva com sucesso.")

# ============================================================
# 37. CARACTERÍSTICAS CALCULADAS NO TREINO INICIAL
# ============================================================

caracteristicas_treino = pd.read_csv(
    PASTA / "caracteristicas_treino.csv"
)

# Reutilizar as razões MAE/Naive já calculadas
base_treino = caracteristicas_treino.merge(
    razoes_naive,
    on="id_tsdl",
    how="inner",
    validate="one_to_one"
)

if len(base_treino) != 16:
    raise ValueError(
        "A base de treinamento não contém as 16 séries."
    )


# ============================================================
# 38. COMPARAR CORRELAÇÕES ORIGINAIS E DO TREINO
# ============================================================

associacoes = [
    ("seasonal_strength", "R_ARIMA_AUTO"),
    ("seasonal_strength", "R_ETS_AUTO"),
    ("entropy", "R_XGBoost")
]

resultados_temporais = []

for caracteristica, modelo in associacoes:

    # Características originais (série completa)
    base_original = analise_relativa[
        ["id_tsdl", caracteristica, modelo]
    ].dropna()

    # Características apenas do treino inicial
    base_nova = base_treino[
        ["id_tsdl", caracteristica, modelo]
    ].dropna()

    rho_original, p_original = spearmanr(
        base_original[caracteristica],
        base_original[modelo]
    )

    rho_treino, p_treino = spearmanr(
        base_nova[caracteristica],
        base_nova[modelo]
    )

    resultados_temporais.append({
        "caracteristica": caracteristica,
        "modelo": modelo,
        "n_original": len(base_original),
        "rho_original": rho_original,
        "p_original": p_original,
        "n_treino": len(base_nova),
        "rho_treino": rho_treino,
        "p_treino": p_treino
    })


comparacao_temporal = pd.DataFrame(
    resultados_temporais
)

print("\nRobustez temporal das associações:")
print(
    comparacao_temporal
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 39. SALVAR RESULTADOS
# ============================================================

comparacao_temporal.to_csv(
    PASTA / "robustez_caracteristicas_treino.csv",
    index=False
)

print("\nRobustez temporal salva com sucesso.")

# ============================================================
# 40. ATUALIZAR TABELA-SÍNTESE DO TCC
# ============================================================

# Carregar a síntese anterior
sintese_atualizada = pd.read_csv(
    PASTA / "tabela_sintese_tcc.csv"
)

# Carregar resultados da robustez temporal
robustez = pd.read_csv(
    PASTA / "robustez_caracteristicas_treino.csv"
)

# Atualizar resultados de associações já existentes
atualizacoes = {
    "Sazonalidade versus Naive": (
        "ARIMA e ETS mantiveram associações negativas "
        "fortes utilizando características do treino.",
        "Maior sazonalidade associada a maior vantagem "
        "relativa sobre o Naive.",
        "Amostra pequena; associação exploratória e não causal."
    ),

    "Entropia e XGBoost": (
        "Associação positiva permaneceu forte com "
        "características do treino.",
        "Maior entropia associada a menor vantagem "
        "relativa do XGBoost.",
        "Não demonstra causalidade nem generalização."
    )
}

for aspecto, valores in atualizacoes.items():

    filtro = sintese_atualizada["aspecto"] == aspecto

    if filtro.any():
        sintese_atualizada.loc[
            filtro,
            ["resultado", "interpretacao", "limitacao"]
        ] = list(valores)


# ============================================================
# 41. ADICIONAR RESULTADOS DA ROBUSTEZ TEMPORAL
# ============================================================

novas_linhas = []

for _, linha in robustez.iterrows():

    caracteristica = linha["caracteristica"]
    modelo = linha["modelo"]

    novas_linhas.append({
        "aspecto": f"Robustez temporal: {caracteristica} x {modelo}",

        "resultado": (
            f"rho original={linha['rho_original']:.4f}; "
            f"rho treino={linha['rho_treino']:.4f}; "
            f"p treino={linha['p_treino']:.4f}"
        ),

        "interpretacao": (
            "Associação preservada com características "
            "calculadas exclusivamente no treino inicial."
        ),

        "limitacao": (
            "Resultado exploratório; não comprova "
            "causalidade ou generalização."
        )
    })


novos_resultados = pd.DataFrame(novas_linhas)

# Evitar duplicações caso o código seja executado novamente
sintese_atualizada = sintese_atualizada[
    ~sintese_atualizada["aspecto"].isin(
        novos_resultados["aspecto"]
    )
]

sintese_atualizada = pd.concat(
    [sintese_atualizada, novos_resultados],
    ignore_index=True
)


# ============================================================
# 42. SALVAR NOVA TABELA
# ============================================================

arquivo_saida = PASTA / "tabela_sintese_tcc_atualizada.csv"

sintese_atualizada.to_csv(
    arquivo_saida,
    index=False,
    encoding="utf-8-sig"
)

print("\nTabela-síntese atualizada:")

print(
    sintese_atualizada.to_string(index=False)
)

print(
    "\nArquivo criado:",
    arquivo_saida
)