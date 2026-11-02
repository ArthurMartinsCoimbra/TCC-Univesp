import pandas as pd
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import warnings

# ============================================================
# 1. CARREGAR OS DADOS
# ============================================================

dados = pd.read_csv("amostra_tcc.csv")

print(dados.head())
print("\nDimensões:", dados.shape)

print("\nSéries encontradas:")
print(
    dados.groupby("id_tsdl")
    .size()
)

# ============================================================
# 2. CONFIGURAÇÕES DO EXPERIMENTO
# ============================================================

K = 10
H = 1


# ============================================================
# 3. FUNÇÃO ROLLING-ORIGIN COM NAIVE
# ============================================================

def rolling_origin_naive(serie, id_tsdl, k=10):

    n = len(serie)
    resultados = []

    if n <= k:
        raise ValueError(
            f"Série {id_tsdl}: observações insuficientes."
        )

    for i in range(k):

        fim_treino = n - k + i

        treino = serie[:fim_treino]

        real = serie[fim_treino]

        previsao = treino[-1]

        resultados.append({
            "id_tsdl": id_tsdl,
            "modelo": "Naive",
            "origem": i + 1,
            "n_treino": len(treino),
            "real": real,
            "previsao": previsao,
            "erro": real - previsao,
            "erro_absoluto": abs(real - previsao)
        })

    return resultados


# ============================================================
# 4. EXECUTAR NAS 16 SÉRIES
# ============================================================

todos_resultados = []

for id_tsdl, grupo in dados.groupby("id_tsdl"):

    serie = (
        grupo
        .sort_values("indice")["valor"]
        .to_numpy()
    )

    resultados_serie = rolling_origin_naive(
        serie,
        id_tsdl,
        k=K
    )

    todos_resultados.extend(resultados_serie)


resultados_naive = pd.DataFrame(todos_resultados)


# ============================================================
# 5. VALIDAR RESULTADOS
# ============================================================

print("\nTotal de previsões:", len(resultados_naive))

print("\nPrevisões por série:")
print(
    resultados_naive
    .groupby("id_tsdl")
    .size()
)

print("\nPrimeiras previsões:")
print(resultados_naive.head(10))

# ============================================================
# 6. CALCULAR MÉTRICAS - MAE E RMSE
# ============================================================


metricas_naive = (
    resultados_naive
    .groupby(["id_tsdl", "modelo"])
    .agg(
        MAE=("erro_absoluto", "mean"),
        RMSE=("erro", lambda x: np.sqrt(np.mean(x**2)))
    )
    .reset_index()
)

print("\nMétricas do Naive por série:")
print(metricas_naive.to_string(index=False))

# ============================================================
# 7. CALCULAR MASE
# ============================================================

escalas_mase = {}

for id_tsdl, grupo in dados.groupby("id_tsdl"):

    # Recuperar a série em ordem cronológica
    serie = (
        grupo
        .sort_values("indice")["valor"]
        .to_numpy()
    )

    # Utilizar apenas o treinamento da primeira origem
    treino_inicial = serie[:-K]

    # Calcular a escala do Naive não sazonal
    escala = np.mean(
        np.abs(np.diff(treino_inicial))
    )

    # Verificar se a escala é válida
    if not np.isfinite(escala) or escala <= 0:
        raise ValueError(
            f"Escala MASE inválida para a série {id_tsdl}"
        )

    escalas_mase[id_tsdl] = escala


# Adicionar a escala correspondente a cada série
metricas_naive["escala_mase"] = (
    metricas_naive["id_tsdl"].map(escalas_mase)
)

# Calcular o MASE
metricas_naive["MASE"] = (
    metricas_naive["MAE"] /
    metricas_naive["escala_mase"]
)

print("\nMétricas completas do Naive:")
print(metricas_naive.to_string(index=False))

# ============================================================
# 8. TESTAR ETS EM UMA SÉRIE
# ============================================================



id_teste = 516

serie = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")["valor"]
    .to_numpy()
)

n = len(serie)

resultados_ets = []

for i in range(K):

    fim_treino = n - K + i

    treino = serie[:fim_treino]
    real = serie[fim_treino]

    # Ajustar ETS somente com os dados disponíveis
    modelo = ETSModel(
        treino,
        error="add",
        trend="add",
        seasonal=None
    )

    ajuste = modelo.fit(disp=False)

    # Prever um período à frente
    previsao = float(ajuste.forecast(steps=1)[0])

    resultados_ets.append({
        "origem": i + 1,
        "n_treino": len(treino),
        "real": real,
        "previsao": previsao,
        "erro": real - previsao
    })


resultados_ets = pd.DataFrame(resultados_ets)

print("\nRolling-origin - ETS - Série 516:")
print(resultados_ets.to_string(index=False))

# ============================================================
# 9. FUNÇÃO ROLLING-ORIGIN - ETS
# ============================================================

def rolling_origin_ets(serie, id_tsdl, k=10):

    n = len(serie)
    resultados = []

    if n <= k:
        raise ValueError(
            f"Série {id_tsdl}: observações insuficientes."
        )

    for i in range(k):

        fim_treino = n - k + i

        treino = serie[:fim_treino]
        real = serie[fim_treino]

        modelo = ETSModel(
            treino,
            error="add",
            trend="add",
            seasonal=None
        )

        ajuste = modelo.fit(disp=False)

        previsao = float(
            ajuste.forecast(steps=1)[0]
        )

        resultados.append({
            "id_tsdl": id_tsdl,
            "modelo": "ETS_AAN",
            "origem": i + 1,
            "n_treino": len(treino),
            "real": real,
            "previsao": previsao,
            "erro": real - previsao,
            "erro_absoluto": abs(real - previsao)
        })

    return resultados


# ============================================================
# 10. EXECUTAR ETS NAS 16 SÉRIES
# ============================================================

todos_resultados_ets = []

for id_tsdl, grupo in dados.groupby("id_tsdl"):

    serie = (
        grupo
        .sort_values("indice")["valor"]
        .to_numpy()
    )

    resultados_serie = rolling_origin_ets(
        serie,
        id_tsdl,
        k=K
    )

    todos_resultados_ets.extend(resultados_serie)


resultados_ets_completo = pd.DataFrame(
    todos_resultados_ets
)

print("\nTotal de previsões ETS:",
      len(resultados_ets_completo))

print("\nPrevisões ETS por série:")
print(
    resultados_ets_completo
    .groupby("id_tsdl")
    .size()
)

# ============================================================
# 12. CALCULAR MÉTRICAS DO ETS
# ============================================================

metricas_ets = (
    resultados_ets_completo
    .groupby(["id_tsdl", "modelo"])
    .agg(
        MAE=("erro_absoluto", "mean"),
        RMSE=("erro", lambda x: np.sqrt(np.mean(x**2)))
    )
    .reset_index()
)

# Reutilizar as mesmas escalas calculadas para o Naive
metricas_ets["escala_mase"] = (
    metricas_ets["id_tsdl"].map(escalas_mase)
)

metricas_ets["MASE"] = (
    metricas_ets["MAE"] /
    metricas_ets["escala_mase"]
)


# ============================================================
# 13. COMPARAR NAIVE E ETS
# ============================================================

comparacao = pd.concat(
    [metricas_naive, metricas_ets],
    ignore_index=True
)

print("\nComparação Naive x ETS:")
print(
    comparacao[
        ["id_tsdl", "modelo", "MAE", "RMSE", "MASE"]
    ].sort_values(["id_tsdl", "modelo"]).to_string(index=False)
)


# Contar vitórias considerando o menor MASE por série
ranking = comparacao.loc[
    comparacao.groupby("id_tsdl")["MASE"].idxmin()
]

print("\nQuantidade de séries com menor MASE por modelo:")
print(ranking["modelo"].value_counts())

# ============================================================
# 14. SELEÇÃO DE CONFIGURAÇÃO ETS POR AIC
# ============================================================



def selecionar_ets(treino, frequencia):

    configuracoes = [
        ("ETS_ANN", None, None),
        ("ETS_AAN", "add", None)
    ]

    # Adicionar modelos sazonais quando houver
    # pelo menos dois ciclos completos
    if frequencia > 1 and len(treino) >= 2 * frequencia:

        configuracoes.extend([
            ("ETS_ANA", None, "add"),
            ("ETS_AAA", "add", "add")
        ])

    candidatos = []

    for nome, tendencia, sazonalidade in configuracoes:

        try:
            modelo = ETSModel(
                treino,
                error="add",
                trend=tendencia,
                seasonal=sazonalidade,
                seasonal_periods=(
                    frequencia if sazonalidade else None
                )
            )

            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                ajuste = modelo.fit(disp=False)

            # Não aceitar ajustes que não convergiram
            convergiu = ajuste.mle_retvals.get(
                "converged", False
            )

            if not convergiu or not np.isfinite(ajuste.aic):
                continue

            candidatos.append({
                "modelo": nome,
                "AIC": ajuste.aic,
                "ajuste": ajuste
            })

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

    if not candidatos:
        raise RuntimeError(
            "Nenhuma configuração ETS convergiu."
        )

    # Menor AIC entre os ajustes válidos
    melhor = min(candidatos, key=lambda x: x["AIC"])

    return melhor


# ============================================================
# 15. TESTAR NA PRIMEIRA ORIGEM DA SÉRIE 122
# ============================================================

id_teste = 122

grupo = dados[dados["id_tsdl"] == id_teste]

serie = (
    grupo.sort_values("indice")["valor"].to_numpy()
)

frequencia = int(grupo["frequency"].iloc[0])

treino = serie[:-K]

melhor = selecionar_ets(treino, frequencia)

previsao = float(
    melhor["ajuste"].forecast(steps=1)[0]
)

print("\nSeleção automática ETS - Série 122")
print("Tamanho do treino:", len(treino))
print("Frequência:", frequencia)
print("Modelo selecionado:", melhor["modelo"])
print("AIC:", melhor["AIC"])
print("Previsão:", previsao)
print("Valor real:", serie[-K])

# ============================================================
# 16. ROLLING-ORIGIN COM SELEÇÃO AUTOMÁTICA DE ETS
# ============================================================

def rolling_origin_ets_auto(serie, id_tsdl, frequencia, k=10):

    n = len(serie)
    resultados = []

    if n <= k:
        raise ValueError(
            f"Série {id_tsdl}: observações insuficientes."
        )

    for i in range(k):

        fim_treino = n - k + i

        # Dados disponíveis nesta origem
        treino = serie[:fim_treino]

        # Valor reservado para avaliação
        real = serie[fim_treino]

        # Selecionar ETS apenas com o treinamento
        melhor = selecionar_ets(
            treino,
            frequencia
        )

        # Previsão de um período à frente
        previsao = float(
            melhor["ajuste"].forecast(steps=1)[0]
        )

        resultados.append({
            "id_tsdl": id_tsdl,
            "modelo": "ETS_AUTO",
            "configuracao": melhor["modelo"],
            "origem": i + 1,
            "n_treino": len(treino),
            "real": real,
            "previsao": previsao,
            "erro": real - previsao,
            "erro_absoluto": abs(real - previsao),
            "AIC": melhor["AIC"]
        })

    return resultados


# ============================================================
# 17. EXECUTAR NAS 16 SÉRIES
# ============================================================

todos_resultados_ets_auto = []

for id_tsdl, grupo in dados.groupby("id_tsdl"):

    grupo = grupo.sort_values("indice")

    serie = grupo["valor"].to_numpy()

    frequencia = int(
        grupo["frequency"].iloc[0]
    )

    print(f"Processando série {id_tsdl}...")

    resultados_serie = rolling_origin_ets_auto(
        serie,
        id_tsdl,
        frequencia,
        k=K
    )

    todos_resultados_ets_auto.extend(
        resultados_serie
    )


resultados_ets_auto = pd.DataFrame(
    todos_resultados_ets_auto
)


# ============================================================
# 18. VALIDAR RESULTADOS
# ============================================================

print(
    "\nTotal de previsões ETS automático:",
    len(resultados_ets_auto)
)

print("\nPrevisões por série:")

print(
    resultados_ets_auto
    .groupby("id_tsdl")
    .size()
)


# ============================================================
# 19. VERIFICAR CONFIGURAÇÕES SELECIONADAS
# ============================================================

print("\nConfigurações ETS selecionadas:")

print(
    pd.crosstab(
        resultados_ets_auto["id_tsdl"],
        resultados_ets_auto["configuracao"]
    )
)

# ============================================================
# 20. CALCULAR MÉTRICAS DO ETS AUTOMÁTICO
# ============================================================

metricas_ets_auto = (
    resultados_ets_auto
    .groupby(["id_tsdl", "modelo"])
    .agg(
        MAE=("erro_absoluto", "mean"),
        RMSE=("erro", lambda x: np.sqrt(np.mean(x**2)))
    )
    .reset_index()
)

# Utilizar as mesmas escalas do MASE calculadas
# com o treinamento inicial de cada série
metricas_ets_auto["escala_mase"] = (
    metricas_ets_auto["id_tsdl"].map(escalas_mase)
)

metricas_ets_auto["MASE"] = (
    metricas_ets_auto["MAE"] /
    metricas_ets_auto["escala_mase"]
)


# ============================================================
# 21. COMPARAR NAIVE E ETS AUTOMÁTICO
# ============================================================

comparacao_final = pd.concat(
    [metricas_naive, metricas_ets_auto],
    ignore_index=True
)

print("\nComparação Naive x ETS automático:")

print(
    comparacao_final[
        ["id_tsdl", "modelo", "MAE", "RMSE", "MASE"]
    ]
    .sort_values(["id_tsdl", "modelo"])
    .to_string(index=False)
)


# ============================================================
# 22. CONTAR VITÓRIAS POR MASE
# ============================================================

vencedores = comparacao_final.loc[
    comparacao_final.groupby("id_tsdl")["MASE"].idxmin()
]

print("\nVitórias por modelo (menor MASE):")

print(
    vencedores["modelo"].value_counts()
)

# ============================================================
# 23. TESTAR ARIMA EM UMA SÉRIE
# ============================================================

from statsmodels.tsa.arima.model import ARIMA

id_teste = 516

serie = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")["valor"]
    .to_numpy()
)

n = len(serie)

resultados_arima = []

for i in range(K):

    fim_treino = n - K + i

    # Somente os dados disponíveis nesta origem
    treino = serie[:fim_treino]

    # Valor reservado para avaliação
    real = serie[fim_treino]

    # Ajustar ARIMA(1,1,1)
    modelo = ARIMA(
        treino,
        order=(1, 1, 1)
    )

    ajuste = modelo.fit()


    # Verificar a convergência da otimização
    convergiu = ajuste.mle_retvals.get("converged", False)

    if not convergiu:
        print(
            f"ATENÇÃO: ARIMA não convergiu na origem {i + 1}"
        )

    # Prever um período à frente
    previsao = float(
        ajuste.forecast(steps=1)[0]
    )

    resultados_arima.append({
        "origem": i + 1,
        "n_treino": len(treino),
        "real": real,
        "previsao": previsao,
        "erro": real - previsao
    })


resultados_arima = pd.DataFrame(
    resultados_arima
)

print("\nRolling-origin - ARIMA - Série 516:")
print(resultados_arima.to_string(index=False))

# ============================================================
# 24. SELEÇÃO AUTOMÁTICA DE ARIMA POR AIC
# ============================================================

import itertools
import warnings

def selecionar_arima(treino):

    candidatos = []

    # Valores permitidos para cada parâmetro
    valores_p = [0, 1, 2]
    valores_d = [0, 1]
    valores_q = [0, 1, 2]

    combinacoes = itertools.product(
        valores_p,
        valores_d,
        valores_q
    )

    for p, d, q in combinacoes:

        try:

            # Constante para d=0; deriva para d=1
            tendencia = "c" if d == 0 else "t"

            modelo = ARIMA(
                treino,
                order=(p, d, q),
                trend=tendencia
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ajuste = modelo.fit()

            # Conferir convergência
            convergiu = ajuste.mle_retvals.get(
                "converged", False
            )

            if not convergiu:
                continue

            # Conferir AIC
            if not np.isfinite(ajuste.aic):
                continue

            candidatos.append({
                "ordem": (p, d, q),
                "AIC": ajuste.aic,
                "ajuste": ajuste
            })

        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError
        ):
            continue

    if not candidatos:
        raise RuntimeError(
            "Nenhuma configuração ARIMA convergiu."
        )

    # Selecionar menor AIC
    melhor = min(
        candidatos,
        key=lambda x: x["AIC"]
    )

    return melhor


# ============================================================
# 25. TESTAR ARIMA AUTOMÁTICO NA SÉRIE 516
# ============================================================

id_teste = 516

serie = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")["valor"]
    .to_numpy()
)

# Primeira janela de treinamento
treino = serie[:-K]

# Selecionar configuração
melhor = selecionar_arima(treino)

# Fazer previsão h=1
previsao = float(
    melhor["ajuste"].forecast(steps=1)[0]
)

print("\nSeleção automática ARIMA - Série 516")
print("Tamanho do treino:", len(treino))
print("Ordem selecionada:", melhor["ordem"])
print("AIC:", melhor["AIC"])
print("Previsão:", previsao)
print("Valor real:", serie[-K])

# ============================================================
# 26. TESTAR SARIMA NA SÉRIE 122
# ============================================================

id_teste = 122

grupo = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")
)

serie = grupo["valor"].to_numpy()

frequencia = int(grupo["frequency"].iloc[0])

# Primeira janela de treinamento
treino = serie[:-K]

# SARIMA(1,1,1)(1,0,0,12)
modelo = ARIMA(
    treino,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 0, frequencia),
    trend="t"
)

ajuste = modelo.fit()

# Conferir convergência
convergiu = ajuste.mle_retvals.get(
    "converged", False
)

# Prever h=1
previsao = float(
    ajuste.forecast(steps=1)[0]
)

print("\nTeste SARIMA - Série 122")
print("Tamanho do treino:", len(treino))
print("Frequência:", frequencia)
print("Ordem:", (1, 1, 1))
print("Ordem sazonal:", (1, 0, 0, frequencia))
print("Convergiu:", convergiu)
print("AIC:", ajuste.aic)
print("Previsão:", previsao)
print("Valor real:", serie[-K])

# ============================================================
# 27. TESTAR SELEÇÃO RESTRITA DE SARIMA
# ============================================================

id_teste = 122

grupo = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")
)

serie = grupo["valor"].to_numpy()
frequencia = int(grupo["frequency"].iloc[0])

treino = serie[:-K]

# Candidatos sazonais predefinidos
configuracoes_sarima = [
    ((0, 1, 0), (1, 0, 0, frequencia)),
    ((1, 1, 0), (1, 0, 0, frequencia)),
    ((0, 1, 1), (0, 0, 1, frequencia)),
    ((1, 1, 1), (1, 0, 0, frequencia))
]

resultados_busca = []

for ordem, ordem_sazonal in configuracoes_sarima:

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            modelo = ARIMA(
                treino,
                order=ordem,
                seasonal_order=ordem_sazonal,
                trend="t"
            )

            ajuste = modelo.fit()

        convergiu = ajuste.mle_retvals.get(
            "converged", False
        )

        if not convergiu or not np.isfinite(ajuste.aic):
            print(
                f"Ajuste inválido: {ordem} {ordem_sazonal}"
            )
            continue

        previsao = float(
            ajuste.forecast(steps=1)[0]
        )

        resultados_busca.append({
            "ordem": str(ordem),
            "ordem_sazonal": str(ordem_sazonal),
            "AIC": ajuste.aic,
            "previsao": previsao
        })

    except (ValueError, RuntimeError, np.linalg.LinAlgError) as erro:
        print(
            f"Erro no ajuste {ordem} {ordem_sazonal}: {erro}"
        )


tabela_busca = pd.DataFrame(resultados_busca)

if tabela_busca.empty:
    raise RuntimeError(
        "Nenhum candidato SARIMA válido."
    )

tabela_busca = tabela_busca.sort_values("AIC")

print("\nComparação de candidatos SARIMA:")
print(tabela_busca.to_string(index=False))

print("\nMelhor candidato sazonal:")
print(tabela_busca.iloc[0].to_string())

# ============================================================
# 28. SELEÇÃO CONJUNTA ARIMA / SARIMA
# ============================================================

def selecionar_arima_sarima(treino, frequencia):

    # 1. Selecionar o melhor ARIMA não sazonal
    melhor_arima = selecionar_arima(treino)

    melhor = {
        "tipo": "ARIMA",
        "ordem": melhor_arima["ordem"],
        "ordem_sazonal": (0, 0, 0, 0),
        "AIC": melhor_arima["AIC"],
        "ajuste": melhor_arima["ajuste"]
    }

    # 2. Verificar se a busca sazonal é permitida
    if frequencia <= 1:
        return melhor

    ciclos_minimos = 3 if frequencia == 52 else 2

    if len(treino) < ciclos_minimos * frequencia:
        return melhor

    # 3. Candidatos sazonais predefinidos
    configuracoes_sarima = [
        ((0, 1, 0), (1, 0, 0, frequencia)),
        ((1, 1, 0), (1, 0, 0, frequencia)),
        ((0, 1, 1), (0, 0, 1, frequencia)),
        ((1, 1, 1), (1, 0, 0, frequencia))
    ]

    # 4. Avaliar candidatos sazonais
    for ordem, ordem_sazonal in configuracoes_sarima:

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                modelo = ARIMA(
                    treino,
                    order=ordem,
                    seasonal_order=ordem_sazonal,
                    trend="t"
                )

                ajuste = modelo.fit()

            convergiu = ajuste.mle_retvals.get(
                "converged", False
            )

            if not convergiu or not np.isfinite(ajuste.aic):
                continue

            # Comparar ARIMA e SARIMA pelo AIC
            if ajuste.aic < melhor["AIC"]:

                melhor = {
                    "tipo": "SARIMA",
                    "ordem": ordem,
                    "ordem_sazonal": ordem_sazonal,
                    "AIC": ajuste.aic,
                    "ajuste": ajuste
                }

        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError
        ):
            continue

    return melhor


# ============================================================
# 29. TESTAR SELEÇÃO CONJUNTA NA SÉRIE 122
# ============================================================

id_teste = 122

grupo = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")
)

serie = grupo["valor"].to_numpy()
frequencia = int(grupo["frequency"].iloc[0])

# Primeira janela: sem os 10 valores de avaliação
treino = serie[:-K]

melhor = selecionar_arima_sarima(
    treino,
    frequencia
)

previsao = float(
    melhor["ajuste"].forecast(steps=1)[0]
)

print("\nSeleção conjunta ARIMA/SARIMA - Série 122")
print("Tipo selecionado:", melhor["tipo"])
print("Ordem:", melhor["ordem"])
print("Ordem sazonal:", melhor["ordem_sazonal"])
print("AIC:", melhor["AIC"])
print("Previsão:", previsao)
print("Valor real:", serie[-K])

# ============================================================
# 30. ROLLING-ORIGIN COM ARIMA/SARIMA AUTOMÁTICO
# ============================================================

def rolling_origin_arima_auto(serie, id_tsdl, frequencia, k=10):

    n = len(serie)
    resultados = []

    if n <= k:
        raise ValueError(
            f"Série {id_tsdl}: observações insuficientes."
        )

    for i in range(k):

        fim_treino = n - k + i

        # Dados disponíveis nesta origem
        treino = serie[:fim_treino]

        # Valor reservado para avaliação
        real = serie[fim_treino]

        # Selecionar ARIMA/SARIMA usando somente o treino
        melhor = selecionar_arima_sarima(
            treino,
            frequencia
        )

        # Previsão de um período à frente
        previsao = float(
            melhor["ajuste"].forecast(steps=1)[0]
        )

        if not np.isfinite(previsao):
            raise ValueError(
                f"Previsão inválida: série {id_tsdl}, "
                f"origem {i + 1}"
            )

        resultados.append({
            "id_tsdl": id_tsdl,
            "modelo": "ARIMA_AUTO",
            "tipo": melhor["tipo"],
            "ordem": str(melhor["ordem"]),
            "ordem_sazonal": str(melhor["ordem_sazonal"]),
            "origem": i + 1,
            "n_treino": len(treino),
            "real": real,
            "previsao": previsao,
            "erro": real - previsao,
            "erro_absoluto": abs(real - previsao),
            "AIC": melhor["AIC"]
        })

    return resultados


# ============================================================
# 31. CARREGAR OU EXECUTAR ARIMA/SARIMA
# ============================================================

from pathlib import Path

arquivo_arima = Path("resultados_arima_parcial.csv")

if arquivo_arima.exists():

    resultados_arima_auto = pd.read_csv(arquivo_arima)

    # Conferir se existem 10 previsões para cada uma das 16 séries
    contagem = resultados_arima_auto.groupby("id_tsdl").size()

    ids_esperados = set(dados["id_tsdl"].unique())

    completo = (
        len(resultados_arima_auto) == 160
        and set(contagem.index) == ids_esperados
        and (contagem == K).all()
        and not resultados_arima_auto[
            ["real", "previsao", "erro", "erro_absoluto"]
        ].isna().any().any()
    )

    if completo:
        print("\nResultados ARIMA carregados do CSV.")
        print("Nenhum modelo será recalculado.")

    else:
        raise ValueError(
            "O CSV do ARIMA está incompleto ou inválido. "
            "Verifique o arquivo antes de continuar."
        )

else:
    raise FileNotFoundError(
        "Arquivo resultados_arima_parcial.csv não encontrado."
    )

# ============================================================
# 32. VALIDAR RESULTADOS
# ============================================================

print(
    "\nTotal de previsões ARIMA/SARIMA:",
    len(resultados_arima_auto)
)

print("\nPrevisões por série:")
print(
    resultados_arima_auto
    .groupby("id_tsdl")
    .size()
)

print("\nTipos de modelo selecionados:")
print(
    pd.crosstab(
        resultados_arima_auto["id_tsdl"],
        resultados_arima_auto["tipo"]
    )
)

# ============================================================
# 33. CALCULAR MÉTRICAS DO ARIMA/SARIMA
# ============================================================

metricas_arima = (
    resultados_arima_auto
    .groupby(["id_tsdl", "modelo"])
    .agg(
        MAE=("erro_absoluto", "mean"),
        RMSE=("erro", lambda x: np.sqrt(np.mean(x**2)))
    )
    .reset_index()
)

# Mesma escala de MASE utilizada nos outros modelos
metricas_arima["escala_mase"] = (
    metricas_arima["id_tsdl"].map(escalas_mase)
)

metricas_arima["MASE"] = (
    metricas_arima["MAE"] /
    metricas_arima["escala_mase"]
)


# ============================================================
# 34. COMPARAR OS TRÊS MÉTODOS
# ============================================================

comparacao_estatisticos = pd.concat(
    [
        metricas_naive,
        metricas_ets_auto,
        metricas_arima
    ],
    ignore_index=True
)

comparacao_estatisticos = (
    comparacao_estatisticos
    .sort_values(["id_tsdl", "modelo"])
    .reset_index(drop=True)
)

print("\nComparação dos três métodos estatísticos:")

print(
    comparacao_estatisticos[
        ["id_tsdl", "modelo", "MAE", "RMSE", "MASE"]
    ].to_string(index=False)
)


# ============================================================
# 35. CONTAR VITÓRIAS POR MASE
# ============================================================

vencedores = comparacao_estatisticos.loc[
    comparacao_estatisticos
    .groupby("id_tsdl")["MASE"]
    .idxmin()
]

print("\nVitórias por modelo (menor MASE):")

print(
    vencedores["modelo"].value_counts()
)


# ============================================================
# 36. SALVAR RESULTADOS
# ============================================================

comparacao_estatisticos.to_csv(
    "metricas_estatisticos.csv",
    index=False
)

print("\nMétricas estatísticas salvas em CSV.")