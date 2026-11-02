# ============================================================
# 1. IMPORTAÇÕES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 2. CONFIGURAÇÕES DO EXPERIMENTO
# ============================================================

K = 10
H = 1

CAMINHO_PROJETO = Path(__file__).resolve().parent

dados = pd.read_csv(
    CAMINHO_PROJETO / "amostra_tcc.csv"
)

print("Dimensões dos dados:", dados.shape)

print(
    "Número de séries:",
    dados["id_tsdl"].nunique()
)


# ============================================================
# 3. DEFINIR LAGS CONFORME A FREQUÊNCIA
# ============================================================

def definir_lags(frequencia):

    lags = [1, 2, 3]

    if frequencia > 1:
        lags.append(int(frequencia))

    return sorted(set(lags))


# ============================================================
# 4. TRANSFORMAR SÉRIE EM DADOS SUPERVISIONADOS
# ============================================================

def criar_dados_supervisionados(serie, lags):

    X = []
    y = []

    maior_lag = max(lags)

    for t in range(maior_lag, len(serie)):

        # Variáveis explicativas: valores anteriores
        entrada = [
            serie[t - lag]
            for lag in lags
        ]

        # Variável alvo: valor no instante t
        alvo = serie[t]

        X.append(entrada)
        y.append(alvo)

    X = np.array(X)
    y = np.array(y)

    return X, y


# ============================================================
# 5. TESTAR NA PRIMEIRA ORIGEM DA SÉRIE 639
# ============================================================

id_teste = 639

grupo = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")
)

serie = grupo["valor"].to_numpy()

frequencia = int(
    grupo["frequency"].iloc[0]
)

lags = definir_lags(frequencia)

# Primeira origem do rolling-origin
fim_treino = len(serie) - K

treino = serie[:fim_treino]

# Criar exemplos somente a partir do treinamento
X_treino, y_treino = criar_dados_supervisionados(
    treino,
    lags
)

# Preparar as variáveis para prever o próximo valor
X_previsao = np.array([
    [treino[-lag] for lag in lags]
])

# Valor real reservado para avaliação
valor_real = serie[fim_treino]


# ============================================================
# 6. VALIDAR A PREPARAÇÃO DOS DADOS
# ============================================================

print("\nPreparação dos dados - Série 639")

print("Frequência:", frequencia)
print("Lags:", lags)

print("Observações na série:", len(serie))
print("Observações no treino:", len(treino))

print("Dimensão de X_treino:", X_treino.shape)
print("Dimensão de y_treino:", y_treino.shape)

print("\nPrimeiro exemplo supervisionado:")
print("Entradas:", X_treino[0])
print("Alvo:", y_treino[0])

print("\nDados para próxima previsão:")
print("Entradas:", X_previsao)
print("Valor real reservado:", valor_real)

# ============================================================
# 7. ROLLING-ORIGIN - RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestRegressor


def rolling_origin_rf(serie, id_tsdl, frequencia, k=10):

    n = len(serie)
    lags = definir_lags(frequencia)

    resultados = []

    for i in range(k):

        fim_treino = n - k + i

        # Dados disponíveis nesta origem
        treino = serie[:fim_treino]

        # Valor reservado para avaliação
        real = serie[fim_treino]

        # Transformação supervisionada
        X_treino, y_treino = criar_dados_supervisionados(
            treino,
            lags
        )

        # Entradas para prever o próximo período
        X_previsao = np.array([
            [treino[-lag] for lag in lags]
        ])

        # Modelo com parâmetros fixos
        modelo = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Ajustar exclusivamente com o treinamento
        modelo.fit(X_treino, y_treino)

        # Prever h=1
        previsao = float(
            modelo.predict(X_previsao)[0]
        )

        resultados.append({
            "id_tsdl": id_tsdl,
            "modelo": "RandomForest",
            "origem": i + 1,
            "n_treino": len(treino),
            "real": real,
            "previsao": previsao,
            "erro": real - previsao,
            "erro_absoluto": abs(real - previsao)
        })

    return resultados


# ============================================================
# 8. TESTAR RANDOM FOREST NA SÉRIE 639
# ============================================================

resultados_teste_rf = rolling_origin_rf(
    serie,
    id_teste,
    frequencia,
    k=K
)

tabela_teste_rf = pd.DataFrame(resultados_teste_rf)

print("\nRolling-origin - Random Forest - Série 639:")
print(tabela_teste_rf.to_string(index=False))

# ============================================================
# 9. EXECUTAR RANDOM FOREST NAS 16 SÉRIES
#    COM REAPROVEITAMENTO DO CSV
# ============================================================

ARQUIVO_RF = CAMINHO_PROJETO / "resultados_random_forest.csv"

if ARQUIVO_RF.exists():

    resultados_rf = pd.read_csv(ARQUIVO_RF)

    contagem = resultados_rf.groupby("id_tsdl").size()

    ids_esperados = set(dados["id_tsdl"].unique())

    completo = (
        len(resultados_rf) == 16 * K
        and set(contagem.index) == ids_esperados
        and (contagem == K).all()
        and resultados_rf.groupby("id_tsdl")["origem"].nunique().eq(K).all()
        and np.isfinite(
            resultados_rf[
                ["real", "previsao", "erro", "erro_absoluto"]
            ].to_numpy()
        ).all()
    )

    if not completo:
        raise ValueError(
            "O CSV do Random Forest está incompleto ou inválido."
        )

    print("\nResultados Random Forest carregados do CSV.")
    print("Nenhum treinamento foi repetido.")

else:

    todos_resultados_rf = []

    for id_tsdl, grupo in dados.groupby("id_tsdl"):

        grupo = grupo.sort_values("indice")

        serie_atual = grupo["valor"].to_numpy()
        frequencia_atual = int(grupo["frequency"].iloc[0])

        print(f"Processando série {id_tsdl}...", flush=True)

        resultados_serie = rolling_origin_rf(
            serie_atual,
            id_tsdl,
            frequencia_atual,
            k=K
        )

        todos_resultados_rf.extend(resultados_serie)

        # Salvar o progresso após cada série
        pd.DataFrame(todos_resultados_rf).to_csv(
            CAMINHO_PROJETO / "resultados_random_forest_parcial.csv",
            index=False
        )

    resultados_rf = pd.DataFrame(todos_resultados_rf)

    resultados_rf.to_csv(
        ARQUIVO_RF,
        index=False
    )

    print("\nRandom Forest concluído e salvo em CSV.")


# ============================================================
# 10. VALIDAR RESULTADOS
# ============================================================

print("\nTotal de previsões Random Forest:", len(resultados_rf))

print("\nPrevisões por série:")
print(
    resultados_rf.groupby("id_tsdl").size()
)

# ============================================================
# 11. CALCULAR MÉTRICAS DO RANDOM FOREST
# ============================================================

metricas_rf = (
    resultados_rf
    .groupby(["id_tsdl", "modelo"])
    .agg(
        MAE=("erro_absoluto", "mean"),
        RMSE=("erro", lambda x: np.sqrt(np.mean(x**2)))
    )
    .reset_index()
)

# Calcular a escala MASE usando somente o treino inicial
escalas_mase = {}

for id_tsdl, grupo in dados.groupby("id_tsdl"):

    serie_atual = (
        grupo.sort_values("indice")["valor"].to_numpy()
    )

    treino_inicial = serie_atual[:-K]

    escala = np.mean(np.abs(np.diff(treino_inicial)))

    if not np.isfinite(escala) or escala <= 0:
        raise ValueError(
            f"Escala MASE inválida para a série {id_tsdl}"
        )

    escalas_mase[id_tsdl] = escala

metricas_rf["escala_mase"] = (
    metricas_rf["id_tsdl"].map(escalas_mase)
)

metricas_rf["MASE"] = (
    metricas_rf["MAE"] / metricas_rf["escala_mase"]
)

print("\nMétricas do Random Forest:")
print(metricas_rf.to_string(index=False))


# ============================================================
# 12. SALVAR MÉTRICAS
# ============================================================

metricas_rf.to_csv(
    CAMINHO_PROJETO / "metricas_random_forest.csv",
    index=False
)

print("\nMétricas do Random Forest salvas.")

# ============================================================
# 13. ROLLING-ORIGIN - XGBOOST
# ============================================================

from xgboost import XGBRegressor


def rolling_origin_xgb(serie, id_tsdl, frequencia, k=10):

    n = len(serie)
    lags = definir_lags(frequencia)

    resultados = []

    if n - k <= max(lags):
        raise ValueError(
            f"Série {id_tsdl}: treinamento insuficiente."
        )

    for i in range(k):

        fim_treino = n - k + i

        treino = serie[:fim_treino]
        real = serie[fim_treino]

        # Dados supervisionados somente com o treinamento
        X_treino, y_treino = criar_dados_supervisionados(
            treino, lags
        )

        X_previsao = np.array([
            [treino[-lag] for lag in lags]
        ])

        # Configuração fixa para todas as séries
        modelo = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )

        modelo.fit(X_treino, y_treino)

        previsao = float(
            modelo.predict(X_previsao)[0]
        )

        if not np.isfinite(previsao):
            raise ValueError(
                f"Previsão inválida: série {id_tsdl}, "
                f"origem {i + 1}"
            )

        resultados.append({
            "id_tsdl": id_tsdl,
            "modelo": "XGBoost",
            "origem": i + 1,
            "n_treino": len(treino),
            "real": real,
            "previsao": previsao,
            "erro": real - previsao,
            "erro_absoluto": abs(real - previsao)
        })

    return resultados


# ============================================================
# 14. TESTAR XGBOOST NA SÉRIE 639
# ============================================================

id_teste = 639

grupo = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")
)

serie_teste = grupo["valor"].to_numpy()
frequencia_teste = int(grupo["frequency"].iloc[0])

resultados_teste_xgb = rolling_origin_xgb(
    serie_teste,
    id_teste,
    frequencia_teste,
    k=K
)

print("\nRolling-origin - XGBoost - Série 639:")
print(
    pd.DataFrame(resultados_teste_xgb)
    .to_string(index=False)
)

# ============================================================
# 15. EXECUTAR XGBOOST NAS 16 SÉRIES
#     COM REAPROVEITAMENTO DO CSV
# ============================================================

ARQUIVO_XGB = CAMINHO_PROJETO / "resultados_xgboost.csv"

if ARQUIVO_XGB.exists():

    resultados_xgb = pd.read_csv(ARQUIVO_XGB)

    # Validar a integridade dos resultados
    contagem = resultados_xgb.groupby("id_tsdl").size()

    ids_esperados = set(dados["id_tsdl"].unique())

    completo = (
        len(resultados_xgb) == len(ids_esperados) * K
        and set(contagem.index) == ids_esperados
        and (contagem == K).all()
        and resultados_xgb.groupby("id_tsdl")["origem"]
            .nunique().eq(K).all()
        and np.isfinite(
            resultados_xgb[
                ["real", "previsao", "erro", "erro_absoluto"]
            ].to_numpy()
        ).all()
    )

    if not completo:
        raise ValueError(
            "O CSV do XGBoost está incompleto ou inválido."
        )

    print("\nResultados XGBoost carregados do CSV.")
    print("Nenhum treinamento foi repetido.")

else:

    todos_resultados_xgb = []

    for id_tsdl, grupo in dados.groupby("id_tsdl"):

        grupo = grupo.sort_values("indice")

        serie_atual = grupo["valor"].to_numpy()

        frequencia_atual = int(
            grupo["frequency"].iloc[0]
        )

        print(
            f"Processando série {id_tsdl}...",
            flush=True
        )

        resultados_serie = rolling_origin_xgb(
            serie_atual,
            id_tsdl,
            frequencia_atual,
            k=K
        )

        todos_resultados_xgb.extend(
            resultados_serie
        )

        # Salvar progresso após cada série
        pd.DataFrame(
            todos_resultados_xgb
        ).to_csv(
            CAMINHO_PROJETO / "resultados_xgboost_parcial.csv",
            index=False
        )

        print(
            f"Série {id_tsdl} concluída.",
            flush=True
        )

    resultados_xgb = pd.DataFrame(
        todos_resultados_xgb
    )

    # Salvar resultados completos
    resultados_xgb.to_csv(
        ARQUIVO_XGB,
        index=False
    )

    print("\nXGBoost concluído e salvo em CSV.")


# ============================================================
# 16. VALIDAR RESULTADOS
# ============================================================

print(
    "\nTotal de previsões XGBoost:",
    len(resultados_xgb)
)

print("\nPrevisões por série:")

print(
    resultados_xgb.groupby("id_tsdl").size()
)

# ============================================================
# 17. CALCULAR MÉTRICAS DO XGBOOST
# ============================================================

metricas_xgb = (
    resultados_xgb
    .groupby(["id_tsdl", "modelo"])
    .agg(
        MAE=("erro_absoluto", "mean"),
        RMSE=("erro", lambda x: np.sqrt(np.mean(x**2)))
    )
    .reset_index()
)

# Reutilizar as escalas MASE já calculadas
metricas_xgb["escala_mase"] = (
    metricas_xgb["id_tsdl"].map(escalas_mase)
)

# Calcular MASE
metricas_xgb["MASE"] = (
    metricas_xgb["MAE"] /
    metricas_xgb["escala_mase"]
)

print("\nMétricas do XGBoost:")
print(metricas_xgb.to_string(index=False))


# ============================================================
# 18. SALVAR MÉTRICAS
# ============================================================

metricas_xgb.to_csv(
    CAMINHO_PROJETO / "metricas_xgboost.csv",
    index=False
)

print("\nMétricas do XGBoost salvas em CSV.")

# ============================================================
# 19. ROLLING-ORIGIN - SVR
# ============================================================

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor


def rolling_origin_svr(serie, id_tsdl, frequencia, k=10):

    n = len(serie)
    lags = definir_lags(frequencia)

    resultados = []

    if n - k <= max(lags):
        raise ValueError(
            f"Série {id_tsdl}: treinamento insuficiente."
        )

    for i in range(k):

        fim_treino = n - k + i

        # Janela de treinamento disponível
        treino = serie[:fim_treino]

        # Observação reservada para avaliação
        real = serie[fim_treino]

        # Criar dados supervisionados
        X_treino, y_treino = criar_dados_supervisionados(
            treino, lags
        )

        # Entradas para prever o próximo período
        X_previsao = np.array([
            [treino[-lag] for lag in lags]
        ])

        # Padronização das entradas + SVR
        pipeline = make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
                gamma="scale"
            )
        )

        # Padronização independente do alvo
        modelo = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=StandardScaler()
        )

        # Ajustar somente com os dados disponíveis
        modelo.fit(X_treino, y_treino)

        # Previsão convertida para a escala original
        previsao = float(
            modelo.predict(X_previsao)[0]
        )

        if not np.isfinite(previsao):
            raise ValueError(
                f"Previsão inválida: série {id_tsdl}, "
                f"origem {i + 1}"
            )

        resultados.append({
            "id_tsdl": id_tsdl,
            "modelo": "SVR",
            "origem": i + 1,
            "n_treino": len(treino),
            "real": real,
            "previsao": previsao,
            "erro": real - previsao,
            "erro_absoluto": abs(real - previsao)
        })

    return resultados


# ============================================================
# 20. TESTAR SVR NA SÉRIE 639
# ============================================================

id_teste = 639

grupo = (
    dados[dados["id_tsdl"] == id_teste]
    .sort_values("indice")
)

serie_teste = grupo["valor"].to_numpy()

frequencia_teste = int(
    grupo["frequency"].iloc[0]
)

resultados_teste_svr = rolling_origin_svr(
    serie_teste,
    id_teste,
    frequencia_teste,
    k=K
)

print("\nRolling-origin - SVR - Série 639:")

print(
    pd.DataFrame(resultados_teste_svr)
    .to_string(index=False)
)

# ============================================================
# 21. EXECUTAR SVR NAS 16 SÉRIES
#     COM REAPROVEITAMENTO DO CSV
# ============================================================

ARQUIVO_SVR = CAMINHO_PROJETO / "resultados_svr.csv"

if ARQUIVO_SVR.exists():

    resultados_svr = pd.read_csv(ARQUIVO_SVR)

    # Validar a integridade do CSV
    contagem = resultados_svr.groupby("id_tsdl").size()

    ids_esperados = set(dados["id_tsdl"].unique())

    completo = (
        len(resultados_svr) == len(ids_esperados) * K
        and set(contagem.index) == ids_esperados
        and (contagem == K).all()
        and resultados_svr.groupby("id_tsdl")["origem"]
            .nunique().eq(K).all()
        and np.isfinite(
            resultados_svr[
                ["real", "previsao", "erro", "erro_absoluto"]
            ].to_numpy()
        ).all()
    )

    if not completo:
        raise ValueError(
            "O CSV do SVR está incompleto ou inválido."
        )

    print("\nResultados SVR carregados do CSV.")
    print("Nenhum treinamento foi repetido.")

else:

    todos_resultados_svr = []

    for id_tsdl, grupo in dados.groupby("id_tsdl"):

        grupo = grupo.sort_values("indice")

        serie_atual = grupo["valor"].to_numpy()

        frequencia_atual = int(
            grupo["frequency"].iloc[0]
        )

        print(
            f"Processando série {id_tsdl}...",
            flush=True
        )

        resultados_serie = rolling_origin_svr(
            serie_atual,
            id_tsdl,
            frequencia_atual,
            k=K
        )

        todos_resultados_svr.extend(resultados_serie)

        # Salvar progresso após cada série
        pd.DataFrame(
            todos_resultados_svr
        ).to_csv(
            CAMINHO_PROJETO / "resultados_svr_parcial.csv",
            index=False
        )

        print(
            f"Série {id_tsdl} concluída.",
            flush=True
        )

    resultados_svr = pd.DataFrame(
        todos_resultados_svr
    )

    # Salvar arquivo completo
    resultados_svr.to_csv(
        ARQUIVO_SVR,
        index=False
    )

    print("\nSVR concluído e salvo em CSV.")


# ============================================================
# 22. VALIDAR RESULTADOS
# ============================================================

print(
    "\nTotal de previsões SVR:",
    len(resultados_svr)
)

print("\nPrevisões por série:")

print(
    resultados_svr.groupby("id_tsdl").size()
)

# ============================================================
# 23. CALCULAR MÉTRICAS DO SVR
# ============================================================

metricas_svr = (
    resultados_svr
    .groupby(["id_tsdl", "modelo"])
    .agg(
        MAE=("erro_absoluto", "mean"),
        RMSE=("erro", lambda x: np.sqrt(np.mean(x**2)))
    )
    .reset_index()
)

# Reutilizar as escalas históricas do MASE
metricas_svr["escala_mase"] = (
    metricas_svr["id_tsdl"].map(escalas_mase)
)

metricas_svr["MASE"] = (
    metricas_svr["MAE"] /
    metricas_svr["escala_mase"]
)

print("\nMétricas do SVR:")
print(metricas_svr.to_string(index=False))


# ============================================================
# 24. SALVAR MÉTRICAS
# ============================================================

metricas_svr.to_csv(
    CAMINHO_PROJETO / "metricas_svr.csv",
    index=False
)

print("\nMétricas do SVR salvas em CSV.")