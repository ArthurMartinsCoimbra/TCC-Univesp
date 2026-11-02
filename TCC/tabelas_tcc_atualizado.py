from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
SAIDA = BASE / 'tabelas'
SAIDA.mkdir(exist_ok=True)

MODELOS = {
    'ARIMA_AUTO': 'ARIMA/SARIMA',
    'ETS_AUTO': 'ETS',
    'Naive': 'Naive',
    'XGBoost': 'XGBoost',
    'RandomForest': 'Random Forest',
    'SVR': 'SVR',
}
FREQUENCIAS = {1: 'Anual', 4: 'Trimestral', 12: 'Mensal', 52: 'Semanal'}
FEATURES = {
    'trend': 'Tendência',
    'seasonal_strength': 'Sazonalidade',
    'x_acf1': 'Autocorrelação (lag 1)',
    'entropy': 'Entropia',
}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'savefig.dpi': 300})


def carregar(nome):
    caminho = BASE / nome
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo necessário não encontrado: {caminho}')
    return pd.read_csv(caminho, encoding='utf-8-sig')


def localizar_coluna(df, *alternativas):
    def normalizar(valor):
        s = unicodedata.normalize('NFKD', str(valor))
        return ''.join(c for c in s if not unicodedata.combining(c)).lower().replace(' ', '').replace('_', '')
    mapa = {normalizar(c): c for c in df.columns}
    for alt in alternativas:
        chave = normalizar(alt)
        if chave in mapa:
            return mapa[chave]
    raise KeyError(f'Coluna esperada: {alternativas}. Disponíveis: {list(df.columns)}')


def numeral(x, casas=3):
    if pd.isna(x):
        return '—'
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f'{float(x):.{casas}f}'.replace('.', ',')


def exportar(tabela, arquivo, titulo, nota='', casas=3):
    """Não sobrescreve qualquer arquivo que já exista."""
    caminhos = [SAIDA / f'{arquivo}.{ext}' for ext in ('csv', 'png', 'pdf')]
    if any(c.exists() for c in caminhos):
        print(f'[PULADA] {arquivo}: existe pelo menos um arquivo com esse nome. Nada foi sobrescrito.')
        return

    tabela.to_csv(caminhos[0], index=False, encoding='utf-8-sig', sep=';')
    exibicao = tabela.copy()
    for coluna in exibicao.columns:
        if pd.api.types.is_float_dtype(exibicao[coluna]):
            exibicao[coluna] = exibicao[coluna].map(lambda x: numeral(x, casas))

    n_linhas, n_colunas = exibicao.shape
    largura = min(17, max(9.5, 1.8 * n_colunas))
    altura = max(3.1, 1.25 + 0.39 * n_linhas)
    fig, ax = plt.subplots(figsize=(largura, altura))
    ax.axis('off')
    fig.text(0.5, 0.96, titulo, ha='center', va='top', fontsize=11, weight='bold')

    topo = 0.85
    rodape = 0.19 if nota else 0.13
    tab = ax.table(
        cellText=exibicao.astype(str).values.tolist(),
        colLabels=exibicao.columns.tolist(),
        cellLoc='center', colLoc='center',
        bbox=[0.015, rodape, 0.97, topo - rodape]
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.2 if n_colunas >= 6 else 9)
    for (i, j), cel in tab.get_celld().items():
        cel.set_facecolor('white')
        cel.set_edgecolor('black')
        cel.set_linewidth(0)
        if i == 0:
            cel.set_text_props(weight='bold')
            cel.visible_edges = 'TB'
            cel.set_linewidth(0.8)
        elif i == n_linhas:
            cel.visible_edges = 'B'
            cel.set_linewidth(0.7)
        else:
            cel.visible_edges = ''

    if nota:
        fig.text(0.035, 0.11, nota, ha='left', va='top', fontsize=8, wrap=True)
    fig.text(0.035, 0.045, 'Fonte: Elaboração própria (2026).', fontsize=8, ha='left')
    fig.savefig(caminhos[1], bbox_inches='tight', dpi=300)
    fig.savefig(caminhos[2], bbox_inches='tight')
    plt.close(fig)
    print(f'[CRIADA] {arquivo} (CSV, PNG e PDF)')


def preparar():
    met = carregar('metricas_consolidadas.csv')
    amostra = carregar('amostra_tcc.csv')
    familias = carregar('comparacao_familias.csv')
    feats = carregar('caracteristicas_treino.csv')

    # Padronização apenas de nomes conhecidos, mantendo os valores originais.
    for df in (met, amostra, familias, feats):
        id_col = localizar_coluna(df, 'id_tsdl')
        df.rename(columns={id_col: 'id_tsdl'}, inplace=True)
        df['id_tsdl'] = pd.to_numeric(df['id_tsdl'], errors='raise').astype(int)

    met.rename(columns={
        localizar_coluna(met, 'modelo'): 'modelo',
        localizar_coluna(met, 'MAE'): 'MAE',
        localizar_coluna(met, 'RMSE'): 'RMSE',
        localizar_coluna(met, 'MASE'): 'MASE',
    }, inplace=True)
    amostra.rename(columns={
        localizar_coluna(amostra, 'subject'): 'subject',
        localizar_coluna(amostra, 'frequency'): 'frequency',
        localizar_coluna(amostra, 'indice'): 'indice',
        localizar_coluna(amostra, 'valor'): 'valor',
    }, inplace=True)
    familias.rename(columns={
        localizar_coluna(familias, 'razao_ml_estatistico'): 'razao_ml_estatistico',
    }, inplace=True)
    for ch in FEATURES:
        feats.rename(columns={localizar_coluna(feats, ch): ch}, inplace=True)

    if len(met) != 96 or met.duplicated(['id_tsdl', 'modelo']).any():
        raise ValueError('metricas_consolidadas.csv deve conter 96 pares únicos (16 séries x 6 modelos).')
    if set(met['modelo']) != set(MODELOS):
        raise ValueError(f'Modelos inesperados: {sorted(set(met["modelo"]))}')
    for df, nome in ((amostra, 'amostra'), (familias, 'famílias'), (feats, 'características')):
        if df['id_tsdl'].nunique() != 16:
            raise ValueError(f'{nome}: esperadas 16 séries distintas.')
    if amostra[['id_tsdl', 'indice']].duplicated().any():
        raise ValueError('Há índices duplicados por série em amostra_tcc.csv')
    for col in ('MAE', 'RMSE', 'MASE'):
        met[col] = pd.to_numeric(met[col], errors='raise')
    if met[['MAE', 'RMSE', 'MASE']].isna().any().any():
        raise ValueError('Métricas com valores ausentes.')
    return met, amostra, familias, feats


def tabela_amostra(amostra):
    df = (amostra.groupby('id_tsdl', as_index=False)
          .agg(Área=('subject', 'first'), Frequência=('frequency', 'first'),
               Observações=('valor', 'size')))
    df['Frequência'] = pd.to_numeric(df['Frequência'], errors='raise').astype(int).map(FREQUENCIAS)
    if df['Frequência'].isna().any():
        raise ValueError('Há frequências desconhecidas na amostra.')
    df = df.rename(columns={'id_tsdl': 'ID TSDL'}).sort_values('ID TSDL')
    exportar(df, 'tabela_1_caracterizacao_amostra',
             'Caracterização das 16 séries temporais',
             'Frequências: anual (1), trimestral (4), mensal (12) e semanal (52).')


def tabela_desempenho(met):
    df = (met.groupby('modelo', as_index=False)
          .agg(**{'MAE médio': ('MAE', 'mean'), 'RMSE médio': ('RMSE', 'mean'),
                  'MASE médio': ('MASE', 'mean'), 'MASE mediano': ('MASE', 'median')}))
    rank = met.copy()
    rank['posição'] = rank.groupby('id_tsdl')['MASE'].rank(method='average', ascending=True)
    pos = rank.groupby('modelo')['posição'].mean().rename('Ranking médio')
    df = df.join(pos, on='modelo').sort_values('MASE médio')
    df['modelo'] = df['modelo'].map(MODELOS)
    df = df.rename(columns={'modelo': 'Método'})
    exportar(df, 'tabela_2_desempenho_geral',
             'Indicadores agregados dos métodos de previsão',
             'MAE e RMSE médios são descritivos: não comparar magnitudes entre séries de escalas diferentes.')


def tabela_familias(familias, amostra):
    metadados = amostra.groupby('id_tsdl', as_index=False).agg(Área=('subject', 'first'))
    df = familias[['id_tsdl', 'razao_ml_estatistico']].copy().merge(metadados, on='id_tsdl', validate='one_to_one')
    df['Vencedora'] = np.select(
        [df['razao_ml_estatistico'] > 1, df['razao_ml_estatistico'] < 1],
        ['Estatísticos', 'Machine Learning'], default='Empate'
    )
    df = df.rename(columns={'id_tsdl': 'ID TSDL', 'razao_ml_estatistico': 'Razão ML/Est.'})
    df = df[['ID TSDL', 'Área', 'Razão ML/Est.', 'Vencedora']].sort_values('ID TSDL')
    exportar(df, 'tabela_3_comparacao_familias',
             'Comparação retrospectiva entre famílias por série',
             'Razão = menor MAE entre modelos de ML / menor MAE entre modelos estatísticos; valores acima de 1 favorecem estatísticos.')


def tabela_correlacoes(met, feats):
    referencia = met.loc[met['modelo'].eq('Naive'), ['id_tsdl', 'MAE']].rename(columns={'MAE': 'MAE_Naive'})
    base = met.merge(referencia, on='id_tsdl', validate='many_to_one')
    base = base.loc[~base['modelo'].eq('Naive')].copy()
    if (base['MAE_Naive'] <= 0).any():
        raise ValueError('Não é possível calcular razões com MAE do Naive menor ou igual a zero.')
    base['R'] = base['MAE'] / base['MAE_Naive']
    base = base.merge(feats[['id_tsdl', *FEATURES.keys()]], on='id_tsdl', validate='many_to_one')
    linhas = []
    for modelo in ('ARIMA_AUTO', 'ETS_AUTO', 'RandomForest', 'XGBoost', 'SVR'):
        grupo = base.loc[base['modelo'].eq(modelo)]
        for feat, nome in FEATURES.items():
            pares = grupo[[feat, 'R']].copy()
            pares[feat] = pd.to_numeric(pares[feat], errors='coerce')
            pares = pares.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pares) < 3 or pares[feat].nunique() < 2 or pares['R'].nunique() < 2:
                raise ValueError(f'Não é possível calcular correlação válida: {modelo}, {feat}')
            rho, p = spearmanr(pares[feat], pares['R'])
            linhas.append({'Característica': nome, 'Modelo / Naive': MODELOS[modelo],
                           'n': len(pares), 'ρ Spearman': rho, 'Valor-p': p})
    df = pd.DataFrame(linhas)
    exportar(df, 'tabela_4_correlacoes_completas',
             'Correlações exploratórias com desempenho relativo ao Naive',
             'Características calculadas no treino inicial. São 20 comparações exploratórias; valores-p não ajustados.', casas=4)


def tabela_mase_sazonal(met, amostra):
    # Reconstitui a escala sazonal exclusivamente do treino inicial (n - 10 observações).
    denominadores = []
    for serie_id, grupo in amostra.groupby('id_tsdl'):
        grupo = grupo.sort_values('indice')
        y = pd.to_numeric(grupo['valor'], errors='raise').to_numpy(dtype=float)
        freq = int(pd.to_numeric(grupo['frequency']).iloc[0])
        treino = y[:-10]
        if len(treino) <= freq:
            raise ValueError(f'Série {serie_id}: treino insuficiente para defasagem {freq}.')
        q = np.mean(np.abs(treino[freq:] - treino[:-freq]))
        if not np.isfinite(q) or q <= 0:
            raise ValueError(f'Série {serie_id}: escala sazonal nula ou inválida.')
        denominadores.append({'id_tsdl': serie_id, 'escala_sazonal': q})
    calc = met.merge(pd.DataFrame(denominadores), on='id_tsdl', validate='many_to_one')
    calc['MASE sazonal'] = calc['MAE'] / calc['escala_sazonal']
    df = (calc.groupby('modelo', as_index=False)
          .agg(**{'MASE original': ('MASE', 'mean'), 'MASE sazonal': ('MASE sazonal', 'mean')}))
    df = df.sort_values('MASE original')
    df['modelo'] = df['modelo'].map(MODELOS)
    df = df.rename(columns={'modelo': 'Método'})
    exportar(df, 'tabela_5_robustez_mase',
             'Sensibilidade do MASE à escala de normalização',
             'Ambos usam escalas calculadas no treino inicial. MASE sazonal usa defasagens 1, 4, 12 e 52.')


def main():
    print('Gerando somente as cinco tabelas complementares.')
    print('A tabela_correlacoes_tcc existente não será acessada nem modificada.\n')
    met, amostra, familias, feats = preparar()
    tarefas = [
        ('Caracterização da amostra', lambda: tabela_amostra(amostra)),
        ('Desempenho geral', lambda: tabela_desempenho(met)),
        ('Comparação das famílias', lambda: tabela_familias(familias, amostra)),
        ('Correlações completas', lambda: tabela_correlacoes(met, feats)),
        ('Robustez do MASE', lambda: tabela_mase_sazonal(met, amostra)),
    ]
    for nome, func in tarefas:
        try:
            func()
        except Exception as exc:
            print(f'[ERRO] {nome}: {type(exc).__name__}: {exc}')
    print('\nConcluído. Confira a pasta:', SAIDA)


if __name__ == '__main__':
    main()
