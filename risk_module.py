# risk_module

# atenção: este módulo assume que os arquivos carregados estão na mesma pasta que o Jupyter Notebook e este arquivo

import pandas as pd
import scipy.stats 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# carregamento de dados e pré-processamento

def get_data_year(year:int)->pd.DataFrame:
    """
    Função que carrega os dados disponíveis em csv do ano dado e retorna um DataFrame com uma coluna extra VOLUME
    que é a quantidade de ações negociadas no dia multiplicada pelo preço, isto é, o volume de negociação no dia
    """
    # carrega o csv do ano dado
    df = pd.read_csv('./ACOES_ORDINARIAS_'+str(year)+'.csv', parse_dates=['DATA_DO_PREGAO'])
    # cria a coluna volume
    df['VOLUME'] = df['PRECO_MEDIO']*df['QUANTIDADE_NEGOCIADA']
    return df

def get_series(cod_neg:str, col:str, data_b3:pd.DataFrame)->pd.Series:
    """
    Função que recebe um código de negociação, uma string com o nome da coluna desejada e o dataframe 
    de dados gerado por get_data e retorna uma série temporal da coluna escolhida, nomeada com o código de 
    negociação e indexada pela data do pregão
    """
    # seleciona somente os papéis desejados
    mask = (data_b3['CODIGO_DE_NEGOCIACAO'] == cod_neg)
    # cria série com papéis selecionados e indexa pelo tempo
    series = data_b3[mask][['DATA_DO_PREGAO', col]]
    series.index = series['DATA_DO_PREGAO']
    series = series[col]
    # nomeia a série
    series.name = cod_neg
    return series

def get_df(data:pd.DataFrame, col:str, papers = None):
    """
    Função que recebe um Dataframe gerado por get_data, uma string com o nome da coluna desejada
    e uma lista de ações desejadas e retorna um DataFrame com todos as ações desejadas e somente
    a coluna escolhida para cada ação indexada pela data do pregão. No caso de não serem dadas
    ações utiliza-se as dadas pelo DataFrame em si aonde deletamos ações não negociadas todos os dias
    """
    # seleciona ações caso não sejam dadas
    if papers is None:
        papers = data.CODIGO_DE_NEGOCIACAO.unique()
    # adiciona ações em um dataframe só indexado pela data do pregão
    df = pd.concat([get_series(asset, col, data) for asset in papers], axis=1)
    return df.dropna(axis='columns') # remove ações que não foram negociadas em algum período

def get_risk_free():
    """
    Função que carrega o DataFrame com os retornos diários percentuais do CDI
    """
    # carrega o DataFrame dos retornos para extrair o index
    rets_5y_ = pd.read_csv('./prices_5y_b3.csv', index_col=0).pct_change().fillna(0)
    # carrega o CDI, nomeia a Series e converte de % para absoluto
    df_ = pd.read_csv('./cdi_pd_percent.csv', sep=';', index_col=0, parse_dates=['Date'], dayfirst=True, skipfooter=1, engine='python')
    df_.columns = ['CDI']
    df_ = df_.loc[:, 'CDI']/100
    # calcula o retorno acumulado do CDI, reindexa a Series e retorna os retornos diários
    return compound(df_)[rets_5y_.index].pct_change().fillna(0)

# análise

def compound(r):
    """
    Função que recebe uma Series ou DataFrame de retornos e retorna a evolução do retorno acumulado
    """
    return (1+r).cumprod()

def drawdown(return_series: pd.Series, initial_wealth:float=1) -> pd.DataFrame:
    """ Recebe uma Series de retornos e o patrimônio inicial e retorna um DataFrame que contém:
        - Patrimônio acumulado
        - Picos anteriores 
        - Drawdowns percentuais
    """
    # patrimônio acumulado
    wealth_index = initial_wealth*compound(return_series)
    # picos anteriores
    previous_peaks = wealth_index.cummax()
    # DD percentuais
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({'Wealth': wealth_index, 'Peaks': previous_peaks, 'Drawdowns': drawdowns})

def annualized_vol(r:pd.Series, periods_per_year:int)->float:
    """
    Anualiza a volatilidade de uma série de retornos
    """
    return r.std()*(periods_per_year**0.5)

def annualized_rets(r:pd.Series, periods_per_year:int)->float:
    """
    Anualiza os retornos de uma série de retornos
    """
    compunded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compunded_growth**(periods_per_year/n_periods) - 1

def kurtosis(r:pd.Series)->float:
    """ 
    Calcula a curtose de uma série de retornos
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    return (demeaned_r**4).mean()/sigma_r**4

def skewness(r:pd.Series)->float:
    """ 
    Calcula a assimetria de uma série de retornos
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    return (demeaned_r**3).mean()/sigma_r**3

def sharpe_ratio(r:pd.Series, riskfree_rate, periods_per_year:float)->float:
    """
    Calcula o índice de Sharpe de uma série de retornos
    """
    excess_ret = r - riskfree_rate
    ann_ex_ret = annualized_rets(excess_ret, periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def var_historic(r, p:float=5):
    """
    Retorna o VaR histórico de uma série de retornos dado uma confiabilidade p
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, p=p)
    elif isinstance(r, pd.Series):
        return np.percentile(r, p)
    else:
        raise TypeError('Os retornos r deve ser uma Series ou DataFrame')
    
def var_gaussian(r, p:float=5, modified:bool=False):
    """
    Retorna o VaR gaussiano ou de Cornish-Fisher (se modified=True) de uma série de retornos dado uma confiabilidade p
    """
    zalpha = scipy.stats.norm.ppf(p/100)
    if modified:
        S, K = skewness(r), kurtosis(r)
        zalphat = zalpha + (zalpha**2-1)*S/6 + (zalpha**3-3*zalpha)*(K-3)/24-(2*zalpha**3-5*zalpha)*S**2/36
        return (r.mean()+zalphat*r.std(ddof=0))
    else:
        return (r.mean()+zalpha*r.std(ddof=0))
    
def cvar(r, var_estimator, p:float=5, **kwargs):
    """
    Retorna o CVaR dado um método de estimar o VaR (var_estimator) de uma série de retornos dado uma confiabilidade p
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar, var_estimator=var_estimator, p=p, **kwargs)
    elif isinstance(r, pd.Series):
        var = var_estimator(r, p, **kwargs)
        return r[r<=var].mean()
    else:
        raise TypeError('Os retornos r deve ser uma Series ou DataFrame')
    
def summary_stats(r, periods_per_year:float, riskfree_rate, p:float=5):
    """
    Retorna um DataFrame que contém um sumário agregado com o retorno e volatilidade anualizados, SR, drawdown máximo, assimetria, curtose, VaR e CVaR históricos das
    colunas de dos retornos r
    """
    if isinstance(r, pd.Series):
        r = pd.DataFrame(r, columns=[r.name])
    ann_r = r.aggregate(annualized_rets, periods_per_year =periods_per_year)*100
    ann_vol = r.aggregate(annualized_vol, periods_per_year =periods_per_year)*100
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate = riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(lambda r: drawdown(r)['Drawdowns'].min())*100
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    hist_var = r.aggregate(var_historic, p=p)*100
    cvar_hist = r.aggregate(cvar, p=p, var_estimator=var_historic)*100

    return pd.DataFrame({'Retorno Anualizado (%)': ann_r.round(2), 
                         'Volatilidade Anualizada (%)': ann_vol.round(2), 
                         'SR Anualizado': ann_sr.round(2),
                         'Drawdown Máximo (%)': dd.round(2),
                         'Assimetria': skew.round(2),
                         'Curtose': kurt.round(2),
                         'VaR Histórico (%)': hist_var.round(2),
                         'CVaR Histórico (%)': cvar_hist.round(2)}
                         )

def portfolio_return(weights, returns):
    """
    Dados os pesos e os retornos calcula o retorno do portfólio
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Dados os pesos e a matriz de covariância calcula a volatilidade do portfólio
    """
    return (weights.T @ covmat @ weights)**0.5

def ew(rets):
    """
    Calcula os pesos de um portfólio EW
    """
    n=rets.shape[0]
    return np.repeat(1/n, n)

def msr(riskfree_rate, er, cov):
    """
    Dados o retorno anualizado do ativo sem risco, o vetor de retornos e a matriz de covariância
    retorna os pesos do portfólio MSR
    """
    # dimensão
    n = er.shape[0]
    # ponto inicial da otimização
    init_guess = np.repeat(1/n, n)
    # limites para os pesos
    bounds = ((0.0, 1.0),)*n
    # constraint para a soma de pesos ser 1
    weights_sum_to_1 = {'type': 'eq', 
                        'fun': lambda weights: np.sum(weights)-1}
    # definição de SR negativo
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
    # minimizar SR negativo = maximizar SR
    results = minimize(neg_sharpe_ratio, init_guess, args=(riskfree_rate, er, cov,), method='SLSQP', constraints=(weights_sum_to_1), bounds=bounds)
    # extract weights
    return results.x

def gmv(cov):
    """
    Returna os pesos do portfólio GMV dada a matriz de covariância
    """
    n=cov.shape[0]
    # portfólio MSR com retorno iguais = minimizar a volatilidade
    return msr(0, np.repeat(1, n), cov)

def minimize_vol(target_return, er, cov):
    """
    Função que dá os pesos do portfólio com um retorno fixo (target_return) dados os vetores de retornos (er) e a matriz de 
    covariância (cov)
    """
    # dimensã0
    n = er.shape[0]
    # ponto inicial da otimização
    init_guess = np.repeat(1/n, n)
    # limites para os pesos
    bounds = ((0.0, 1.0),)*n
    # constraint para o retorno ser fixo
    return_is_target = {'type': 'eq', 
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights, er)}
    # constraint para a soma de pesos ser 1
    weights_sum_to_1 = {'type': 'eq', 
                        'fun': lambda weights: np.sum(weights)-1}
    # minimização da volatilidades
    results = minimize(portfolio_vol, init_guess, args=(cov,), method='SLSQP', constraints=(return_is_target, weights_sum_to_1), bounds=bounds)
    # pesos
    return results.x

def optimal_weights(n_points, er, cov):
    """
    Gera uma lista de n_points pesos para cada portfólio na fronteira eficiente dados o vetor de retornos er e a matriz
    de covariância cov
    """
    target_rts = np.linspace(er.min(), er.max(), n_points)
    return [minimize_vol(t_ret, er, cov) for t_ret in target_rts]

def plot_ef(n_points, er, cov, style=':', figsize=(12,6), show_endpoints=False, show_ew=False, show_gmv=False, show_mcap=False, show_msr=False, w_mcap=None, riskfree_rate=None):
    """
    Plot da fronteira eficiente dados os retornos er e a matriz de covariância cov, com n_points sendo o número de pontos
    na curva e com kwargs para mostrar os pontos finais (show_endpoints), portfólio EW (show_ew), portfólio GMV (show_gmv),
    portfólio MCap (show_mcap), portfólio MSR (show_msr)
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er)*100 for w in weights]
    vols = [portfolio_vol(w, cov)*100 for w in weights]
    ef = pd.DataFrame({'Retornos (%)': rets, 'Volatilidade (%)': vols})
    ax = ef.plot.line(x='Volatilidade (%)', y='Retornos (%)', style=style, figsize=figsize, ylabel='Retornos (%)', label='Fronteira eficiente', c='black')
    if show_endpoints:
        ax.scatter(x = np.sqrt(cov[er.idxmax()][er.idxmax()])*100, y=er.max()*100, label=er.idxmax(), c='red', zorder=2, marker='s')
        ax.scatter(x = np.sqrt(cov[er.idxmin()][er.idxmin()])*100, y=er.min()*100, label=er.idxmin(), c='blue', zorder=2, marker='D')
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv, vol_gmv = portfolio_return(w_gmv, er)*100, portfolio_vol(w_gmv, cov)*100
        ax.scatter(x=vol_gmv, y=r_gmv, c= 'midnightblue', marker='o', label='GMV')
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew, vol_ew = portfolio_return(w_ew, er)*100, portfolio_vol(w_ew, cov)*100
        ax.scatter(x=vol_ew,y=r_ew, color = 'goldenrod', marker='^', label='EW')
    if show_mcap and w_mcap is not None:
        r_mcap, vol_mcap = portfolio_return(w_mcap, er)*100, portfolio_vol(w_mcap, cov)*100
        ax.scatter(x=vol_mcap,y=r_mcap, color = 'green', marker='X', label='MCap')
    if show_msr:
        w_msr = msr(riskfree_rate, er, cov)
        r_msr, vol_msr = portfolio_return(w_msr, er)*100, portfolio_vol(w_msr, cov)*100
        ax.scatter(x=vol_msr, y=r_msr, c= 'orange', marker='*', label='MSR')
    plt.legend(loc=1)
    return ax

def plot_ef_multiple(n_curves, n_points, er, cov, figsize=(12,6)):
    """
    Plot da fronteira eficiente para um portfólio com dois ativos e diferentes valores da correlação dados os retornos er, a função
    de matriz de covariância cov que recebe uma correlação, n_points sendo o número de pontos na curva e
    n_curves sendo o número de curvas entre as correlações -1 e 1
    """
    corr = np.linspace(-1, 1, n_curves)
    weights = optimal_weights(n_points, er, cov(corr[0]))
    rets = [portfolio_return(w, er)*100 for w in weights]
    vols = [portfolio_vol(w, cov(corr[0]))*100 for w in weights]
    ef = pd.DataFrame({'Retornos (%)': rets, 'Volatilidade (%)': vols})
    ax = ef.plot.line(x='Volatilidade (%)', y='Retornos (%)', figsize=figsize, ylabel='Retornos (%)', label=f'{corr[0]:.2f}', c='black')
    for corr_point in corr[1:]:
        weights = optimal_weights(n_points, er, cov(corr_point))
        rets = [portfolio_return(w, er)*100 for w in weights]
        vols = [portfolio_vol(w, cov(corr_point))*100 for w in weights]
        ef = pd.DataFrame({'Retornos (%)': rets, 'Volatilidade (%)': vols})
        ef.plot.line(x='Volatilidade (%)', y='Retornos (%)', ax=ax, label=f'{corr_point:.2f}')
    
    plt.legend(loc=1)
    return ax

# alocação

def run_cppi(risky_r, safe_r, m=3, start=1, floor=0.8, drawdown=None):
    """
    Calcula os pesos do portfólio CPPI em cada pregão dado os retornos do ativo com risco (risky_r), retornos do ativo sem risco (safe_r),
    multiplicador (m), patrimônio inicial (start), piso (floor) e um possível MaxDD (drawdown)
    Returna um dict com: patrimônio acumulado CPPI, patrimônio acumulado CPPI do ativo com risco, pesos do ativo com risco
    """

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r)    

    # parâmetros do CPPI

    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = floor*start
    peak = start

    # ponto inicial do algoritmo
    account_history = pd.DataFrame().reindex_like(risky_r)
    gap_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    # algoritmo CPPI
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        gap = (account_value-floor_value)/account_value
        risky_w = m*gap
        # sem empréstimo e alocação positiva
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # atualizar valor da carteira
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # atualizar histórico
        account_history.iloc[step] = account_value
        risky_w_history.iloc[step] = risky_w
        gap_history.iloc[step] = gap

    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {'Wealth': account_history,
                       'Risky Wealth': risky_wealth,
                       'Risky Allocation': risky_w_history,
                       'm': m,
                       'start': start,
                       'floor': floor,
                       'risky_r': risky_r

    }
    return backtest_result

def plot_cppi(cppi, riskfree_asset, title='', figsize=(12,6), stock_name='Ação'):
    """
    Recebe um dict do CPPI, retornos de um ativo sem risco e retorna um plot do patrimônio acumulado
    """
    ax = cppi['Wealth'].plot(figsize=figsize)
    cppi['Risky Wealth'].plot(ax=ax)
    compound(riskfree_asset).plot(ax=ax, style=':k')
    ax.axhline(cppi['floor']*cppi['start'], color='red', linestyle='--')
    plt.xlabel('Data do pregão')
    plt.ylabel('Patrimônio acumulado (R$)')
    plt.title(title)
    plt.legend(['CPPI', stock_name, 'CDI' , 'Piso'], loc=2)

def plot_allocation(cppi, figsize=(20,10), stock_name='Ação', title=''):
    """
    Recebe um dict do CPPI e retorna plots do patrimônio acumulado CPPI e histórico dos pesos CPPI
    """
    _, ax = plt.subplots(2, sharex=True, figsize=figsize)

    cppi['Wealth'].plot(ax=ax[0], color='black', ylabel = 'Patrimônio acumulado (R$)')
    ax[0].axhline(cppi['floor']*cppi['start'], color='red', linestyle=':')
    ax[0].legend(['CPPI', 'Piso'], loc=2)
    ax[0].tick_params(bottom=False)

    cppi['Risky Allocation'].multiply(100).plot(ax=ax[1], c='red', ylabel = 'Pesos (%)')
    (1-cppi['Risky Allocation']).multiply(100).plot(ax=ax[1], c='blue')
    ax[1].legend([stock_name, 'CDI'], loc=2)
    ax[1].tick_params(bottom=False)

    plt.xlabel('Data do pregão')
    plt.title(title)

def pf_mix(r1, r2, allocator, **kwargs):
    """
    Mistura dos retornos de dois ativos r1 e r2 dado uma função de alocação para os pesos, retorna um DataFrame com os retornos do portfólio
    """
    if isinstance(r1, pd.Series):
        r1 = pd.DataFrame(r1, columns=[r1.name])
    if isinstance(r2, pd.Series):
        r2 = pd.DataFrame(r2, columns=[r2.name])

    weights = allocator(r1, r2, **kwargs).to_numpy()
    columns=r1.columns
    index=r1.index
    r1 = r1.to_numpy()
    r2 = r2.to_numpy()
    r_mix = weights*r1 + (1-weights)*r2
    return pd.DataFrame(r_mix, index=index, columns=columns)

def fixed_weights_mix(rets, weights, name):
    """
    Dados os retornos (rets) e os pesos (weights) devolve os retornos do portfólio
    """
    series = (rets*weights).sum(axis='columns')
    series.name = name
    return series

def fixedmix_allocator(r1, r2, w1):
    """
    Alocador de mistura fixa dado um peso w1 e retornos r1 e r2
    """
    if isinstance(r1, pd.Series):
        r1 = pd.DataFrame(r1, columns=[r1.name])
    
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def glidepath_allocation(r1, r2, start_glide=1, end_glide=0):
    """
    Alocador glidepath para os retornos r1 e r2 dados pesos iniciais (start_glide) e finais (end_glide) no ativo r1
    """
    if isinstance(r1, pd.Series):
        r1 = pd.DataFrame(r1, columns=[r1.name])
    
    n_points, n_columns = r1.shape
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_columns, axis='columns')
    paths.index = r1.index
    paths.columns = r1.columns
    return paths