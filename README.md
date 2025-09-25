# Análise de Dados da B3

**Autor:** Matheus Augusto Fabri

Este repositório contém uma análise das ações ordinárias (código 3) negociadas na B3 no quinquênio de 2020-2024. No Jupyter notebook `rk_analysis_pf_construction_b3.ipynb` realizamos uma análise de risco destes ativos, isto é, discutimos retorno anualizado, a volatilidade anualizada, o drawdown máximo, índice de Sharpe, valor em risco, valor em risco condicional, assimetria e curtose. Também analisamos estratégias para a formação de portfólios usando estas ações (mistura fixa, glidepath e constant proportion portfolio insurance) e contruímos a fronteira eficiente. Além disso criamos também um índice de mercado baseado em volume de negociação análogo ao IBOV e S&P500 para comparação das estratégias de investimento. Os arquivos deste repositório são divididos em duas categorias: data e code/analysis. Na parte de data temos:
- `ACOES_ORDINARIAS_YYYY.csv`: Tabelas com data do pregão, código de negociação, preço médio, nome da empresa e quantidade negociada das ações ordinárias negociadas na B3 no período 2020-2024 com YYYY sendo ano em questão. Estes arquivos são gerados com `script_b3_collect.py`, para maiores informações ver `readme_script.txt`, `script_b3_collect.py` e `rk_analysis_pf_construction_b3.ipynb`.
- `cdi_pd_percent.csv`: Série histórica do CDI (percentual) no período 2020-2024 extraído de [Time Series Management System BCB](https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries).
- `prices_5y_b3.csv`, `rets_5y_b3.csv`, `vol_5y_b3.csv`: Tabelas auxiliares contendo preços, retornos e volumes de negociação gerados pelo notebook `rk_analysis_pf_construction_b3.ipynb`.

Já na parte de code/analysis temos:
- `rk_analysis_pf_construction_b3.ipynb`: Jupyter notebook contendo a análise de dados e construção de portfólios descritos acima e informações acerca da coleta e seleção de dados.
- `risk_module.py`: Módulo auxiliar com as funções usadas em `rk_analysis_pf_construction_b3.ipynb`.
- `script_b3_collect.py`: Script para a coleta de dados da B3 com as informações de uso dadas em `readme_script.txt`.
- `readme_script.txt`: Informações de uso de `script_b3_collect.py`.

A versão do Python utilizada neste projeto foi 3.12.7. Já as versões das bibliotecas utilizadas são:
- `pandas`: 2.2.2
- `numpy`: 1.26.4
- `matplotlib`: 3.9.2
- `seaborn`: 0.13.2
- `scipy`: 1.13.1

Além disso usamos também o módulo b3fileparser para fazer o parsing dos arquivos de cotações históricas da B3 (com `script_b3_collect.py`) que pode ser instalado via pip:
```python 
pip install b3fileparser
```

