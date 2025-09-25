import sys
import pandas as pd
from b3fileparser.b3parser import B3Parser
import re

tipo_de_acao = sys.argv[1]
ano = sys.argv[2]
path = sys.argv[3]

print('Fazendo o parsing dos dados da B3 do ano de '+ano+' para '+tipo_de_acao)

def find_stocks(codigo_negociacao:str, tipo_acao:str)->bool:
  """
  Esta função recebe um código de negociação e um tipo de ação e verifica se o código em questão é compatível
  com o tipo de ação. A variável tipo_acao assume os seguintes valores 'ACOES_ORDINARIAS',
  'ACOES_PREFERENCIAIS', 'ACOES_PREFERENCIAIS_CL_A' e 'ACOES_PREFERENCIAIS_CL_B'
  """
  dict_codigos = {'ACOES_ORDINARIAS' : '^[A-Z]{4}3$', 'ACOES_PREFERENCIAIS': '^[A-Z]{4}4$', 
                  'ACOES_PREFERENCIAIS_CL_A': '^[A-Z]{4}5$', 'ACOES_PREFERENCIAIS_CL_B': '^[A-Z]{4}6$'}
  pattern = re.compile(dict_codigos[tipo_acao])
  return bool(pattern.match(codigo_negociacao))

def select_stocks(data:pd.DataFrame, tipo_acao:str)->pd.DataFrame:
  """
  Esta função recebe o dataframe da B3 completo e um tipo de ação e e retorna um DataFrame reduzido
  com todas as ações deste tipo. A variável tipo_acao assume os seguintes valores 'ACOES_ORDINARIAS',
  'ACOES_PREFERENCIAIS', 'ACOES_PREFERENCIAIS_CL_A' e 'ACOES_PREFERENCIAIS_CL_B'
  """
  mask = data['CODIGO_DE_NEGOCIACAO'].map(lambda x: find_stocks(x, tipo_acao))
  return data[mask][['DATA_DO_PREGAO', 'PRECO_MEDIO', 'CODIGO_DE_NEGOCIACAO', 'NOME_DA_EMPRESA', 'QUANTIDADE_NEGOCIADA']]

# importar B3 dados do ano

parser = B3Parser.create_parser(engine='pandas')
dados_b3 = parser.read_b3_file(path+'COTAHIST_A'+ano+'.TXT')
select_stocks(dados_b3, tipo_de_acao).to_csv(path+tipo_de_acao+'_'+ano+'.csv', index=False)

print('Parsing completo e dados salvos em .csv em '+path+tipo_de_acao+'_'+ano+'.csv')