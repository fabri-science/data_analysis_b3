O arquivo script_b3_collect.py é um script que quando executado ele realiza o parsing das cotações históricas 
fornecidas pela B3 em um ano especificado pelo usuário. Para usá-lo o usuário deve extrair o TXT das cotações históricas
(COTAHIST_AYYYY.TXT) do arquivo zip e ter os módulos b3fileparser, pandas e re instalados. Para executar o script o 
usuário deve dar as três seguintes sys.argv:
    - tipo_da_acao: as strings aceitas são 'ACOES_ORDINARIAS', 'ACOES_PREFERENCIAIS', 'ACOES_PREFERENCIAIS_CL_A' 
    e 'ACOES_PREFERENCIAIS_CL_B'
    - ano: no formato YYYY
    - path: diretório das cotações históricas COTAHIST_AYYYY.TXT em relação a script_b3_collect.py
Com isso feito será criado um arquivo csv path+tipo_de_acao+'_'+ano+'.csv'. Por exemplo, o comando 
    python3 script_b3_collect.py ACOES_ORDINARIAS 2020 ./data
Gera o arquivo ACOES_ORDINARIAS_2020.csv a partir de COTAHIST_A2020.TXT no diretório ./data