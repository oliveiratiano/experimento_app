import sys
import warnings
import os, shutil
import numpy as np
import traceback
from src import transformer
import random

def main():
    import pdb; pdb.set_trace()
    erro = False
    warnings.filterwarnings("ignore")
    if not os.path.exists('dados'):
        try:
            transformer.baixar_corpus()
        except:
            print('erro: o corpus deverá ser baixado manualmente no diretório dados/corpus_tratado')
            erro = True
    if not os.path.exists('modelos'):
        try:
            transformer.baixar_modelos()
        except:
            print('erro: os modelos do nilc deverao ser baixados manualmente no diretório modelos/100 e modelos/300')
            erro = True
    if not os.path.exists('resultados'):
        os.mkdir('resultados')
    if not erro: 
        rnd = random.randint(0,10000)
        documentos_validos = transformer.ler_documentos_validos(quantidade = 1000) 
        lista_k = np.arange(2,5)
        grid_minfreqs = [0]
        grid_stopwords = [True]
        grid_ica = [True]
        grid_tesauro = [True]
        grid_dimensoes = [300]


        #execução do grid do experimento
        for tam_vec in grid_dimensoes:
            # importando modelos de domínio geral
            #w2v_geral, ftt_geral, glv_geral = importar_modelos_nilc(tam_vec)
            #w2v_geral, ftt_geral, glv_geral = [],[],[]
            for remover_stopwords_pt in grid_stopwords:
                for usar_ica in grid_ica:
                    for usar_tesauro in grid_tesauro:
                        for freq_min in grid_minfreqs:
                            exp = rnd
                            opc_tesauro = '__com_crit_tesauro' if usar_tesauro  else '__sem_crit_tesauro'
                            opc_ica = '__com_crit_ica' if usar_ica  else '__sem_crit_ica'
                            opc_stopwords = '__removeu_sw_pt' if remover_stopwords_pt  else '__manteve_sw_pt'
                            exp = '__minfreq_' + str(freq_min) + opc_tesauro + opc_ica + opc_stopwords + '__' + str(tam_vec) + '_dims__seed-' + str(exp)
                            dir_experimento = 'experimento_'+str(exp)
                            print("----------------------- INICIANDO EXPERIMENTO "+ str(exp) + " -----------------------")

                            transformer.rodar_experimento(dir_experimento, documentos_validos, grid_minfreqs, grid_stopwords, grid_ica, grid_tesauro, grid_dimensoes, lista_k, rnd, exp)
                            
                            print('fazendo faxina nos dados do experimento...')                            
                            shutil.rmtree('dados/'+dir_experimento)
                            print('...faxina concluída.')
                            print("----------- EXPERIMENTO COM SEED "+ str(rnd) + " CONCLUIDO -----------")

if __name__ == "__main__":
    main()




"""from src import transformer
import warnings
import os, shutil
import numpy as np
import traceback
warnings.filterwarnings("ignore")

transformer.extrair_corpus()
documentos_validos = transformer.ler_documentos_validos(quantidade = 1000)

os.mkdir('resultados')

lista_k = np.arange(2,5)
grid_minfreqs = [50]
grid_stopwords = [True]
grid_ica = [True]
grid_tesauro = [True]
grid_dimensoes = [100]
try:
    transformer.rodar_experimento(documentos_validos, grid_minfreqs, grid_stopwords, grid_ica, grid_tesauro, grid_dimensoes, lista_k)
except Exception as e:
    import pdb; pdb.set_trace()
    traceback.print_exc()

os.remove('tesauro_stf.csv')
shutil.rmtree('dados') """

