import sys
import warnings
import os, shutil
import numpy as np
import traceback
from src import transformer
import random
import time

def main():
    start = time.time()
    erro = False
    warnings.filterwarnings("ignore")
    if not os.path.exists('dados'):
        try:
            transformer.baixar_corpus()
        except:
            print('erro: o corpus deverá ser baixado manualmente no diretório dados/corpus_tratado')
            traceback.print_exc()
            erro = True
    if not os.path.isfile('tesauro_stf.csv'):
        try:
            transformer.baixar_tesauro()
        except:
            print('erro: o tesauro deverá ser baixado manualmente no diretório experimento_app')
            traceback.print_exc()
            erro = True
    if not os.path.exists('modelos'):
        try:
            transformer.baixar_modelos()
        except:
            print('erro: os modelos do nilc deverao ser baixados manualmente no diretório modelos/100 e modelos/300')
            traceback.print_exc()
            erro = True
    if not os.path.exists('resultados'):
        os.mkdir('resultados')
    if not erro: 
        rnd = random.randint(0,10000)
        documentos_validos = transformer.ler_documentos_validos() 
        lista_k = np.arange(2,5)
        freq_min = int(sys.argv[1])
        remover_stopwords_pt = sys.argv[2] == '--true'
        usar_ica = sys.argv[3] == '--true'
        usar_tesauro = sys.argv[4] == '--true'
        tam_vec = int(sys.argv[5])
        #execução do grid do experimento

        w2v_geral, ftt_geral, glv_geral = transformer.importar_modelos_nilc(tam_vec)
        #w2v_geral, ftt_geral, glv_geral = [],[],[]
        exp = rnd
        opc_tesauro = '__com_crit_tesauro' if usar_tesauro  else '__sem_crit_tesauro'
        opc_ica = '__com_crit_ica' if usar_ica  else '__sem_crit_ica'
        opc_stopwords = '__removeu_sw_pt' if remover_stopwords_pt  else '__manteve_sw_pt'
        exp = '__minfreq_' + str(freq_min) + opc_tesauro + opc_ica + opc_stopwords + '__' + str(tam_vec) + '_dims__seed-' + str(exp)
        dir_experimento = 'experimento_'+str(exp)
        print("----------------------- INICIANDO EXPERIMENTO "+ str(exp) + " -----------------------")
        
        transformer.rodar_experimento(dir_experimento, documentos_validos, freq_min, remover_stopwords_pt, 
                                        usar_ica, usar_tesauro, tam_vec, lista_k, rnd, exp, 
                                        w2v_geral, ftt_geral, glv_geral)

        print('fazendo faxina nos dados do experimento...')                            
        shutil.rmtree('dados/'+dir_experimento)
        print('...faxina concluída.')
        print("----------- EXPERIMENTO COM SEED "+ str(rnd) + " CONCLUIDO -----------")
        end = time.time()
        print('tempo do experimento: ' + str((end - start)/60) +' minutos')
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

