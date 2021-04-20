import sys
import warnings
import os, shutil
import numpy as np
import traceback
from src import transformer

def main():
    erro = False
    warnings.filterwarnings("ignore")
    if not os.path.exists('dados'):
        try:
            transformer.baixar_corpus()
        except:
            print('erro: o corpus deverá ser baixado manualmente no diretório dados/corpus_tratado')
    if not os.path.exists('modelos'):
        try:
            transformer.baixar_modelos()
        except:
            print('erro: os modelos do nilc deverao ser baixados manualmente no diretório modelos/100 e modelos/300')
    if not os.path.exists('resultados'):
        os.mkdir('resultados')
    if not erro:        
        documentos_validos = transformer.ler_documentos_validos(quantidade = 1000) 
        lista_k = np.arange(2,5)
        grid_minfreqs = [0, 50, 100]
        grid_stopwords = [True]
        grid_ica = [True, False]
        grid_tesauro = [True, False]
        grid_dimensoes = [100, 300]
        transformer.rodar_experimento(documentos_validos, grid_minfreqs, grid_stopwords, grid_ica, grid_tesauro, grid_dimensoes, lista_k)

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

