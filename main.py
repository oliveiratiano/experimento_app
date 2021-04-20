import sys
def main():
    if sys.argv == '-d':
        print('baixando dados')
    else:
        print('hello world')

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

