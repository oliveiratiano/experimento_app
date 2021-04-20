from src import transformer
import warnings
import os, shutil
import numpy as np
import traceback
warnings.filterwarnings("ignore")

transformer.extrair_corpus()
documentos_validos = transformer.ler_documentos_validos(quantidade = 1000)

os.mkdir('resultados')

lista_k = np.arange(2,5)
grid_minfreqs = [0, 50, 100]
grid_stopwords = [True, False]
grid_ica = [True, False]
grid_tesauro = [True, False]
grid_dimensoes = [100]
try:
    transformer.rodar_experimento(documentos_validos, grid_minfreqs, grid_stopwords, grid_ica, grid_tesauro, grid_dimensoes, lista_k)
except Exception e:
    import pdb; pdb.set_trace()
    traceback.print_exc()

os.remove('tesauro_stf.csv')
shutil.rmtree('dados')

