from src import transformer
import warnings
import os, shutil
import numpy as np
warnings.filterwarnings("ignore")

transformer.extrair_corpus()
documentos_validos = transformer.ler_documentos_validos()

os.mkdir('resultados')

lista_k = np.arange(2,5)
grid_minfreqs = [0, 50, 100]
grid_stopwords = [True, False]
grid_ica = [True, False]
grid_tesauro = [True, False]
grid_dimensoes = [100]
transformer.rodar_experimento(documentos_validos, grid_minfreqs, grid_stopwords, grid_ica, grid_tesauro, grid_dimensoes, lista_k)
os.remove('tesauro_stf.csv')
shutil.rmtree('dados')

