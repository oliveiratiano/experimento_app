from src import transformer
import warnings
import os, shutil
warnings.filterwarnings("ignore")

transformer.extrair_corpus()
documentos_validos = transformer.ler_documentos_validos(quantidade=1000)

lista_k = [2,3,4,5]
grid_minfreqs = [0]
grid_stopwords = [True]
grid_ica = [True]
grid_tesauro = [True]
grid_dimensoes = [100]
transformer.rodar_experimento(documentos_validos, grid_minfreqs, grid_stopwords, grid_ica, grid_tesauro, grid_dimensoes, lista_k)
shutil.rmtree('dados')
os.remove('tesauro.csv')

