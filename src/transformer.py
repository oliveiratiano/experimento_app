from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import multiprocessing
import nltk
import logging
import subprocess
import os
import time
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, precision_recall_fscore_support, homogeneity_completeness_v_measure
import requests
import random
from tqdm import tqdm
import zipfile
import stat
import shutil

#recebe o id de um documento e o diretorio onde ele se encontra, como strings
#retorna o texto contido neste documento
def recuperar_teor(idx, diretorio):
    with open(diretorio + idx, "r", encoding='utf-8') as f:
        contents = f.read()
    return contents

#recebe como entrada o dataframe do conjunto de treinamento contendo id, teores e assunto
#retorna conjunto com termos cujo ICA é maior que a média observada no vocabulário extraído dos teores
def sel_termos_ica(X_treino):
    #agrupa teores por assunto, concatenando-os
    df_assuntos = pd.DataFrame()
    for classe in tqdm(X_treino['assunto'].unique()):
        concat = ' '.join(X_treino.loc[X_treino['assunto'] == classe].teores)
        df_assuntos = pd.concat([df_assuntos, pd.DataFrame([(classe, concat)])], ignore_index = True)
    df_assuntos.columns = ['assuntos', 'teores']
    print("-processando strings do corpus")
    #calcula o ICA dos termos que aparecem no conjunto de treino
    stopwords = nltk.corpus.stopwords.words('portuguese')
    df_assuntos['teores'] = df_assuntos.teores.str.replace('\n', ' ').str.strip()
    print("-treinando vetorizador")
    vectorizer = TfidfVectorizer(stop_words = stopwords, max_df = 50, smooth_idf = False)
    vec = vectorizer.fit(df_assuntos.teores)
    
    #retorna conjunto com termos cujo ICA é maior que a média de ICA do vocabulário
    df = pd.DataFrame(list(zip([k for k, v in vec.vocabulary_.items()], vec.idf_)), columns = ['termo', 'idf']).sort_values(by='idf')
    estats_idf = pd.DataFrame(df.idf.describe())
    corte_idf = estats_idf.loc['mean',:]
    df = df[df.idf >= corte_idf[0]]
    print("-ICA processado")
    return(set(df.termo))

#recebe uma série do pandas contendo o corpus, o valor mínimo de frequência das palavras e um conjunto de stopwords
#retorna um conjunto de palavras que aparecem no mínimo freq_min vezes no corpus
def sel_termos_freq(corpus, freq_min, stopwords, remover_stopwords_pt):
    contagem = corpus.teores.str.split(expand=True).stack().value_counts()
    if remover_stopwords_pt:
        p_validas = set(contagem.index) - set(stopwords)
    else:
        p_validas = set(contagem.index)
    contagem = contagem.loc[contagem.index.intersection(p_validas)][contagem>=freq_min]
    return(set(contagem.index))

#retorna os termos do tstf compostos por apenas uma palavra
def sel_termos_tesauro():    
    tesauro = pd.read_csv("tesauro_stf.csv")
    tesauro['unico'] = tesauro.apply(lambda x: 1 if x['termo'].strip().count(' ') == 0
                                        else 0, axis = 1)
    termos_tesauro = tesauro.loc[tesauro.unico == 1].reset_index().termo.str.lower()
    return(set(termos_tesauro))

#recebe o corpus, a frequência mínima de palavras, e um conjunto de stopwords
#retorna um conjunto com o vocabulário
def extrair_vocabulario(corpus, corte_freq, stopwords, remover_stopwords_pt, usar_ica, usar_tesauro):
    if usar_ica:
        print("extraindo termos com base no ICA")
        termos_ica = sel_termos_ica(corpus)
    else:
        termos_ica = set()

    print("extraindo termos com base na frequência - geralmente leva menos de 4 minutos")
    termos_freq = sel_termos_freq(corpus, corte_freq, stopwords, remover_stopwords_pt)

    if usar_tesauro:
        print("extraindo termos do tesauro")
        termos_tesauro = sel_termos_tesauro()
    else:
        termos_tesauro = set()
    vocabulario = termos_tesauro.union(termos_ica).union(termos_freq)
    print("***************extração de vocabulário concluída: " + str(len(vocabulario)) + ' palavras******************')
    return vocabulario

def treinar_word2vec(corpus, exp, tam_vec):    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("treinando modelo word2vec")
    model = Word2Vec(LineSentence(corpus), size=tam_vec, window=5,                 
                 min_count=5, sg=1, hs=1, iter=10, workers=multiprocessing.cpu_count(), sample = 0.00001)
    model.save("/app/dados/experimento_" + str(exp) + "/w2v_jur.model")
    return model

def treinar_fasttext(corpus, exp, tam_vec):    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("treinando modelo fasttext")
    model = FastText(size=tam_vec, window=5,                 
                 min_count=5, sg=1, hs=1, iter=10, workers=multiprocessing.cpu_count())
    model.build_vocab(corpus_file=corpus)
    total_words = model.corpus_total_words
    model.train(corpus_file=corpus, total_words=total_words, epochs=5)
    model.save("/app/dados/experimento_" + str(exp) + "/ftt_jur.model")
    return model

    
def treinar_glove(exp, tam_vec):    
    print("treinando modelo glove")
    corpus="/app/dados/experimento_"+str(exp)+"/base_treino_glv.txt"
    vocab_file="/app/dados/experimento_"+str(exp)+"/glove_vocab.txt"
    coocurrence_file="/app/dados/experimento_"+str(exp)+"/glv_concurrence.bin"
    coocurrence_shuf_file="/app/dados/experimento_"+str(exp)+"/glv_concurrence_shuf.bin"
    save_file="/app/dados/experimento_"+str(exp)+"/glv_jur"
    vector_size=tam_vec

    #import pdb; pdb.set_trace()
    st = os.stat('/app/glove/glove.sh')
    os.chmod('/app/glove/glove.sh', st.st_mode | stat.S_IEXEC)
    cmd = ["./glove.sh", corpus, vocab_file, coocurrence_file, coocurrence_shuf_file, save_file, str(vector_size)]
    treinar_glove = subprocess.Popen(cmd,  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='glove')
    output, errors = treinar_glove.communicate()
    treinar_glove.wait()
    print('output:')
    print(output.decode('utf-8'))
    print('errors:')
    print(errors.decode('utf-8'))
    print("treinamento concluído")

    glove_file = '/app/dados/experimento_'+str(exp)+'/glv_jur.txt'
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    return(model)

def remover_stopwords(texto, stopwords):
    tokens = texto.split(" ")
    tokens_filtrados = [p for p in tokens if not p in stopwords]
    return (" ").join(tokens_filtrados).strip()

#recebe uma serie com documentos, um modelo gensim e um conjunto com o vocabulário
#retorna um vetor numpy com a soma dos vetores dos termos dos documentos que estão vocabulário
def calc_vet_soma(serie_documentos, modelo, vocab):
    lista_vecs = []
    for documento in tqdm(serie_documentos):
        lista_tokens = documento.split(' ')
        vetor_doc = np.zeros(100).reshape(1, -1)
        for token in lista_tokens:
            if token in vocab:
                try:
                    vetor_doc += modelo.wv.get_vector(token)
                except KeyError:
                    pass
        lista_vecs.append(normalize(vetor_doc.reshape(1, -1)))
    return(lista_vecs)


def criar_representacoes_soma(X_teste, y_teste, vocab, diretorio, w2v_jur, ftt_jur, glv_jur, w2v_geral, ftt_geral, glv_geral, exp):
    base_teste = pd.DataFrame(X_teste)
    base_teste['id'] = base_teste.id + '.txt'
    print("recuperando teores da base de teste")
    base_teste['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(base_teste.id)]
    base_teste['assunto'] = y_teste
    docs_teste = base_teste.teores.reset_index().teores

    print("criando representações word2vec geral")
    base_teste['vec_w2v_ger_soma'] = calc_vet_soma(docs_teste, w2v_geral, vocab)
    print("criando representações fasttext geral")
    base_teste['vec_ftt_ger_soma'] = calc_vet_soma(docs_teste, ftt_geral, vocab)
    print("criando representações glove geral")
    base_teste['vec_glv_ger_soma'] = calc_vet_soma(docs_teste, glv_geral, vocab)
    print("criando representações word2vec juridico")
    base_teste['vec_w2v_jur_soma'] = calc_vet_soma(docs_teste, w2v_jur, vocab)
    print("criando representações fasttext juridico")
    base_teste['vec_ftt_jur_soma'] = calc_vet_soma(docs_teste, ftt_jur, vocab)
    print("criando representações glove juridico")
    base_teste['vec_glv_jur_soma'] = calc_vet_soma(docs_teste, glv_jur, vocab)
    base_teste.to_csv('dados/experimento_'+str(exp)+'/vetores_teste.csv', index=False)

def importar_modelos_nilc(tam_vec):
    # importando modelos de domínio geral
    if tam_vec == 100:
        #w2v
        file_id = '1UHqJiYMhe0V2PfpUIWBIHtNCp6CB2bKj'
        destination = 'w2v_skip_nilc.txt'
        print('Baixando arquivo '+ destination + '. Por favor aguarde.')
        download_file_from_google_drive(file_id, destination, 895000000)
        print("Importando w2v geral")
        w2v_geral = KeyedVectors.load_word2vec_format(destination)
        os.remove(destination)

        #ftt
        file_id = '160wENCPPnZwOBm7gN95NXeCu4Uf2X-AN'
        destination = 'ftt_skip_nilc.txt'
        print('Baixando arquivo '+ destination + '. Por favor aguarde.')
        download_file_from_google_drive(file_id, destination, 812000000)
        print("Importando ftt geral")
        ftt_geral = KeyedVectors.load_word2vec_format(destination)
        os.remove(destination)

        #glv
        file_id = '1SeI7dZ4ZBF0_W9d6ROnl5S_71_VjSDGT'
        destination = 'glove_nilc.txt'
        print('Baixando arquivo '+ destination + '. Por favor aguarde.')
        download_file_from_google_drive(file_id, destination, 895000000)
        print("Importando glv geral (leva uns 3 minutos)")
        glv_geral = KeyedVectors.load_word2vec_format(destination)
        os.remove(destination)
    else:
        #w2v
        file_id = '10rXEAfSITfBxx4MlPcubnUGGbZNDumjI'
        destination = 'w2v_skip_nilc.txt'
        print('Baixando arquivo '+ destination + '. Por favor aguarde.')
        download_file_from_google_drive(file_id, destination, 2800000000)
        print("Importando w2v geral (leva uns 3 minutos)")
        w2v_geral = KeyedVectors.load_word2vec_format(destination)
        os.remove(destination)

        #ftt
        file_id = '17YlATNDj1wuI9_jZSEE4Z0jaLESpf5e6'
        destination = 'ftt_skip_nilc.txt'
        print('Baixando arquivo '+ destination + '. Por favor aguarde.')
        download_file_from_google_drive(file_id, destination, 2800000000)
        print("Importando ftt geral (leva uns 3 minutos)")
        ftt_geral = KeyedVectors.load_word2vec_format(destination)
        os.remove(destination)

        #glv
        file_id = '1qZJQuVGctPJ7DzatiyBTg0-nZvXsVicT'
        destination = 'glove_nilc.txt'
        print('Baixando arquivo '+ destination + '. Por favor aguarde.')
        download_file_from_google_drive(file_id, destination, 2800000000)
        print("Importando glv geral")
        glv_geral = KeyedVectors.load_word2vec_format(destination)
        os.remove(destination)

    return(w2v_geral, ftt_geral, glv_geral)

def treinar_modelos_jur(X_treino, X_teste, y_treino, y_teste, vocab, diretorio, exp, tam_vec):
    # treinando modelos de dominio juridico
    glv_jur = treinar_glove(exp, tam_vec)
    w2v_jur = treinar_word2vec('dados/experimento_'+str(exp)+'/base_treino.txt', exp, tam_vec)
    ftt_jur = treinar_fasttext('dados/experimento_'+str(exp)+'/base_treino.txt', exp, tam_vec)
    
    return(w2v_jur, ftt_jur, glv_jur)

#recebe o número do experimento e os ids da base de treino
#cria os arquivos necessários para o treinamento dos modelos
#retorna a base de treino enriquecida dos teores e dos assuntos
def criar_base_treino(exp, X_treino, y_treino, diretorio, stopwords):
    X_treino = pd.DataFrame(X_treino)
    X_treino['id'] = X_treino.id + '.txt'


    print("criando base de treino para o experimento "+str(exp))
    if not os.path.exists('dados/experimento_'+str(exp)):
        os.makedirs('dados/experimento_'+str(exp))

    #a base de treino para o word2vec e fasttext deve ter uma frase por linha
    #a base de treino para o glove deve ter um documento por linha
    base_treino = open('dados/experimento_'+str(exp)+'/base_treino.txt', 'w+', encoding='utf8')
    base_treino_glv = open('dados/experimento_'+str(exp)+'/base_treino_glv.txt', 'w+', encoding='utf8')
    tokens = 0
    for documento in tqdm(X_treino.id.values):
        doc = open(diretorio + documento, 'r', encoding='utf8')
        for frase in doc:
            base_treino.write(frase)
            tokens += len(frase.split(" "))
        doc.close()

        doc = open(diretorio + documento, 'r', encoding='utf8')
        teor_completo = doc.read().replace('\n', '')
        teor_completo = remover_stopwords(teor_completo, stopwords)
        base_treino_glv.write(teor_completo + '\n')
        doc.close()

    base_treino.close()
    base_treino_glv.close()
    print(str(tokens)+ " tokens copiados com sucesso")

    print("preparando documentos para extração do vocabulário:")
    X_treino['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(X_treino.id)]
    X_treino['assunto'] = y_treino
    return X_treino

def converter_string_array(str_arr):
    str_arr = re.sub(' +', 
           ' ', 
           str_arr.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '))
    arr = np.fromstring(str_arr, sep=' ')
    return(arr.reshape(1, -1).reshape(1,-1))

def calc_matriz_sim(vetores, dir_experimento):
    print("calculando matriz de similaridade nos vetores "+vetores.name)
    sim_m = np.empty(shape=[vetores.shape[0], vetores.shape[0]])
    for i in tqdm(range(0, vetores.shape[0])):
        for j in range(0, vetores.shape[0]):
            a, b = vetores[i], vetores[j].T
            sim_m[i][j] = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return(sim_m)

def calcular_sim_assuntos(assuntos, sim_docs, modelo, dir_experimento):
    print('calculando a similaridade entre assuntos para o modelo '+modelo)
    lista_sim_assuntos = []
    lista_assuntos = assuntos.unique()
    
    for i in tqdm(range(0, lista_assuntos.shape[0])):
        for j in range(0, lista_assuntos.shape[0]):    
            assunto_a = lista_assuntos[i]
            assunto_b = lista_assuntos[j]

            indices_a = assuntos[assuntos == assunto_a].index.values
            indices_b = assuntos[assuntos == assunto_b].index.values
            x = sim_docs[np.ix_(indices_a,indices_b)]

            #Se os assuntos forem os mesmos, apenas o triângulo superior (sem a diagonal principal) deve ser consideradono cálculo da média
            #caso contrário, todos os elementos podem ser considerados
            if assunto_a == assunto_b:
                ind_sup = np.triu_indices(max(len(indices_a), len(indices_b)), k=1)
                sim = x[ind_sup].mean()
            else:
                sim = sim_docs[np.ix_(indices_a,indices_b)].mean()
            lista_sim_assuntos.append((assunto_a, assunto_b, sim))
    lista_sim_assuntos = pd.DataFrame.from_records(lista_sim_assuntos, columns = ['assunto_a', 'assunto_b', 'sim_cos'])
    pivot = lista_sim_assuntos.pivot(index='assunto_a', columns='assunto_b', values='sim_cos')
    pivot.to_csv('resultados/'+dir_experimento+'/sim_assuntos_'+modelo+'.csv')
    plt.cla()

def computar_scores_agrupamento(X, y, dir_experimento, modelo, lista_k):
    lista_scores_k = []
    for k in tqdm(lista_k):
        kmeans = KMeans(n_clusters=k, random_state=0, verbose = 0).fit(X)
        preds_kmeans = kmeans.predict(X)
        sil_score_kmeans = silhouette_score(X, preds_kmeans)
        ari_kmeans = adjusted_rand_score(y, preds_kmeans)
        hcv_kmeans = homogeneity_completeness_v_measure(y, preds_kmeans)
        lista_scores_k.append((k, 
                              sil_score_kmeans, 
                              ari_kmeans, 
                              hcv_kmeans[0],
                              hcv_kmeans[1], 
                              hcv_kmeans[2]))
    return(lista_scores_k)

def gerar_graficos_kmeans(lista_scores_k, dir_experimento, modelo):
    plt.rcParams.update(plt.rcParamsDefault)

    k_vals = [lista_scores_k[i][0] for i in range(1, len(lista_scores_k))]
    sil_scores = [lista_scores_k[i][1] for i in range(1, len(lista_scores_k))]
    ari_vals = [lista_scores_k[i][2] for i in range(1, len(lista_scores_k))]
    homog_vals = [lista_scores_k[i][3] for i in range(1, len(lista_scores_k))]
    complet_vals = [lista_scores_k[i][4] for i in range(1, len(lista_scores_k))]
    vscore_vals = [lista_scores_k[i][5] for i in range(1, len(lista_scores_k))]

    ax = sns.lineplot(k_vals, 
                    sil_scores, 
                    color = 'g', 
                    label = 'Silhueta', 
                    legend = False)    
    sns.lineplot(k_vals, 
                ari_vals,  
                label = 'ARI', 
                color = 'b', 
                legend = False)
    ax = sns.lineplot(k_vals, 
                    vscore_vals, 
                    color = 'r', 
                    label = 'V_score', 
                    legend = False)
    ax = sns.lineplot(k_vals, 
                    homog_vals, 
                    color = 'k', 
                    label = 'Homogeneidade', 
                    legend = False)
    ax = sns.lineplot(k_vals, 
                    complet_vals, 
                    color = 'y', 
                    label = 'Completude', 
                    legend = False)
    ax.set(xlabel='k', ylabel='Valor')
    ax.grid(False)

    x_vline_ari = k_vals[ari_vals.index(max(ari_vals))]
    x_vline_sil = k_vals[sil_scores.index(max(sil_scores))]
    x_vline_v =  k_vals[vscore_vals.index(max(vscore_vals))]
    x_vline_h =  k_vals[homog_vals.index(max(homog_vals))]
    x_vline_c =  k_vals[complet_vals.index(max(complet_vals))]

    ax.axvline(x_vline_ari, color = 'royalblue', linestyle = '--')
    ax.axvline(x_vline_sil, color = 'seagreen', linestyle = '--')
    ax.axvline(x_vline_v, color = 'indianred', linestyle = '--')
    ax.axvline(x_vline_h, color = 'dimgray', linestyle = '--')
    ax.axvline(x_vline_c, color = 'olive', linestyle = '--')

    ax.text(x_vline_ari+1, .01, x_vline_ari, color = 'royalblue')
    ax.text(x_vline_ari+3, max(ari_vals), round(max(ari_vals), 2), color = 'royalblue')
    ax.text(x_vline_sil+1, 0.01, x_vline_sil, color = 'seagreen')
    ax.text(x_vline_sil+3, max(sil_scores), round(max(sil_scores), 2), color = 'seagreen')
    ax.text(x_vline_v+1, 0.01, x_vline_v, color = 'indianred')
    ax.text(x_vline_v+3, max(vscore_vals), round(max(vscore_vals), 2), color = 'indianred')
    ax.text(x_vline_h+1, 0.01, x_vline_h, color = 'dimgray')
    ax.text(x_vline_h+3, max(homog_vals), round(max(homog_vals), 2), color = 'dimgray')
    ax.text(x_vline_c+1, 0.01, x_vline_c, color = 'olive')
    ax.text(x_vline_c+3, max(complet_vals), round(max(complet_vals), 2), color = 'olive')
    ax.set(ylim=(0, .7))

    lgd = ax.figure.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

    ax.get_figure().savefig('dados/'+dir_experimento+'/kmeans_'+modelo+'.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()  

def download_file_from_google_drive(id, destination, file_size):
    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    CHUNK_SIZE = 32*1024
    # TODO: this doesn't seem to work; there's no Content-Length value in header?
    total_size = file_size

    with tqdm(desc=destination, total=total_size, unit='B', unit_scale=True) as pbar:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    pbar.update(CHUNK_SIZE)
                    f.write(chunk)

def ler_documentos_validos(quantidade=40009, min_freq_assunto = 50, min_palavras_documento = 50):    
    df = pd.read_csv('dados/corpus_tratado/metadados.csv')[0:quantidade]
    documentos_validos = filtrar_assuntos(min_freq_assunto, df)
    documentos_validos = filtrar_documentos_curtos(min_palavras_documento, documentos_validos)
    print('total de documentos no corpus: '+str(documentos_validos.shape[0]))
    return documentos_validos

def filtrar_assuntos(min_freq_assunto, df):
    print('filtrando assuntos com frequencia minima ' + str(min_freq_assunto))
    df_validos = pd.DataFrame(df.groupby("Assunto").size()).reset_index()
    df_validos.columns = ["assunto", "quant"]
    df_validos = df_validos.loc[df_validos.quant >= min_freq_assunto]
    documentos_validos = df[df.Assunto.isin(df_validos.assunto)][["id", "Assunto"]].reset_index().drop('index', axis = 1)
    diretorio = "dados/corpus_tratado/"
    documentos_validos['arquivo'] = documentos_validos.id + '.txt'
    documentos_validos['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(documentos_validos['arquivo'])]
    return documentos_validos

def filtrar_documentos_curtos(min_palavras_documento, documentos_validos):
    print('filtrando assuntos com no minimo ' + str(min_palavras_documento) + ' palavras')
    documentos_validos['validos'] = documentos_validos.teores.apply(lambda x: 0 if len(x.split(' ')) <= min_palavras_documento else 1)
    documentos_validos = documentos_validos[documentos_validos['validos'] == 1].drop(['arquivo', 'teores', 'validos'], axis = 1).reset_index().drop('index', axis = 1)
    return documentos_validos

def extrair_corpus():
    file_id = '14UJLcDDZX5dIX6CJVCFkNHXZttTKfbC5'
    destination = 'corpus_tratado.zip'
    print('Baixando arquivo corpus. Por favor aguarde.')
    download_file_from_google_drive(file_id, destination, 150000000)
    with zipfile.ZipFile('corpus_tratado.zip') as zf:
        for member in tqdm(zf.infolist(), desc='Extraindo corpus: '):
            try:
                zf.extract(member, 'dados/')
            except zipfile.error as e:
                pass
    os.remove(destination)

def rodar_experimento(documentos_validos, minfreqs, op_stopwords, op_ica, op_tesauro, op_tam_vec, lista_k):
    rnd = random.randint(0,10000)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rnd)
    X = documentos_validos.id
    y = documentos_validos.Assunto
    stopwords = nltk.corpus.stopwords.words('portuguese')
    diretorio = "dados/corpus_tratado/"
    le = LabelEncoder()

    file_id = '1nyQS7VZMVTxx9sn9_cddsUJd01lTcOPj'
    destination = 'tesauro_stf.csv'
    download_file_from_google_drive(file_id, destination, 2030000)
    
    #index[0] são os indices de treino, e index[1] são os de teste
    #i é o código do experimento
    for i, index in enumerate(sss.split(X, y)):
        start = time.time()
        X_treino, X_teste = X[index[0]], X[index[1]]
        y_treino, y_teste = y[index[0]], y[index[1]]

        #execução do grid do experimento
        for tam_vec in op_tam_vec:
            # importando modelos de domínio geral
            w2v_geral, ftt_geral, glv_geral = importar_modelos_nilc(tam_vec)
            #w2v_geral, ftt_geral, glv_geral = [],[],[]
            for remover_stopwords_pt in op_stopwords:
                for usar_ica in op_ica:
                    for usar_tesauro in op_tesauro:
                        for freq_min in minfreqs:
                            exp = rnd
                            opc_tesauro = '__com_crit_tesauro' if usar_tesauro  else '__sem_crit_tesauro'
                            opc_ica = '__com_crit_ica' if usar_ica  else '__sem_crit_ica'
                            opc_stopwords = '__removeu_sw_pt' if remover_stopwords_pt  else '__manteve_sw_pt'
                            exp = '__minfreq_' + str(freq_min) + opc_tesauro + opc_ica + opc_stopwords + '__' + str(tam_vec) + '_dims__seed-' + str(exp)
                            print("----------------------- EXPERIMENTO "+ str(exp) + " -----------------------")

                            # instanciando o corpus do conjunto de treinamento
                            base_treino = criar_base_treino(exp, X_treino, y_treino, diretorio, stopwords)
                            # criando vocabulário
                            vocab = extrair_vocabulario(base_treino, freq_min, stopwords, remover_stopwords_pt, usar_ica, usar_tesauro)
                            # treinando modelos juridicos
                            w2v_jur, ftt_jur, glv_jur = treinar_modelos_jur(X_treino, X_teste, y_treino, y_teste, vocab, diretorio, exp, tam_vec)
                            #criando representações através da soma de vetores
                            criar_representacoes_soma(X_teste, y_teste, vocab, diretorio, w2v_jur, ftt_jur, glv_jur, w2v_geral, ftt_geral, glv_geral, exp)
                            end = time.time()
                            print('tempo do experimento: ' + str((end - start)/60) +' minutos')

                            ######DOC2VEC####
                            print('--------- Treinando doc2vec do experimento '+ str(exp)+' ---------')
                            dir_experimento = 'experimento_'+str(exp)
                            os.mkdir('resultados/'+dir_experimento)
                            corpus="dados/"+dir_experimento+"/base_treino_glv.txt"
                            model = Doc2Vec(corpus_file = corpus, vector_size=100, window=5, min_count=1, workers=8)
                            model.save("dados/"+dir_experimento+"/doc2vec_jur.model")
                            print('--------- Inferindo vetores para docs de teste do experimento '+ str(exp)+' ---------')
                            base_teste = pd.read_csv("dados/"+dir_experimento+"/vetores_teste.csv")
                            base_teste['doc2vec_jur'] = [normalize(model.infer_vector(x[0].split(' ')).reshape(1,-1)) for x in base_teste.teores]
                            base_teste.to_csv('dados/experimento_'+str(exp)+'/vetores_teste.csv', index=False)
                            
                            df = pd.read_csv('dados/'+dir_experimento+'/vetores_teste.csv')
                            print('++++++ modelos ++++++ ' + df.iloc[:,3:].columns)
                            
                            for modelo in df.iloc[:,3:].columns:
                                #####AGRUPAMENTOS###############
                                print('--------- Agrupando dados para o modelo '+ modelo + ' no experimento' +str(exp)+' ---------')
                                df[modelo] = df[modelo].apply(lambda x: converter_string_array(x))
                                X_kmeans = np.stack(df[modelo])
                                X_kmeans = X_kmeans.reshape(X_kmeans.shape[0], X_kmeans.shape[2])
                                y_kmeans = df['assunto']
                                le.fit(y_kmeans)
                                y_kmeans = le.transform(y_kmeans)
                                lista_scores_k = computar_scores_agrupamento(X_kmeans, y_kmeans, dir_experimento, modelo, lista_k)
                                gerar_graficos_kmeans(lista_scores_k, dir_experimento, modelo)
                                np.save('resultados/'+dir_experimento + '/lista_scores_k.npy', lista_scores_k)
                                print('******   dados de agrupamento do modelo ' + modelo + 'salvos.')
                                
                                #####MATRIZES DE SIMILARIDADE##############
                                print('--------- executando analyzer para experimento '+ str(exp)+' ---------')
                                sim_m = calc_matriz_sim(df[modelo], dir_experimento)
                                calcular_sim_assuntos(df['assunto'], sim_m, df[modelo].name, dir_experimento)
                                plt.close()
    print('fazendo faxina nos dados do experimento...')                            
    shutil.rmtree('dados/'+dir_experimento)
    print('...faxina concluída.')
    print("----------- EXPERIMENTO COM SEED "+ str(rnd) + " CONCLUIDO -----------")
 