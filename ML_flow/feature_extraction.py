import pandas as pd
import numpy as np
import re
import csv
import configparser
#
from hermetrics.levenshtein import Levenshtein
from hermetrics.dice import Dice
from hermetrics.metric_comparator import MetricComparator

import nltk
from nltk.metrics import *
from nltk.metrics.distance import jaro_winkler_similarity, jaro_similarity

import math
import itertools
import collections

from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import manifold, datasets

class feature_extraction:

    def __init__(self):
        self.prueba = {
            'CODIGO': "COD",
            'PERSONA': "PER[S]?",
            'RESIDENCIA':'(RES|RESID)', #RES[IDENCIA]?
            'DOCUMENTO':'DOC',
            'PROVINCIA':'PROV',
            'PLAZA': 'PZA',  #'(PL|PLZA|PZA)',
            'NUMERO': 'NRO', #(NRO|NUM),
            'ABREVIATURA':  'ABREV', #(ABR|ABREV)',
            'DIRECCION':'DIR',
            #'PAIS':'PAIS',
            #'VIA': 'VIA',
            "CNAE93": "CNAE93",
            "CNO": "CNO",
            "ESTUDIO":"ESTU[DIO]?",
            "TRABAJO":"TRAB[AJO]?",
            "IDIOMA":"IDIO[MA]?",
            "NOMBRE":"NOM[BRE]?",
            "DOMICILIO":"DOMI[CILIO]?",
            "NACIMIENTO":"NACI[MIENTO]?"}

        self.file_name = self.get_config_data()
        self.dni_regex = "([a-z]|[A-Z]|[0-9])[0-9]{7}([a-z]|[A-Z]|[0-9])"
        self.email_regex = '^(.+\@.+\..+)$'
        self.movil_regex = '(\+34|0034|34)[ -]*(6|7)[ -]*([0-9][ -]*){8}'
        self.telephone_regex = "(\+34|0034|34)[ -]*(8|9)[ -]*([0-9][ -]*){8}"
        self.web_regex = '^(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)' \
                         '( [a-zA-Z0-9\-\.\?\,\’\/\\\+&amp;%\$#_]*)?$'
        self.datetime_regex = '^\d{2,5}[-/]\d{2}[-/]\d{2,5}$'
        self.val_cnae93 = list(pd.read_csv('../Input/CNAE93.csv')['Clase'])
        self.char_regex = "^([a-zA-Z]+[\s-])*[a-zA-Z]+$"
        self.direct_regex = "^([a-zA-Z]+\s)+(.)*\d{5}$"
        self.mic = MetricComparator()
        self.lev = Levenshtein()
        self.dice = Dice()
        #

    def get_config_data(self):
        '''
        Def: get config params from configuration file
        param
        return: output_path param
        '''
        file = '../Config/config.ini'
        config = configparser.ConfigParser()
        config.read(file)
        report_info = config['features_info']
        file_name = report_info['features_file']

        return file_name

    def init_feauture_columns(self, df):
        '''

        @param df:
        @return :
        '''
        dict_columns = {'len': 0,
                        'num_words': 0, 'is_date': False,
                        'is_all_int': False, 'is_all_characters': False,
                        'is_cnae': False, 'is_mobile': False,
                        'is_tlf': False, 'is_email': False,
                        'is_doc_identificate': False, 'is_website': False,
                        'is_direc':False, 'is_pais':False, 'list_words':None,
                        'is_cno':False, 'is_cnae93':False
                        }
        return df.assign(**dict_columns)

    ############################
    ## data from column field ##
    ############################

    def len_column(self, df):
        '''
        Create an additional column with the length of the values of column 'FIELD
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['len'] = df['FIELD'].apply(lambda x: len(x))
        return df

    def count_words(self, df):
        '''
        Create an additional column checking the number of words of char values
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['num_words'] = df['FIELD'].apply(lambda x: len(x.split(' ')), )
        return df

    def date_columns(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are strings
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_date'] = df['FIELD'].apply(lambda x: bool(re.findall(self.datetime_regex, str(x))))
        match_columns =  df[df['is_date']]['data_split'].unique()
        return match_columns

    def is_mobile(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['is_mobile'] = df['FIELD'].apply(lambda x: bool(re.findall(self.movil_regex, str(x))))
        match_columns =  df[df['is_mobile']]['data_split'].unique()
        return match_columns

    def is_tlf(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['is_tlf'] = df['FIELD'].apply(lambda x: bool(re.findall(self.telephone_regex, str(x))))
        match_columns =  df[df['is_tlf']]['data_split'].unique()
        return match_columns


    def is_email(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_email'] = df['FIELD'].apply(lambda x: bool(re.findall(self.email_regex, str(x))))
        match_columns =  df[df['is_email']]['data_split'].unique()
        return match_columns

    def is_doc_identificate(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_doc_identificate'] = df['FIELD'].apply(lambda x: bool(re.findall(self.dni_regex, str(x))))
        match_columns =  df[df['is_doc_identificate']]['data_split'].unique()
        return match_columns

    def is_website(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are web
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['is_website'] = df['FIELD'].apply(lambda x: bool(re.findall(self.web_regex, str(x))))
        match_columns =  df[df['is_website']]['data_split'].unique()
        return match_columns

    def is_all_char(self, df):
        '''
        '''
        df['is_all_characters'] = df['FIELD'].apply(lambda x: bool(re.findall(self.char_regex, str(x))))
        match_columns =  df[df['is_all_characters']]['data_split'].unique()
        return match_columns


    def all_int(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are numbers
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_all_int'] = df['FIELD'].apply(lambda x: str(x).isdigit())
        match_columns =  df[df['is_all_int']]['data_split'].unique()
        return match_columns


    def is_direct(self, df):
        '''
        '''
        df['is_direc'] = df['FIELD'].apply(lambda x: bool(re.findall(self.direct_regex, str(x))))
        match_columns =  df[df['is_direc']]['data_split'].unique()
        return match_columns


    ############################
    ## data from column class ##
    ############################

    def split_name_colum(self, df):
        '''
        @param
        @return
        '''
        df['data_split'] = df['CLASS'].apply(lambda x: ''.join([i for i in x.split('_')]))
        return df

    def is_cnae(self, df):
        '''
        '''
        val_cnae = list(map(str, self.val_cnae93))
        df['is_cnae'] = df.apply(lambda x: True if x['CLASS'] == 'CNAE93'
                                                   and x['FIELD'] in val_cnae else False, axis=1)
        return df

    def is_cno(self, df):
        '''
        @param
        @return
        '''


    def export_csv(self, df):
        return df.to_csv(self.file_name, sep=';', index=False)

    def run_feature(self, df):
        '''
        Main class process
        @param origin_df : dataframe initial
        @return None
        '''
        df = self.init_feauture_columns(df)
        ##

        df = self.split_name_colum(df)
        x = list(df['data_split'].unique())
        ########################################
        # 1. Información -- data in columns ###
        #######################################
        df = self.len_column(df)
        df = self.count_words(df)

        ######
        DF_matrix_info = pd.DataFrame(data=0, index=x, columns=['FECHA', 'ALL_INT',
                                                                  'ALL_CHAR', 'MOBILE',
                                                                  'TELEPHONE', 'EMAIL', 'IDENTIFI_PERSON',
                                                                  'WEBSITE', 'ALL_DIRECCION'
                                                                  ])
        list_match = self.date_columns(df)
        DF_matrix_info.loc[list_match, 'FECHA'] = 1
        list_match = self.all_int(df)
        DF_matrix_info.loc[list_match, 'ALL_INT'] = 1
        list_match = self.is_all_char(df)
        DF_matrix_info.loc[list_match, 'ALL_CHAR'] = 1
        list_match = self.is_mobile(df)
        DF_matrix_info.loc[list_match, 'MOBILE'] = 1
        list_match = self.is_tlf(df)
        DF_matrix_info.loc[list_match, 'TELEPHONE'] = 1
        list_match = self.is_email(df)
        DF_matrix_info.loc[list_match, 'EMAIL'] = 1
        list_match = self.is_doc_identificate(df)
        DF_matrix_info.loc[list_match, 'IDENTIFI_PERSON'] = 1
        list_match = self.is_website(df)
        DF_matrix_info.loc[list_match, 'WEBSITE'] = 1
        list_match = self.is_direct(df)
        DF_matrix_info.loc[list_match, 'ALL_DIRECCION'] = 1
        # 1.0 Empleando valores del proceos
        list_data = []
        #for name, group in df.groupby('CLASS'):
        #    set_data = set(group['FIELD'].unique())

        ######################################
        # 2. Basados en nombre de la columna #
        ######################################
        #df = self.split_name_colum(df)
        # PAIS ,PERSONA, VIAc
        # 2.1 Len = 1
        #
        #df_chars = df[df['is_all_characters']]
        #x = list(df['data_split'].unique())
        #print('cccccccccccccccccc')
        '''
        # Goods # VIA, PAIS, CODIGO --> VER PROBLEMA CON [IDDISPOSITIVO]
        for r in ['VIA', 'PERS', 'PAIS', 'NOMBRE', 'PERSONA', 'CODIGO']:
            #xXwer = [(item, jaccard_distance(item, r)) for item in x]
            xwer =  [(item, nltk.edit_distance(item, r)) for item in x]
            lis_items_p0 = [(item, self.lev.similarity(item, r, (0.5,1,0))) for item in x]
            lis_items_p0_ = [(item, self.dice.similarity(item, r)) for item in x]
            lis_items_p = [(item, self.mic.similarity(item, r)) for item in x]
            lis_items_p2 = [item for item in x if self.mic.similarity(item, r)['Dice'] >= 0.5
                            and self.mic.similarity(item, r)['Levenshtein'] >= 0.4]
        list_items = []
        '''
        #################################################
        # 2. Información -- data extract name columns ###
        #################################################
        DF_matrix_info_2 = pd.DataFrame(data=0, index=x,columns=['PERSONA','CODIGO',
                                                       'RESIDENCIA', 'DOCUMENTO',
                                                       'PROVINCIA', 'PLAZA', 'NUMERO',
                                                        'ABREVIATURA','DIRECCION', 'PAIS', 'VIA',
                                                        'CNAE93','CNO', "ESTUDIO",
                                                        "TRABAJO", "IDIOMA","NOMBRE",
                                                        "DOMICILIO", "NACIMIENTO"])
        DF_matrix_info_2['words'] = None

        ############
        DF_MATRIX_INFO = pd.DataFrame(data=None, columns=['words'],
                                      index=df.columns)


        #for r in ['PERSONA','CODIGO', 'RESIDENCIA','DOCUMENTO', 'PROVINCIA', 'PLAZA', 'NUMERO',
        #          'ABREVIATURA','DIRECCION', 'PAIS','VIA','CNAE93','CNO']:
        for r in ['CODIGO', 'PERSONA','RESIDENCIA','DOCUMENTO','PROVINCIA',
            'PLAZA','NUMERO', 'ABREVIATURA','DIRECCION',"CNAE93","CNO",
            "ESTUDIO", "TRABAJO", "IDIOMA", "NOMBRE", "DOMICILIO", "NACIMIENTO"]:
            R = re.compile(self.prueba[r])
            #XXXX=  list(R.findall(x))
            output = list(filter(R.findall, x))
            #SERTA = [(item, self.mic.similarity(item, r)) for item in output ]
            #sim_wols1 = [item for item in output if self.mic.similarity(item, r)['Dice'] > 0.45]
            #sim_wols2 = [item for item in output if self.mic.similarity(item, r)['Jaccard'] > 0.45]
            sim_wols3 = [item for item in output
                         if self.mic.similarity(item, r)['Dice'] >= 0.5 or
                         self.mic.similarity(item, r)['Jaro'] >= 0.6]
            #sim_wols4 = [item for item in output if self.mic.similarity(item, r)['Jaro'] > 0.45]

            ##### NUEVA INFORMACION DEL PROCEOS #####

            #for itemss in sim_wols3:
            #    sim_wolse3 = re.split(self.prueba[r],itemss)
            #    DF_matrix_info_2.loc[itemss,'words'] = [xx for xx in re.split(self.prueba[r],itemss) if xx != '']
            ####

            #sim_wols3_values = [df[df['data_split'] == dats].shape[0] for dats in sim_wols3]
            #datas =  math.log10( len(x) / (len(x) - len(sim_wols3)) )
            #pd_part = pd.DataFrame(1,index=[r] ,columns= sim_wols3)
            DF_matrix_info_2.loc[sim_wols3, r] = 1 #*datas #sim_wols3_values
            #DF_matrix_info_2.loc[sim_wols3, r] = 1
            #output = list(filter(map(lambda y: R.findall(y), x)),x)
            #xx = [ re.findall(, x) for item in x]

            #print(re.findall("(funny|fun)", )
            #apply(lambda x: bool(re.findall(self.email_regex, str(x))))
            #dict(columns=)
            #xx = [(item, nltk.edit_distance(item, r)) for item in x]
        #items_part1 = [ for key, value in DF_matrix_info_2.to_dict().items() ]

        ########################################################
        # 3. Información -- split_name from regex data CLASS ###
        ########################################################
        list_datas = []
        for colum_df in x:
            #print(colum_df)
            pp = []
            #not_str = [colum_df]
            #resultss = [  for name, reg in self.prueba.items() ]
            for name, reg in self.prueba.items():
                app_regex = re.split(reg, colum_df)
                result = list(np.setdiff1d(app_regex, [colum_df,'']))
                if len(result):
                    pp.append(result)
            ### Clean data
            list_join = list(itertools.chain.from_iterable(pp))
            info = [k for k, v in collections.Counter(list_join).items() if v >= len(pp)]

            ###
            #list_datas.append(  { colum_df:','.join(info)})
            list_datas.append(dict(colum=colum_df,words=','.join(info)))
            print('wwwwwwwww')


        ##########################
        ### INFORMATION CLASES ###
        ##########################
        result = DF_matrix_info_2.transpose()
        v_result = result.to_numpy()
        vectors = DF_matrix_info_2.to_numpy()

        clusterer = AgglomerativeClustering(n_clusters=4,
                                            linkage="average", affinity="euclidean")
        clusters = clusterer.fit_predict(v_result)


        ###########
        list_items = [dict(colum=key,
                           words=[y_key for y_key, y_value in value.items() if y_value != 0])
                for key, value in DF_matrix_info_2.to_dict().items() ]
        # Se inserta
        for ixt in list_items:
            #df['list_words'] = df['data_split'].apply(lambda x: ','.join(ixt['words'])
             #                  if (len(ixt['words']) > 0 and x == ixt['colum'] ) else None)
            df.loc[df.data_split == ixt['colum'],'list_words'] = ','.join(ixt['words'])

        ### unique values ###
        for name, group in df.groupby(['CLASS']):
            print(name)
            print(group)
        print('pppp')
        '''
        df['is_pais'] = df['FIELD'].apply(lambda x: True if x.lower() in self.cod_pais else False)
        # PAIS
        x2 = [ (item,self.mic.similarity(item,'VIA'))  for item in x  ]
        x21 = [ (item,self.mic.similarity(item,'PERS')) for item in x ]
        x22 = [  (item,self.mic.similarity(item,'PAIS'))  for item in x ]
        x23 = [ (item,self.mic.similarity(item,'NOMBRE'))  for item in x   ]
        x24= [ (item,self.mic.similarity(item,'PERSONA')) for item in x   ]
        x25 = [ (item,self.mic.similarity(item,'CODIGO')) for item in x   ]
        '''
        '''
        df = self.len_column(df)
        df = self.date_columns(df)
        # df = self.all_char(df)
        df = self.all_int(df)
        df = self.count_words(df)

        ###
        df = self.split_name(df)
        df = self.is_cnae(df)

        df_partial = df.copy()
        df_partial['is_cnae_'] = False
        df_partial = df_partial[df_partial['all_int'] == True]
        self.is_cnae(df_partial)
        df = df.join(df_partial['is_cnae'])

        df_partial = df.copy()
        df_partial['is_mobile'] = False
        df_partial = df_partial[df_partial['all_int'] == True]
        self.is_mobile(df_partial)
        df = df.join(df_partial['is_mobile'])

        df_partial = df.copy()
        df_partial['is_tlf'] = False
        df_partial = df_partial[df_partial['all_int'] == True]
        self.is_tlf(df_partial)
        df = df.join(df_partial['is_tlf'])

        df_partial = df.copy()
        df_partial['is_email'] = False
        df_partial = df_partial[df_partial['all_char'] == True]
        self.is_email(df_partial)
        df = df.join(df_partial['is_email'])

        df_partial = df.copy()
        df_partial['is_web'] = False
        df_partial = df_partial[df_partial['all_char'] == True]
        self.is_web(df_partial)
        df = df.join(df_partial['is_web'])

        df_partial = df.copy()
        df_partial[' '] = False
        df_partial = df_partial[df_partial['all_char'] == True]
        self.is_dni(df_partial)
        df = df.join(df_partial['is_dni'])

        df_partial = df.copy()
        df_partial['num_words'] = False
        df_partial = df_partial[df_partial['all_char'] == True]
        self.count_words(df_partial)
        df_partial['num_words'].apply(lambda x: int(x))
        df_partial['num_words'] = df_partial['num_words'].astype('int')
        df = df.join(df_partial['num_words'])

        print(df.dtypes)

        return self.export_csv(df)
        '''


if __name__ == '__main__':
    df = pd.read_csv('../Input/provisional_dummy_nodup.csv', sep=';', quoting=csv.QUOTE_ALL, keep_default_na=False)
    feature = feature_extraction()
    feature.run_feature(df)
