import pandas as pd
import numpy as np
import re
import csv
import configparser
#
from hermetrics.levenshtein import Levenshtein
from hermetrics.dice import Dice
from hermetrics.metric_comparator import MetricComparator

import itertools
import collections

from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering


class feature_extraction:

    def __init__(self):
        self.keys_dominios = {
            'idd_data': ['estudio','sexo','nombre','nacimiento','nacionalidad','persona'],
            'add': ['residencia','domicilio','direccion'],
            'contac': ['telephone','email','website','movil','dispositvo'],
            'legal_id': ['documento']
        }

        self.prueba = {
            'codigo': "COD",
            'persona': "PER[S]?",
            'residencia': 'RESI',
            'documento': 'DOC',
            "cnae93": "CNAE93",
            "cno": "CNO",
            "estudio": "ESTU[DIO]?",
            "trabajo": "TRAB[AJO]?",
            "idioma": "IDIO[MA]?",
            "domicilio": "DOMI[CILIO]?",
            "nacimiento": "NACIM[IENTO]?",
            "nacionalidad": "NACIO[NALIDAD]?"
        }

        self.file_name = self.get_config_data()
        self.email_regex = '^(.+\@.+\..+)$'
        self.movil_regex = '(\+34|0034|34)[ -]*(6|7)[ -]*([0-9][ -]*){8}'
        self.telephone_regex = "(\+34|0034|34)[ -]*(8|9)[ -]*([0-9][ -]*){8}"
        self.web_regex = '^(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)' \
                         '( [a-zA-Z0-9\-\.\?\,\’\/\\\+&amp;%\$#_]*)?$'
        self.datetime_regex = '^\d{2,5}[-/]\d{2}[-/]\d{2,5}$'
        self.char_regex = "^([a-zA-Z]+[\s-])*[a-zA-Z]+$"
        self.direct_regex = "^([a-zA-Z]+\s)+(.)*\d{5}$"

        #########################################
        ## Nuevos valores que se deben incluir ##
        #########################################
        self.name_regex = '^(?=.{3,36}$)[a-zñA-ZÑ](\s?[a-zñA-ZÑ])*$'
        self.surname_regex = '^(?=.{3,36}$)[a-zñA-ZÑ](\s?[a-zñA-ZÑ])*$'
        self.document_regex = "^(\d{8})([A-Z])$"
        self.document_regex_2 = "^([ABCDEFGHJKLMNPQRSUVW])(\d{7})([0-9A-J])$"
        self.document_regex_3 = "^[XYZ]\d{7,8}[A-Z]$"

        # Comparador de datos del proceso #
        self.mic = MetricComparator()

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
                        'num_words': 0, 'date': False,
                        'int': False, 'characters': False,
                        'movil': False, 'telefono': False,
                        'email': False, 'website': False,
                        'direccion': False,
                        'nombre': False, 'documento': False,
                        'hora': False
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
        df['date'] = df['FIELD'].apply(lambda x: bool(re.findall(self.datetime_regex, str(x))))
        match_columns = df[df['date']]['data_split'].unique()
        return match_columns

    def is_mobile(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['mobile'] = df['FIELD'].apply(lambda x: bool(re.findall(self.movil_regex, str(x))))
        match_columns = df[df['mobile']]['data_split'].unique()
        return match_columns

    def is_tlf(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['telefono'] = df['FIELD'].apply(lambda x: bool(re.findall(self.telephone_regex, str(x))))
        match_columns = df[df['telefono']]['data_split'].unique()
        return match_columns

    def is_email(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['email'] = df['FIELD'].apply(lambda x: bool(re.findall(self.email_regex, str(x))))
        match_columns = df[df['email']]['data_split'].unique()
        return match_columns

    def is_website(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are web
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['website'] = df['FIELD'].apply(lambda x: bool(re.findall(self.web_regex, str(x))))
        match_columns =  df[df['website']]['data_split'].unique()
        return match_columns

    def is_all_char(self, df):
        '''
        '''
        df['characters'] = df['FIELD'].apply(lambda x: bool(re.findall(self.char_regex, str(x))))
        match_columns = df[df['characters']]['data_split'].unique()
        return match_columns

    def all_int(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are numbers
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['int'] = df['FIELD'].apply(lambda x: str(x).isdigit())
        match_columns = df[df['int']]['data_split'].unique()
        return match_columns

    def is_direct(self, df):
        '''
        '''
        df['direccion'] = df['FIELD'].apply(lambda x: bool(re.findall(self.direct_regex, str(x))))
        match_columns = df[df['direccion']]['data_split'].unique()
        return match_columns

    def is_name(self, df):
        '''
        '''
        df['nombre'] = df['FIELD'].apply(lambda x: bool(re.findall(self.name_regex, str(x))))
        match_columns = df[df['nombre']]['data_split'].unique()
        return np.delete(match_columns, np.argwhere(match_columns == 'ITIPDOCU'))

    def is_documents(self, df):
        '''
        '''
        df['documento'] = df['FIELD'].apply(lambda x: bool(re
                            .findall(r"({}|{}|{})".format(self.document_regex,self.document_regex_2,self.document_regex_3), str(x))))
        match_columns = df[df['documento']]['data_split'].unique()
        return match_columns

    def is_hora(self, df):
        '''

        @param
        @return
        '''
        df['hora'] = np.where(((df['len'] == 6) & (df['int'])), True, False)
        match_columns = df[df['hora']]['data_split'].unique()
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

    ####################
    ### proces text ###
    ##################

    def name_keywords_matrix(self, df_matrix):
        '''
        @params
        @return
        '''
        df2 = df_matrix.dot(df_matrix.columns + ';').str.rstrip(';')
        return df2

    def export_csv(self, df):
        return df.to_csv(self.file_name, sep=';', index=False)

    def run_feature(self, df):
        '''
        Main class process
        @param origin_df : dataframe initial
        @return None
        '''
        df = self.init_feauture_columns(df)
        df = self.split_name_colum(df)
        x = list(df['data_split'].unique())
        ########################################
        # 1. Información -- data in columns ###
        #######################################
        df = self.len_column(df)
        df = self.count_words(df)

        DF_matrix_info = pd.DataFrame(data=0, index=x,
                            columns=['date', 'int', 'characters', 'movil',
                                    'telefono', 'email', 'website',
                                    'direccion', 'nombre','documento','hora'])

        list_match = self.date_columns(df)
        DF_matrix_info.loc[list_match, 'date'] = 1
        list_match = self.all_int(df)
        DF_matrix_info.loc[list_match, 'int'] = 1
        list_match = self.is_all_char(df)
        DF_matrix_info.loc[list_match, 'characters'] = 1
        list_match = self.is_mobile(df)
        DF_matrix_info.loc[list_match, 'movil'] = 1
        list_match = self.is_tlf(df)
        DF_matrix_info.loc[list_match, 'telefono'] = 1
        list_match = self.is_email(df)
        DF_matrix_info.loc[list_match, 'email'] = 1
        list_match = self.is_website(df)
        DF_matrix_info.loc[list_match, 'website'] = 1
        list_match = self.is_direct(df)
        DF_matrix_info.loc[list_match, 'direccion'] = 1
        list_match = self.is_name(df)
        DF_matrix_info.loc[list_match, 'nombre'] = 1
        list_match = self.is_documents(df)
        DF_matrix_info.loc[list_match, 'documento'] = 1
        list_match = self.is_hora(df)
        DF_matrix_info.loc[list_match, 'hora'] = 1

       #################################################
        # 2. Información -- data extract name columns ###
        #################################################
        DF_matrix_info_2 = pd.DataFrame(data=0, index=x,
                                        columns=self.prueba.keys())

        for key, value in self.prueba.items():
            R = re.compile(value)
            output = list(filter(R.findall, x))
            sim_wols3 = [item for item in output
                         if self.mic.similarity(item.lower(), key)['Dice'] >= 0.5 or
                         self.mic.similarity(item.lower(), key)['Jaro'] >= 0.6]
            DF_matrix_info_2.loc[sim_wols3, key] = 1

        ########################################################
        # 3. Información -- split_name from regex data CLASS ###
        ########################################################
        list_datas = []
        for colum_df in x:
            pp = []
            for name, reg in self.prueba.items():
                app_regex = re.split(reg, colum_df)
                result = list(np.setdiff1d(app_regex, [colum_df,'']))
                if len(result):
                    pp.append(result)
            list_join = list(itertools.chain.from_iterable(pp))
            info = [k for k, v in collections.Counter(list_join).items() if v >= len(pp)]
            #list_datas.append(  { colum_df:';'.join(info)})
            list_datas.append(dict(colum=colum_df,words=','.join(info)))

        ###########################
        # Generación dataframe s3 #
        ###########################
        df3 = pd.DataFrame(list_datas, index=x)['words']
        list_unique_val = [item for word in list(df3.unique())
                            for item in word.split(',') if word != '']
        # 2.Generate columns
        DF_matrix_info_3 = pd.DataFrame(data=0, index=x, columns=list_unique_val)
        # Add values in matrix_info #
        for key, value in df3.to_dict().items():
            for x in value.split(','):
                if x != '':
                    DF_matrix_info_3.loc[key,  x] = 1

        ###########################################
        # 2. Union de los datos del proceso total #
        ###########################################
        '''
        df1 = self.name_keywords_matrix(DF_matrix_info)
        df2 = self.name_keywords_matrix(DF_matrix_info_2)
        #
        data_all = pd.DataFrame(dict(s1=df1, s2=df2))
        #
        data_all = data_all.join(df3) 

        #########
        list_unique_val_s1= [item for word in list(data_all['s1'].unique())
                           for item in word.split(';') if word != '']
        list_unique_val_s2 = [item for word in list(data_all['s2'].unique()) 
                           for item in word.split(';') if word != '']

        all_columns_matrix = list_unique_val_s1 +list_unique_val_s2 + list_unique_val
        '''
        #DF_matrix_info_final = pd.concat([DF_matrix_info, DF_matrix_info_2,DF_matrix_info_3], axis=1)
        DF_matrix_info_final = pd.concat([DF_matrix_info, DF_matrix_info_2], axis=1)

        ##################################################
        # 4. VERIFICACIÓN DE KEY_VALUES IN PROCESS TEXTO #
        ##################################################

        df_validate = pd.DataFrame(self.name_keywords_matrix(DF_matrix_info_final),
                                     columns=['colums'])

        for key, value in self.keys_dominios.items():
            DF_matrix_info_final[key] = df_validate['colums'].apply( lambda x: 1
                                        if list(set(x.lower().split(';')) & set(value)) else 0)

        DF_matrix_info_final.drop(['int', 'characters'], axis=1, inplace=True)

        '''
        ## Modelos del proceso ##
        clusterer_subsub_domain = AgglomerativeClustering(n_clusters=6,
                                        linkage="average", affinity="euclidean")
        clusterer_domain = AgglomerativeClustering(n_clusters=3,
                                            linkage="average", affinity="euclidean")
        clusters_subsubdomain = clusterer_subsub_domain.fit_predict(DF_matrix_info_final.to_numpy())
        clusters_domain = clusterer_domain(DF_matrix_info_final.to_numpy())
        '''
        return DF_matrix_info_final

if __name__ == '__main__':
    df = pd.read_csv('../Input/provisional_dummy_nodup_v3.csv', sep=';', quoting=csv.QUOTE_ALL, keep_default_na=False)
    feature = feature_extraction()
    feature.run_feature(df)
