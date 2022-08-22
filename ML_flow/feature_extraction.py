import pandas as pd
import numpy as np
import re
import csv
import configparser


class feature_extraction:

    def __init__(self):
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
                        'is_direc':False
                        }
        # df = df.assign(**dict_)
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
        return df

    def is_mobile(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['is_mobile'] = df['FIELD'].apply(lambda x: bool(re.findall(self.movil_regex, str(x))))
        return df

    def is_tlf(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['is_tlf'] = df['FIELD'].apply(lambda x: bool(re.findall(self.telephone_regex, str(x))))
        return df

    def is_email(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_email'] = df['FIELD'].apply(lambda x: bool(re.findall(self.email_regex, str(x))))
        return df

    def is_doc_identificate(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_doc_identificate'] = df['FIELD'].apply(lambda x: bool(re.findall(self.dni_regex, str(x))))
        return df

    def is_website(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are web
        @param df: dataframe
        @return list_values: dataframe
        '''

        df['is_website'] = df['FIELD'].apply(lambda x: bool(re.findall(self.web_regex, str(x))))
        return df

    def is_all_char(self, df):
        '''
        '''
        df['is_all_characters'] = df['FIELD'].apply(lambda x: bool(re.findall(self.char_regex, str(x))))
        return df

    def all_int(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are numbers
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_all_int'] = df['FIELD'].apply(lambda x: str(x).isdigit())
        return df

    def is_direct(self, df):
        '''
        '''
        df['is_direc'] = df['FIELD'].apply(lambda x: bool(re.findall(self.direct_regex, str(x))))
        return df

    ############################
    ## data from column class ##
    ############################

    def split_name_colum(self, df):
        '''
        @param
        @return
        '''
        df['data_split'] = df['CLASS'].apply(lambda x: ','.join([i for i in x.split('_')]))
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

    def all_char_old(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are chars
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['all_char'] = df['FIELD'].apply(lambda x: not str(x).isdigit())
        return df

    def is_cnae_old(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        pass

    def count_words_old(self, df):
        '''
        Create an additional column checking the number of words of char values
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['num_words'] = df['FIELD'].apply(lambda x: int(len(x.split(' '))))
        return df

    def export_csv(self, df):
        return df.to_csv(self.file_name, sep=';', index=False)

    def run_feature(self, df):
        '''
        Main class process
        @param origin_df : dataframe initial
        @return None
        '''
        df = self.init_feauture_columns(df)
        # 1.
        df = self.len_column(df)
        df = self.count_words(df)
        df = self.date_columns(df)
        df = self.all_int(df)
        df = self.is_all_char(df)
        df = self.is_mobile(df)
        df = self.is_tlf(df)
        df = self.is_email(df)
        df = self.is_doc_identificate(df)
        df = self.is_website(df)
        df = self.is_direct(df)

        # 2.
        df = self.split_name_colum(df)
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
