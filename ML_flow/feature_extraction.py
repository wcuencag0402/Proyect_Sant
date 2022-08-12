import pandas as pd
import numpy as np
import re
import csv
import configparser

from Utilities.utilities import Utilities


class feature_extraction:

    def __init__(self):

        self.file_name = self.get_config_data()

        self.dni_regex = "([a-z]|[A-Z]|[0-9])[0-9]{7}([a-z]|[A-Z]|[0-9])"
        self.email_regex = '^(.+\@.+\..+)$'
        self.movil_regex = '(\+34|0034|34)?[ -]*(6|7)[ -]*([0-9][ -]*){8}'
        self.telephone_regex = "(\+34|0034|34)?[ -]*(8|9)[ -]*([0-9][ -]*){8}"
        self.web_regex = '^(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)' \
                         '( [a-zA-Z0-9\-\.\?\,\’\/\\\+&amp;%\$#_]*)?$'
        self.datetime_regex = '^\d{2,5}[-/]\d{2}[-/]\d{2,5}$'
        self.val_cnae93 = list(pd.read_csv('../Input/CNAE93.csv')['Clase'])

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

    def len_column(self, df):
        '''
        Create an additional column with the length of the values of column 'FIELD
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['len_value'] = df['FIELD'].apply(lambda x: len(str(x)))
        return df

    def date_columns(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are strings
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['date_value'] = df['FIELD'].apply(lambda x: bool(re.findall(self.datetime_regex,str(x))))
        return df

    def all_int(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are numbers
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['all_int'] = df['FIELD'].apply(lambda x: str(x).isdigit())
        return df

    def all_char(self, df):
        '''
        Create an additional column checking whether the values of column 'FIELD' are chars
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['all_char'] = df['FIELD'].apply(lambda x: not str(x).isdigit())
        return df

    def is_cnae(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        val_cnae = list(map(str, self.val_cnae93))
        df['is_cnae'] = df['FIELD'].apply(lambda x: str(x) in val_cnae)
        return df

    def is_mobile(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        
        df['is_mobile'] = df['FIELD'].apply(lambda x: bool(re.findall(self.movil_regex ,str(x))))
        return df

    def is_tlf(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        
        df['is_tlf'] = df['FIELD'].apply(lambda x: bool(re.findall(self.telephone_regex ,str(x))))
        return df

    def is_email(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_email'] = df['FIELD'].apply(lambda x: bool(re.findall(self.email_regex ,str(x))))
        return df

    def is_dni(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are cnae
        @param df: dataframe
        @return list_values: dataframe
        '''
        df['is_dni'] = df['FIELD'].apply(lambda x: bool(re.findall(self.dni_regex ,str(x))))
        return df

    def is_web(self, df):
        '''
        Create an additional column checking whether the numerical values of column 'FIELD' are web
        @param df: dataframe
        @return list_values: dataframe
        '''
    
        df['is_web'] = df['FIELD'].apply(lambda x: bool(re.findall(self.web_regex ,str(x))))
        return df

    def count_words(self, df):
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
        df = self.len_column(df)
        df = self.date_columns(df)
        df = self.all_char(df)
        df = self.all_int(df)

        df_partial = df.copy()
        df_partial['is_cnae_'] = False
        df_partial = df_partial[df_partial['all_int']==True]
        self.is_cnae(df_partial)
        df = df.join(df_partial['is_cnae'])    

        df_partial = df.copy()
        df_partial['is_mobile'] = False
        df_partial = df_partial[df_partial['all_int']==True]
        self.is_mobile(df_partial)
        df = df.join(df_partial['is_mobile'])     

        df_partial = df.copy()
        df_partial['is_tlf'] = False
        df_partial = df_partial[df_partial['all_int']==True]
        self.is_tlf(df_partial)
        df = df.join(df_partial['is_tlf']) 

        df_partial = df.copy()
        df_partial['is_email'] = False
        df_partial = df_partial[df_partial['all_char']==True]
        self.is_email(df_partial)
        df = df.join(df_partial['is_email']) 

        df_partial = df.copy()
        df_partial['is_web'] = False
        df_partial = df_partial[df_partial['all_char']==True]
        self.is_web(df_partial)
        df = df.join(df_partial['is_web']) 

        df_partial = df.copy()
        df_partial['is_dni'] = False
        df_partial = df_partial[df_partial['all_char']==True]
        self.is_dni(df_partial)
        df = df.join(df_partial['is_dni']) 

        df_partial = df.copy()
        df_partial['num_words'] = False
        df_partial = df_partial[df_partial['all_char']==True]
        self.count_words(df_partial)
        df_partial['num_words'].apply(lambda x: int(x))
        df_partial['num_words'] = df_partial['num_words'].astype('int')
        df = df.join(df_partial['num_words']) 

        print(df.dtypes)
        

        return self.export_csv(df)

    

if __name__ == '__main__':
    df = pd.read_csv('../Input/provisional_dummy_nodup.csv', sep=';', quoting=csv.QUOTE_ALL, 
    keep_default_na=False)
    feature = feature_extraction()
    feature.run_feature(df)
