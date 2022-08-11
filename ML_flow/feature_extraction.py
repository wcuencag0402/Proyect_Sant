import pandas as pd
import numpy as np
import re
import csv

class feature_extraction:

    def __init__(self):

        self.dni_regex = "([a-z]|[A-Z]|[0-9])[0-9]{7}([a-z]|[A-Z]|[0-9])"
        self.email_regex = '^(.+\@.+\..+)$'
        self.movil_regex = '(\+34|0034|34)?[ -]*(6|7)[ -]*([0-9][ -]*){8}'
        self.telephone_regex = "(\+34|0034|34)?[ -]*(8|9)[ -]*([0-9][ -]*){8}"
        self.web_regex = '^(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)' \
                         '( [a-zA-Z0-9\-\.\?\,\’\/\\\+&amp;%\$#_]*)?$'
        self.datetime_regex = '^\d{2,5}[-/]\d{2}[-/]\d{2,5}$'
        self.val_cnae93 = list(pd.read_csv('../Input/CNAE93.csv')['Clase'])
        self.DIGITO_CONTROL = "TRWAGMYFPDXBNJZSQVHLCKE"

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
        df_cnae = self.all_int(df)
        for i in df_cnae['all_int']:
            if i== True:
                df_cnae['is_cnae'] = df_cnae['FIELD'].apply(lambda x: x in self.val_cnae93)
        return df_cnae

    def run_feature(self, df):
        '''
        Main class process
        @param origin_df : dataframe initial
        @return None
        '''
        
        return 

    

if __name__ == '__main__':
    df = pd.read_csv('../Input/provisional_dummy_nodup.csv', sep=';', quoting=csv.QUOTE_ALL, 
    keep_default_na=False)
    feature = feature_extraction()
    x=feature.is_cnae(df)
    x.to_csv('ss.csv')
    print(x)