from gc import get_threshold
import pandas as pd
import regex as re
import configparser

class CleanData:

    def __init__(self, df):
        self.df = df
            # Dict definitions
        self.replacement_dic = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n', 'ç':'c', '-': ' ', 'à': 'a',
              'è':'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'}
        self.domains_spain = ['es', 'com', 'net', 'eu', 'org', 'cat', 'info', 'biz', 'gal', 'eus']
        self.threshold = self.get_threshold()

    def get_threshold(self):
        '''
        Def: get param threshold from configuration file
        param
        return: threshold param
        '''
        file = './Config/config.ini'
        config = configparser.ConfigParser()
        config.read(file)
        self.threshold = float(config['input_info']['threshold'])
        return self.threshold
    
    def change_datetime(self):
        '''
        Def: checks if a sufficcient amount of values meet the date format and if so, change dtype to datetime
        params:
        return: dataframe with changed dtype to datetime
        '''

        valor = self.df.shape[0] * self.threshold

        for col in self.df:
            self.df[str(col) + '_check'] = self.df[col].apply(lambda x: bool(re.findall('^\d{2,5}[-/]\d{2}[-/]\d{2,5}$',str(x))))
            filter_cols = self.df.filter(regex='_check')

        df_ = filter_cols.apply(pd.value_counts)
        n_true = df_.loc[True]
        n_true = n_true.dropna()
        n_true = n_true[n_true > valor]

        change_columns = ['_'.join(i.split('_')[:-1]) for i in n_true.index]

        for col in n_true.index:
            self.df = self.df.loc[self.df[col], :]

        self.df[change_columns] = self.df[change_columns].apply(pd.to_datetime) 

        self.df = self.df.drop(columns=filter_cols, axis=1)    

        return self.df


    def replace_chars(self):
        '''
        Def: runs function replace_char_by_column for every column in dataframe
        param
        return: dataframe with chars replaced
        '''
        for column in self.df:
            self.df[column] = self.replace_char_by_column(self.df, column)
        return self.df


    def replace_char_by_column(self, df, column_name):
        '''
        Def: checks is column is type object (string) and replaces its characters attending 
        to a dictionary. Also returns strings in lower case
        params: dataframe and column name, called only by function replace_chars
        return: column of dataframe with chars replaced
        '''
        # Take only string columns (object)
        if df[column_name].dtype == 'object':
            
        # Check column is not an address or a website
            if any(df[column_name][0].endswith(x) for x in self.domains_spain):
                return df[column_name]
            
            for key in self.replacement_dic:
                df[column_name] = df[column_name].str.lower().str.replace(key, self.replacement_dic[key])
        
        return df[column_name]

    def run_clean(self):
        '''
        Def: runs functions of replacement. This is the function which will be called
        params
        return: dataframe with chars replaced
        '''
        self.df = self.change_datetime()
        self.df = self.replace_chars()


if __name__ == '__main__':
    df = pd.read_csv('provisional_dummy.csv')
    dataCleaner = CleanData(df)
    newdf = dataCleaner.run_clean()

    newdf.sample(n=20)
