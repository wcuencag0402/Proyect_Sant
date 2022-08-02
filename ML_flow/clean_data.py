import pandas as pd
import regex as re

class CleanData:

    # Dict definitions
    replacement_dic = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n', 'ç':'c', '-': ' ', 'à': 'a',
              'è':'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'}
    domains_spain = ['es', 'com', 'net', 'eu', 'org', 'cat', 'info', 'biz', 'gal', 'eus']

    def __init__(self, df):
        self.df = df
    
    # Changing format to datetype
    def change_datetime(self, threshold = 0.85):

        valor = self.df.shape[0] * threshold

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

    # Replace chars ONLY if dtype is object. This method returns in lowercase

    def replace_chars(self):
        for column in self.df:
            self.df[column] = self.replace_char_by_column(self.df, column)
        return self.df


    def replace_char_by_column(self, df, column_name):
        # Take only string columns (object)
        if df[column_name].dtype == 'object':
            
        # Check column is not an address or a website
            if any(df[column_name][0].endswith(x) for x in self.domains_spain):
                return df[column_name]
            
            for key in self.replacement_dic:
                df[column_name] = df[column_name].str.lower().str.replace(key, self.replacement_dic[key])
        
        return df[column_name]


if __name__ == '__main__':
    df = pd.read_csv('provisional_dummy.csv')
    dataCleaner = CleanData(df)
    newdf = dataCleaner.change_datetime()
    dataCleaner = CleanData(newdf)
    newdf = dataCleaner.replace_chars()

    newdf.sample(n=20)
