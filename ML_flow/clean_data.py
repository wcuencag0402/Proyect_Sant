# Dictionaries for replacement

replacement_dic = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n', 'ç':'c', '-': ' ', 'à': 'a',
              'è':'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'}
domains_spain = ['es', 'com', 'net', 'eu', 'org', 'cat', 'info', 'biz', 'gal', 'eus']

# Changing format to datetype

def change_datetime(df, threshold):

    valor = df.shape[0] * threshold

    for col in df:
        df[str(col) + '_check'] = df[col].apply(lambda x: bool(re.findall('^\d{2,5}[-/]\d{2}[-/]\d{2,5}$',str(x))))
        filter_cols = df.filter(regex='_check')

    df_ = filter_cols.apply(pd.value_counts)
    n_true = df_.loc[True]
    n_true = n_true.dropna()
    n_true = n_true[n_true > valor]

    change_columns = ['_'.join(i.split('_')[:-1]) for i in n_true.index]

    for col in n_true.index:
        df = df.loc[df[col], :]

    df[change_columns] = df[change_columns].apply(pd.to_datetime) 

    df = df.drop(columns=filter_cols, axis=1)    

    return df

# Replace chars ONLY if dtype is object. This method returns in lowercase

def replace_chars(dataframe, column_name):
    # Take only string columns (object)
    if dataframe[column_name].dtype == 'object':
        
    # Check column is not an address or a website
        if any(dataframe[column_name][0].endswith(x) for x in domains_spain):
            return dataframe[column_name]
        
        for key in replacement_dic:
            dataframe[column_name] = dataframe[column_name].str.lower().str.replace(key,replacement_dic[key])
    return dataframe[column_name]