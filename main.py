from logging import raiseExceptions
import re
import pandas as pd
import csv
from csv import Error

from configparser import ConfigParser
from ML_flow.clean_data import CleanData
from ML_flow.general_reports import report_quality


file = './Config/config.ini'
config = ConfigParser()
config.read(file)

def check_input(input_file):
    '''
    def: Checks if input file has header and retrieves the delimiter
    params: dataframe
    return: has_header(bool), delimiter(str)
    '''

    #TODO capturar exception para una columna
    has_header = False
    delimiter = None
    try:
        with open(input_file, 'r') as csvfile:
            sample = csvfile.read(64)
            has_header = csv.Sniffer().has_header(sample)
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
            print('Header and delimiter detected')
    except(Error):
        raise Exception ('No delimiter detected')
    
    return has_header, delimiter
    

def read_input(input_file):
    # causistica varios ficheros
    '''
    def: reads input file
    params: input file (.csv)
    return: dataframe
    '''
    #TODO por ahora lo dejamos en csv que es con lo que estamos trabajando, ampliar a excel?
    #fichero config, separador y tipo de fichero .csv .xls .xlsx (llamar funcion para leer excel, y ver si 
    # sniffer es capaz de extraer separador)
    try:
        has_header, delimiter = check_input(input_file)

        if has_header:
            df = pd.read_csv(input_file, sep=delimiter, quoting=csv.QUOTE_ALL, keep_default_na=False)
            print('File read with detected header and delimiter')

            return df
        else:
            raise Exception ("No headers provided")

    except(FileNotFoundError, PermissionError, TypeError) as f:
        if TypeError:
            print('Please provide a csv with delimiter')
        else:
            print(f)

def transpose(dataframe, transpose_bool):
    '''
    def: takes column names and adds a column 'CLASS' filled with the column name, changes 
        column names to 'FIELD' and concats dataframe
    params: dataframe
    return: dataframe
    '''
    if eval(transpose_bool):
        df = pd.DataFrame()
        lista_df = []
        for col in dataframe:
            df['CLASS'] = col
            df['FIELD'] = dataframe[col]
            lista_df.append(df[['FIELD', 'CLASS']])
        concat_df = pd.concat(lista_df, axis=0).reset_index(drop=True)
        return concat_df
    else:
        df = dataframe
        return df

    

if __name__ == "__main__":
    input_file = config['input_info']['input_file']
    transpose_bool = config['input_info']['clean_transpose'] 

    read_df = read_input(input_file)

    report = report_quality(read_df)
    reporte = report.run_report(read_df)

    dataCleaner = CleanData(read_df)
    newdf = dataCleaner.run_clean()  

    transpose_df = transpose(newdf, transpose_bool)
    print(transpose_df)
