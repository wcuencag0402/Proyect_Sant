from logging import raiseExceptions
import re
import pandas as pd
import csv
from csv import Error

from configparser import ConfigParser


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
            df = pd.read_csv(input_file, sep=delimiter)
            print('File read with detected header and delimiter')
            print(df)
            return df
        else:
            raise Exception ("No headers provided")

    except(FileNotFoundError, PermissionError, TypeError) as f:
        if TypeError:
            print('Please provide a csv with delimiter')
        else:
            print(f)

    

if __name__ == "__main__":
    input_file = config['input_info']['input_file']
    read_input(input_file)
    
    