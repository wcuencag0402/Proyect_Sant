import re
import pandas as pd
import csv

from configparser import ConfigParser

#TODO funcion que lea las variables del fichero config
#TODO caso de una columna

file = './Config/config.ini'
config = ConfigParser()
config.read(file)

def check_input(input_file):
    """
    Checks if input file has header and retrieves the delimiter
    """
    with open(input_file, 'r') as csvfile:
        sample = csvfile.read(64)
        has_header = csv.Sniffer().has_header(sample)#esto una funcion si tiene cabecera

        delimiter = csv.Sniffer().sniff(sample)#esto una funcion encontrar delimitador
        #try except no encuentra delimitador
        
    
    return has_header, delimiter
    

def read_input(input_file):
    #TODO por ahora lo dejamos en csv que es con lo que estamos trabajando, ampliar a excel?
    #fichero config, separador y tipo de fichero .csv .xls .xlsx (llamar funcion para leer excel, y ver si 
    # sniffer es capaz de extraer separador)
    has_header, delimiter = check_input(config['read'][input_file])

    if has_header:
        try:
            df = pd.read_csv(input_file, sep=delimiter)
            print('1111')
        except(FileNotFoundError, PermissionError) as f:
            print(f)

        return df
    else:
        print('Please provide input with headers')

    

if __name__ == "__main__":
    print(config.sections())
    #read_input(config['input_info']['input_file'])