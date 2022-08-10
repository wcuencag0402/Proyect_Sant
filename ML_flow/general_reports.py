import pandas as pd
import numpy as np
import re
import os.path
from configparser import ConfigParser
import openpyxl
import sys
from xlwt import Workbook
import csv
import configparser



class report_quality:

    def __init__(self, df):
        self.out_path = ''
        self.name_columns = ['column_name',
                            'total_values',
                            'null_values',
                            'unique_values',
                            'duplicate_values']
        self.output_path = self.get_output_path()

    def get_output_path(self):
        '''
        Def: get param threshold from configuration file
        param
        return: threshold param
        '''
        file = './Config/config.ini'
        config = configparser.ConfigParser()
        config.read(file)
        output_path = config['report_info']['path_output']
        return output_path
        
    def calculate_statistics(self, df, column):
        '''
        def: Calculate statistics form all columns in dataframes
        param: datframes
        return: dataframes
        '''

        output_path = self.output_path

        column_name = column
        total_values = len(df.index)
        nulls = self.calculate_nulls(df, column) 
        uniques = self.calculate_unique_values(df, column) 
        duplicates_df = self.calculate_duplicates(df, column)

        list_duplicates = []
        total_duplicates = 0

        dict_duplicates = {}
        dict_total= {}        

        for index in duplicates_df.index:
            if duplicates_df[index] > 1:
                total_duplicates += duplicates_df[index]
                list_duplicates.append(index)
                dict_duplicates[index] = duplicates_df[index]
                dict_total[duplicates_df.index.name] = dict_duplicates
        
        for k,v in dict_total.items():
            with open(output_path+str(k)+'.csv', 'w') as csvfile:
                fields = ['Value', 'Repeated']
                writer = csv.DictWriter(csvfile, delimiter=';', fieldnames= fields, 
                lineterminator='\n')
                writer.writeheader()

                for i,j in v.items():
                    writer.writerow({'Value':i, 'Repeated':j})
                 
        return [column_name, total_values, nulls, uniques, total_duplicates]
            

def run_report(self, df):
        '''
        def: Run data quality report
        params: dataframe
        return: Excel file
        '''
        
        # Create dataframe and fill it by rows
        reports_df = pd.DataFrame(columns=self.name_columns)
        
        for column in df:
            reports_df.loc[column] = self.calculate_statistics(df, column)
            
        vol_input, vol_output = self.check_volumetry(df, df)
        
        return self.export_to_xls(reports_df, vol_input, vol_output)
        
    def calculate_nulls(self, df, column_name):
        '''
        def: Get number of null or NaN values in a column
        params: dataframe, column of dataframe
        return: Number of NaN in column
        '''
        return df[column_name].isna().sum()

    def calculate_unique_values(self, df, column_name):
        '''
        def: Get number of unique values in a column ([A, B, B, A, C] will have 3 unique values: A, B and C)
        params: dataframe, column of dataframe
        return: Number of uniques in column
        '''
        return df[column_name].nunique()

    def calculate_duplicates(self, df, column_name):
        '''
        def: Get number of duplicate values in a column
        params: dataframe, column of dataframe
        return: Dataframe containing two columns: 1st is value duplicated and 2nd is how many times

        return df.pivot_table(index=[column_name], aggfunc='size')

    
    def check_volumetry(self, df_in, df_out):
        '''
        def: check if input data equals output data
        params: dataframe input, dataframe output
        return: Dataframe only true or false
        '''

        vol_input = f'{df_in.shape[1]} columns, {df_in.shape[0]} rows'
        vol_output = f'{df_out.shape[1]} columns, {df_out.shape[0]} rows'
        
        return vol_input, vol_output
    
    def export_to_xls(self, df_quality, vol_input, vol_output):
        '''
        def: Export dataframe to xls format (Excel)
        params: dataframe
        return: Dataframe in xls format
        '''
        report_name = 'quality_report.xlsx'
        
        wb = Workbook()
        sheet_quality = wb.add_sheet('Quality_validations')
        sheet_volumetry = wb.add_sheet('Volumetry')
                
        x = 0
        for name in self.name_columns:
            sheet_quality.write(0, x, name)
            x += 1

        i = 0

        for column in df_quality:
            j = 0
            for row in df_quality[column]:
                sheet_quality.write(j + 1, i, str(df_quality[column][j]))
                j += 1
            i += 1


        sheet_volumetry.write(0, 0, 'Volumetry details:')
        sheet_volumetry.write(1, 1, 'Input:')
        sheet_volumetry.write(1, 2, vol_input)
        sheet_volumetry.write(2, 1, 'Output:')
        sheet_volumetry.write(2, 2, vol_output)

        report_name = 'quality_report.xls'
        wb.save(os.path.join(self.out_path, report_name))
        print("Saved correctly")


class report_validation:

    def __init__(self):
        '''
        Constructor classes
        '''
        self.dict_operation = {
            "DATE": {"columns": ["FEC_CTE", "FEC_NACIM", "FEC_ALTSIS",
                                 "FEC_ULTACT", "FECH_INI", "FECH_FIN",
                                 "FEULTACT", "FECCONST"],
                     "operation": self.validate_date},
            "MOBILE_PHONE": {"columns": ['DIRELE'],
                             "operation": self.validate_mobilephone},
            "EMAIL": {"columns": ['DIRELE'],
                      "operation": self.validate_email},
            "WEB": {"columns": ['DIRELE'],
                    "operation": self.validate_web},
            "JOB_INDICATOR": {"columns": ['SWTRABJ'],
                              "operation": self.validate_job_indicator},
            "DOCUMENTATION": {"columns": ['COD_DOCUM'],
                              "operation": self.validate_documentation},
            "CNAE": {"columns": ['CNAE93'],
                     "operation": self.validate_cnae93},
            "TELEPHONE": {"columns": ['DIRELE'],
                          "operation": self.validate_telephone}
        }
        self.full_path_out = self.load_config()
        self.dni_regex = "([a-z]|[A-Z]|[0-9])[0-9]{7}([a-z]|[A-Z]|[0-9])"
        self.email_regex = '^(.+\@.+\..+)$'
        self.movil_regex = '(\+34|0034|34)?[ -]*(6|7)[ -]*([0-9][ -]*){8}'
        self.telephone_regex = "(\+34|0034|34)?[ -]*(8|9)[ -]*([0-9][ -]*){8}"
        self.web_regex = '^(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)' \
                         '( [a-zA-Z0-9\-\.\?\,\’\/\\\+&amp;%\$#_]*)?$'
        self.datetime_regex = '^\d{2,5}[-/]\d{2}[-/]\d{2,5}$'
        self.val_cnae93 = list(pd.read_csv('../Input/CNAE93.csv')['Clase'])
        self.DIGITO_CONTROL = "TRWAGMYFPDXBNJZSQVHLCKE"
        self.INVALIDOS = {"00000000T", "00000001R", "99999999R"}
        self.job_indicators = ('A', 'O', 'P')

    def load_config(self):
        '''
        Load report configuration
        @param None
        @return full_path: full path to save results
        '''
        config = ConfigParser()
        config.read('../Config/config.ini')
        return config['reports_info']['path_output'] + config['reports_info']['name_error_report']

    def validate_mobilephone(self, df, name):
        '''
        Validate the web value
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column] \
                .apply(lambda x: bool(re.findall(self.movil_regex, str(x))))
            false_count = (~df_cop[column + '_check']).sum()
            #self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        #
        return list_values

    def validate_email(self, df, name):
        '''
        Validate the email value
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column] \
                .apply(lambda x: bool(re.findall(self.email_regex, str(x))))
            false_count = (~df_cop[column + '_check']).sum()
            # self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        return list_values

    def validate_web(self, df, name):
        '''
        Validate the web value
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column] \
                .apply(lambda x: bool(re.findall(self.web_regex, str(x))))
            false_count = (~df_cop[column + '_check']).sum()
            #self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        #
        return list_values

    def validate_job_indicator(self, df, name):
        '''
        Validate the job_indicator value
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column].apply(lambda x: True if x in self.job_indicators else False)
            false_count = (~df_cop[column + '_check']).sum()
            # self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        #
        return list_values

    def validate_documentation(self, df, name):
        '''
        Validate the document value
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column] \
                .apply(lambda x: bool(re.findall(self.dni_regex, str(x))))
            false_count = (~df_cop[column + '_check']).sum()
            # self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        #
        return list_values

    def validate_cnae93(self, df, name):
        '''
        Validate the cnae value
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column].apply(lambda x: True if x in self.val_cnae93 else False)
            false_count = (~df_cop[column + '_check']).sum()
            # self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        #
        return list_values

    def validate_date(self, df, name):
        '''
        Validate the date format
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column] \
                .apply(lambda x: bool(re.findall(self.datetime_regex, str(x))))
            false_count = (~df_cop[column + '_check']).sum()
            # self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            # data_false = df_cop[[column]][~df_cop[column + '_check']]
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        #
        return list_values

    def validate_telephone(self, df, name):
        '''
        Validate the phone format
        @param df: dataframe
               name: operation name
        @return list_values: list of dictionary
        '''
        df_cop = df.copy()
        list_values = []
        for column in df_cop.columns:
            df_cop[column + '_check'] = df_cop[column] \
                .apply(lambda x: bool(re.findall(self.telephone_regex, str(x))))
            false_count = (~df_cop[column + '_check']).sum()
            # self.export_error(df_cop[[column]][~df_cop[column + '_check']])
            list_values.append(dict(name=name, colum_name=column, num_error=false_count))
        #
        return list_values

    def check_name_columns(self, df_columns, operate_columns):
        '''
        Extract the columns that are validated
        @param  df_columns: initial datraframe column list
                operate_columns: list of columns to validate
        @return exist: boolean to perform the process
                common_columns: list of common columns
        '''
        common_columns = df_columns.intersection(operate_columns)
        exist = True if len(common_columns) > 0 else False
        #
        return exist, common_columns

    def export_xls(self, list_data, total_data):
        '''
        Save previously calculated values to a file with format xls
        @param  list_data: list with previously calculated values
                total_data: total records
        @return None
        '''
        df_data = pd.DataFrame(list_data)
        if not df_data.empty:
            df_data['percent_error'] = (df_data['num_error'] / total_data) * 100
            df_data.to_excel(self.full_path_out, sheet_name='erroneous_report', index=False)

    def export_error(self, df):
        '''
        Save errors to a file with format csv
        @param df: dataframe with detected errors
        @return None
        '''
        if not df.empty:
            df.to_csv("name_column.csv", index=False)

    def run_report(self, origin_df):
        '''
        Main class process
        @param origin_df : dataframe initial
        @return None
        '''
        list_result = []
        total_data = origin_df.shape[0]
        for key, values in self.dict_operation.items():
            check_column, common_columns = self.check_name_columns(origin_df.columns, values['columns'])
            if check_column:
                partial_df = origin_df[common_columns]
                df_final = values['operation'](partial_df, key)
                list_result.extend(df_final)
            else:
                print("No se han detectado columnas a procesar")
        #
        self.export_xls(list_result, total_data)

if __name__ == '__main__':
    df = pd.read_csv('../Input/per_jdca_classes (1).csv', sep=';')
    report = report_validation()
    report.run_report(df)
