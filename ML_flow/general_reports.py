import pandas as pd
import numpy as np
import os.path
from xlwt import Workbook


class report_quality:
    def __init__(self, df):
        self.out_path = ''
        self.name_columns = ['column_name',
                            'total_values',
                            'null_values',
                            'unique_values',
                            'duplicate_values']
        
    def calculate_statistics(self, df, column):
        '''
        def: Calculate statistics form all columns in dataframes
        param: datframes
        return: dataframes
        '''
        
        column_name = column
        total_values = len(df.index)
        nulls = self.calculate_nulls(df, column)
        uniques = self.calculate_unique_values(df, column)
        duplicates_df = self.calculate_duplicates(df, column)

        list_duplicates = []
        total_duplicates = 0


        for index in duplicates_df.index:
            if duplicates_df[index] > 1:
                total_duplicates += duplicates_df[index]
                list_duplicates.append(index)
                    
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
            
        vol_input, vol_output, volumetry_result = self.check_volumetry(df, df)
        
        return self.export_to_xls(reports_df, vol_input, vol_output, volumetry_result)
        
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
        ''' 
        return df.pivot_table(columns=[column_name], aggfunc='size')
    
    def check_volumetry(self, df_in, df_out):
        '''
        def: check if input data equals output data
        params: dataframe input, dataframe output
        return: Dataframe only true or false
        '''
        vol_input = f'input:    {df_in.shape[1]} columns, {df_in.shape[0]} rows'
        vol_output = f'output:  {df_out.shape[1]} columns, {df_out.shape[0]} rows'
        
        return vol_input, vol_output, df_in.equals(df_out)
    
    def export_to_xls(self, df_quality, vol_input, vol_output, volumetry_result):
        '''
        def: Export dataframe to xls format (Excel)
        params: dataframe
        return: Dataframe in xls format
        ''' 
        report_name = 'quality_report.xlsx'
        
        wb = Workbook()
        sheet_quality = wb.add_sheet('Quality_validations')
        sheet_volumetry = wb.add_sheet('Volumetry')
        
        #sheet.write(row, col data, style)
        
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

        sheet_volumetry.write(0, 0, 'Volumetry is equal in input data and output data:')
        sheet_volumetry.write(1, 1, volumetry_result)
        sheet_volumetry.write(2, 1, vol_input)
        sheet_volumetry.write(3, 1, vol_output)

        report_name = 'quality_report.xls'
        wb.save(os.path.join(self.out_path, report_name))
        print("Saved correctly")

if __name__ == '__main__':
    df = pd.read_csv('provisional_dummy.csv')
    report = report_quality(df)
    report.run_report(df)