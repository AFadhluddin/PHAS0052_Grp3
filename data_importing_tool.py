import os
import glob
import pandas as pd
import numpy as np

def csv_to_dataframe(path):
    """
    Input: the path of the folder including the .csv files, should be mentioned with " " (should be a string)
    (recommand: for each set of data, use a independent folder)
    The function will import all the .csv files in the folder mentioned by the path while excluding
    'areaType', 'areaCode', 'areaName' columns and set 'date' as the index
    Output: a dataframe, df
    """
    # setting the path to the folder containing the .csv files
    # the only thing you need to do here is to put all the files belonging to the same dataset in the same folder which you reference here
    os.chdir(path)

    # chose only .csv files
    file_extension = ".csv"
    all_filenames = [i for i in glob.glob(f"*{file_extension}")]

    # concat is for merging the different csv files where axis=1 means horizontally, (default: axis=0 vertically)
    # index set as 'date' column so the other data will be merged according to the date
    # header = 0 so the first row indicate the name of the column
    df_original = pd.concat([pd.read_csv(file, usecols=lambda col: col not in ['areaType', 'areaCode', 'areaName'], index_col='date', header=0)
                            for file in all_filenames], axis=1)

    # since the date in the file starts from the latest date, this reverse the dataframe and start from the very first day
    df = df_original.iloc[::-1]

    return df


def parameter_importer(path, file_name):
    """
    Input: path of the folder and the name of the csv file
    Output: a dataframe containing the parameters found by data group
    """
    
    #setting the path to the folder containing the .csv files
    #the only thing you need to do here is to put all the files belonging to the same dataset in the same folder which you reference here
    dirName = path 
    dirName_feats = dirName + file_name

    #index set as 'Age' column so the other data will be merged according to the date
    #header = 0 so the first row indicate the name of the column
    df_parameters = pd.read_csv(dirName_feats, index_col='Age', header=0)


    #this is to form a new column 'percentage_population' that shows the percentage of the population in certain age range, in case you need it
    df_parameters['percentage_population'] = df_parameters['populationUK']/df_parameters['populationUK'].sum()

    return df_parameters


def column_extractor(df, header):
    """
    Input: a dataframe, df, and the header of the column
    header should be mentioned with ' ' (should be a string)
    The function convert the column under a "header" in a dataframe "df" to a numpy array
    Output: a numpy array
    """

    array = df[header].to_numpy()

    return array
