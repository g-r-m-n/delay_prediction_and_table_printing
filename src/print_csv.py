# -*- coding: utf-8 -*-
import pandas as pd  # used for handeling the table object.
from tabulate import tabulate # used for formatted printing of the table.  

def main():
    # get the file name and path input:
    file_path = input("Enter file name and path (default: C:/DEV/delay_prediction_and_table_printing/src/input/personen.csv)\n")
    # determine the default file name and path:
    if file_path == "":
        file_path = "C:/DEV/delay_prediction_and_table_printing/src/input/personen.csv"
    
    print( "The entered file name and path is the following:  "+file_path+'\n')
    
    # Read the dataframe:
    df = pd.read_csv(file_path,  sep=";", encoding='utf-8', )
    position    = 0  # determine the start position
    next_input  = '' # initialize the next_input variable
    page_length = 10 # determine the page length
    
    # continue to show the output until exit:
    while next_input != 'E':
        # determine the position  of the last row of the subtable to be printed:
        end_postion = min(position+page_length,len(df))-1
        # print the output at the current position and page length:
        print(tabulate(df.loc[position:end_postion,:], tablefmt="orgtbl", headers="keys", showindex="no"))
        # get the input for the next step:
        next_input = input("F)irst page, P)revious page, N)ext page, L)ast page, E)xit\n")
        # default input:
        if next_input == "": 
            next_input = "N"
        # go as next to the first page:
        if next_input == "F":
            position = 0
        # go as next to the previous page:    
        if next_input == "P":
            position -= page_length  
            position  = max(0,position)
        # go as next to the next page:    
        if next_input == "N":
            position += page_length  
            position  = min(position,len(df)-1)    
        # go as next to the last page:    
        if next_input == "L":
            position  = len(df) -  page_length 
              

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()

