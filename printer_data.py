
import os
import plotters as mplt

"""
How to use printer_data: 
    download from drive the folder of csv files
    assing to 'FOLDER_DATA' variable the path to that folder
    assing to 'FOLDER_PLOT_DATA' variable the path to write png plot image
"""

FOLDER_DATA = "Data/Shallow/"
FOLDER_PLOT_DATA = FOLDER_DATA + "Plot/"

def main():

    files = os.listdir(FOLDER_DATA)

    for f in files:
        print(f)
        mplt.plot_file(f, FOLDER_DATA, FOLDER_PLOT_DATA)
        print("Done")


if __name__ == "__main__":
    main()