# DBScan_Python
Fast DBScan implementation


This algorithem is manipulating large amounts of data and 
taking advantage of multithreding to achive a faster implementation of the DBScan algorithem.

The Test data will be added here and the amount of threds should vary acording to youre personal computer.

at line 97  -  

    pointcount = get_pointcount(Eps, data, 20)
    
the number 20 indicates the number of threads. 


To run it on your cmputer check that the data filepath is changed as well.
you will need to change line 166 - 

    d = np.loadtxt("/Users/rafaelelkoby/Desktop/DBScan_Python/data_1_3.txt", delimiter = ",")
    
Where the file path should be the path to the data file. 


the data is 100,000 vectors of th 100th dimension. 
