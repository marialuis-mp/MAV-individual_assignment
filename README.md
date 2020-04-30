# MAV-individual_assignment

Read.me

This code has two main files:
1. Neural_network_YOLO: pre-processes data, builds and trains the neural 
network (NN)
2. Test_YOLO_NN: Tests the neural network, shows detection on training data 
and test data, evaluates the detection frequency, calculates and plots the
ROC curves

There are two modes to run the program:
1. There is an additional .mat file called NN_YOLO, which contains the already 
trained NN used. One can use only the 2nd file ("Test_YOLO_NN") which loads 
the trained network, avoiding the time to train the NN again.
2. If one desires to train the NN again, it is possible to do so, by running the
first file, which will overwrite the .mat file, and only then the second file. 
Note: The first file should be run in the same directory where the images from 
the data_set and the "corners.csv" are
