# This Repository contains the implementation of StockGAT, i.e. Graph Attention Networks for node classification of financial knowledge graphs

## Instructions for running the project

 -  The project requires python3.7; so please ensure you're working on this version of python
 -  Navigate to the current folder; where readme.md is saved 
 -  Install the required dependencies using the command: pip3.7 install -r requirements.txt
 -  To train the StockGAT model run the command: python3.7 train_stock_gat.py   
    This script trains and saves the best model
 -  To evaluated the performance of the saved model run the command: python3.7 test_stock_gat.py
    This script loads the trained model and evaluates the performance on the test set
 -  To evaluate the performance of the baseline models run the command: python3.7 baselines.py
    This script provides stock-wise accuracy metrics as well as the overall classification report of all stocks
 -  The model parameters are saved in the file gat_parameters.py
    The parameters can be altered in this file to run a new configuration of the StockGAT model



## This project constructs a financial knowledge graph from fundamental and technical features using angular correlation distance for the adjacency matrix 

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/118315635-c31cd080-b4c3-11eb-97f5-17ef45694d9b.png"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/118315746-e34c8f80-b4c3-11eb-9134-1b91dce1811b.png"></p>

