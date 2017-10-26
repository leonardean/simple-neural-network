# Simple Neural Network 

This is a python implementation of a simple neural network using vanila numpy. The neural network contains three layers: an input layer, a hidden layer, and an output layer.The data used for the network comes from a bike sharing company. It contains bike sharing counts aggregated on hourly basis. Records: 17379 hours. This neural network can be used to predict the bike sharing counts in the future given relevant feature inputs.

<img src="https://leonardean.cc/content/images/2017/10/neural_network.png" width=300px>

# How to play with it
## Viewing data
You can check the data structure of each stage of data processing right from command line:
### Original data
```
>>> from DataProcessor import DataProcessor as dp
>>> data_processor = dp('Bike-Sharing-Dataset/hour.csv')
>>> data_processor.show_data()
   instant      dteday  season  yr  mnth  hr  holiday  weekday  workingday  \
0        1  2011-01-01       1   0     1   0        0        6           0
1        2  2011-01-01       1   0     1   1        0        6           0
2        3  2011-01-01       1   0     1   2        0        6           0
3        4  2011-01-01       1   0     1   3        0        6           0
4        5  2011-01-01       1   0     1   4        0        6           0

   weathersit  temp   atemp   hum  windspeed  casual  registered  cnt
0           1  0.24  0.2879  0.81        0.0       3          13   16
1           1  0.22  0.2727  0.80        0.0       8          32   40
2           1  0.22  0.2727  0.80        0.0       5          27   32
3           1  0.24  0.2879  0.75        0.0       3          10   13
4           1  0.24  0.2879  0.75        0.0       0           1    1
```
### Data after virtualization
```
>>> data_processor.virtualize()
>>> data_processor.show_data()
   yr  holiday  temp   hum  windspeed  casual  registered  cnt  season_1  \
0   0        0  0.24  0.81        0.0       3          13   16         1
1   0        0  0.22  0.80        0.0       8          32   40         1
2   0        0  0.22  0.80        0.0       5          27   32         1
3   0        0  0.24  0.75        0.0       3          10   13         1
4   0        0  0.24  0.75        0.0       0           1    1         1

   season_2    ...      hr_21  hr_22  hr_23  weekday_0  weekday_1  weekday_2  \
0         0    ...          0      0      0          0          0          0
1         0    ...          0      0      0          0          0          0
2         0    ...          0      0      0          0          0          0
3         0    ...          0      0      0          0          0          0
4         0    ...          0      0      0          0          0          0

   weekday_3  weekday_4  weekday_5  weekday_6
0          0          0          0          1
1          0          0          0          1
2          0          0          0          1
3          0          0          0          1
4          0          0          0          1

[5 rows x 59 columns]
```
### Data after normalization
```
>>> data_processor.normalize()
>>> data_processor.show_data()
   yr  holiday      temp       hum  windspeed    casual  registered       cnt  \
0   0        0 -1.334609  0.947345  -1.553844 -0.662736   -0.930162 -0.956312
1   0        0 -1.438475  0.895513  -1.553844 -0.561326   -0.804632 -0.823998
2   0        0 -1.438475  0.895513  -1.553844 -0.622172   -0.837666 -0.868103
3   0        0 -1.334609  0.636351  -1.553844 -0.662736   -0.949983 -0.972851
4   0        0 -1.334609  0.636351  -1.553844 -0.723582   -1.009445 -1.039008

   season_1  season_2    ...      hr_21  hr_22  hr_23  weekday_0  weekday_1  \
0         1         0    ...          0      0      0          0          0
1         1         0    ...          0      0      0          0          0
2         1         0    ...          0      0      0          0          0
3         1         0    ...          0      0      0          0          0
4         1         0    ...          0      0      0          0          0

   weekday_2  weekday_3  weekday_4  weekday_5  weekday_6
0          0          0          0          0          1
1          0          0          0          0          1
2          0          0          0          0          1
3          0          0          0          0          1
4          0          0          0          0          1

[5 rows x 59 columns]
```
## Training network
You can train the network by running:
```
python Train.py
```
But before that, you may want to edit some parameters for the network. This can be done by editing `networkConfig.json` :
```
{
  "iterations": 10000,
  "learning_rate": 0.1,
  "hidden_nodes": 7,
  "output_nodes": 1
}

```

You can see the losses (the lower, the better) of the trained network during training and how they change over time. After training, you will have:

 - a figure output showing losses over iteration
 <img src="https://leonardean.cc/content/images/2017/10/download-1.png" width=600px>
 - two 'npy' files will be generated as the outcome of network training. They will be used when running the network on test data.

## Running network
After training, the network can be run to predict bike sharing counts:
```
python Run.py
```
<img src="https://leonardean.cc/content/images/2017/10/download-2.png" width=600px>

