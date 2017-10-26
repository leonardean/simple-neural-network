import DataProcessor as dp

data_processor = dp.DataProcessor('Bike-Sharing-Dataset/hour.csv')
# data_processor.show_data()
data_processor.virtualize()
# data_processor.show_data()
data_processor.normalize()
data_processor.split()
a,b,c,d,e,f = data_processor.get_data()
