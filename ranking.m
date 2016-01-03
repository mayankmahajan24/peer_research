import FEAST.*

data = csvread('all_combined.csv', 1, 4);

labels = round(rand(length(data),1)) ;

%JMI
jmi_indeces = feast('jmi',size(data,2),data,labels);

%MRMR
mrmr_indeces = feast('mrmr',size(data,2),data,labels);

disp(jmi_indeces);