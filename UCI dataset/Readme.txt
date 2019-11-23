You can use the following matlab code to read the datasets:

1) For duke dataset. 

 [label_vector, instance_matrix] = libsvmread('duke');   %libsvmread is a commend from lib svm
 data1=full(instance_matrix);
data=[label_vector data1];
Training=data(1:22,:);
Testing=data(23:44,:);


2) leu dataset.

[label_vector, instance_matrix] = libsvmread('leu');
data1=full(instance_matrix);
instance_matrix=mapminmax(data1);
label_vector=mapminmax(label_vector',0,1);
data=[label_vector' instance_matrix];
Training=data;
[label_vector, instance_matrix] = libsvmread('leu.t');
data1=full(instance_matrix);
instance_matrix=mapminmax(data1);
label_vector=mapminmax(label_vector',0,1);
data=[label_vector' instance_matrix];
data=full(data);
Testing=data;