# msds
Machine Learning Projects

Classification of the EuroSAT dataset provided here: https://github.com/phelber/eurosat

This project tackles image Classification of the EuroSAT dataset for land use and land cover. Using a convolutional neural network built in TensorFlow, 64x64 RGB images are classified into 10 different classes. Of the 27000 original images, 75% (20250 images) were used for training, 20% (5400 images) were used for validation during training, and 5% (1350 images) were held out for testing. The neural network uses 4 Convolution layers, 4 MaxPooling layers, and 3 fully connected layers to perform the classification. The training images were slightly rotated, zoomed in, and/or flipped horizontally between each epoch of training using the preprocessing layers in Keras. Dropout layers were included between the dense layers, as well as before the final convolution layer for regularization. 


The training time for the model was ~8 minutes on a nVidia GTX 1060 6GB GPU. The final model achieved a held out test set accuracy of 94-95% over different runs. 


**References**

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

[2] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.
