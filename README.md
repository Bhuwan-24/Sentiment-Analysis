dataset : IMDB rating (source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

model : linear SVM

This project uses TFidVectrorizer to convert text IMDB rating to vectors. While converting texts into vector the arrays remains sparse so no further pandas function were applied. The sparse array is best in terms of speed for model training and testing.

In the case of visualization the sparse arrays were converted into dense array. PCA was used to convert higher dimensional data into 2D. From the 2D view the data looked unseperable but in terms of higher dimension SVM will be able to classifiy them.

SVM is sensible in case of c value(penalty). The c value determines the number of penalty applied for each mistake. In case of soft margin finding the best c value is very important so GridSearch technique is applied in order to find optimal c value among given. The model is trained on train values of x and y then evaluated using classification report, the result was better than expected.
