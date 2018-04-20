Data processing for supervised machine learning using scikitlearn library: 

Used a sample small dataset of countries, age, salary and whether they would buy a certain product or not, from Udemy ML tutorial of countries and performed Data preprocessing.

STEPS:

1. Import data from CSV file by setting work environment and directory

2. Finding missing values in the dataset by importing Scikit learn libraries - Imputer object, 
   To calculate mean for missing 'NaN' values and complete the data set.(Median or other values can also be chosen)

3. Categorizing data using Label encoder class to give labels to Countries and the output Yes or no.
   Further using OneHotEncoder to avoid confusion between dummy encoded labels and their actual degree For Ex: Labels for countries 	      are not to be mistaken, by their rank or frequency, hence, bit encoding three countries into three columns.

4. Splitting the dataset into Training and Test set, using crossvalidation sublibrary and train_test_split class.
   Before applying any machine learning algorithm, it's must to keep a training dataset and a test dataset.
   To compare the accuracy on both and if the accuracies for test and training are different, that means our training model isn't working    as efficiently as it should be.
   Selected 20% of samples for testing set and remaining 80% for training set.
   As always the better the learning, better the prediction.
	
5.  Scaling_Independent variables using StandardScalar class.
    If indepedent input variables in the dataset are not on equal scales. It will be difficult to find how one varies from other. In this     dataset age and salary are not on equal scale and 
	
