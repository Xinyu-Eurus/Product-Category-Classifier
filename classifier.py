import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from joblib import dump, load

from scipy.sparse import hstack
from scipy.sparse import vstack


"""
read tsv data and merge product data with reviews 
(left join, so that every row has a value of category, but might not have a value of review_text)
n is the number of pairs of tsv files, out_name is name of saved integrated new file
"""
def read_and_merge_data(n=4,out_name='all'):
    df_products = pd.read_csv('./dataset/products-data-0.tsv', sep='\t', header=None, names=['id', 'category','title'], index_col=0)
    df_reviews = pd.read_csv('./dataset/reviews-0.tsv', sep='\t', header=None, names=['id', 'rating','review_text'], index_col=0)
    print("products-data-0", df_products.shape, "reviews-0", df_reviews.shape)
    
    for i in range(1,n):
        df_1 = pd.read_csv('./dataset/products-data-'+str(i)+'.tsv', sep='\t', header=None, names=['id', 'category','title'], index_col=0)
        df_2 = pd.read_csv('./dataset/reviews-'+str(i)+'.tsv', sep='\t', header=None, names=['id', 'rating','review_text'], index_col=0)
        print("products-data-"+str(i), df_1.shape, "reviews-"+str(i), df_2.shape)
        df_products=pd.concat([df_products, df_1])
        df_reviews=pd.concat([df_reviews, df_2])
        # print("products-data", df_products.shape, "reviews", df_reviews.shape)
        
    df_all=df_products.merge(df_reviews, how='left', left_on='id', right_on='id')
    print("df_all", df_all.shape)
    #save the merged original data to a new csv file in case we need to retrain the model later
    df_all.to_csv('./dataset/'+out_name+'.csv')
    return df_all


"""
For data integration.
    1) Handle missing values
    2) Remove or correct outliers
    3) Transformation: vectorize text data
"""
def clean_preprocess_data(df_all):
    # fill the nan value as 'null'
    df_all.fillna(value='null', inplace=True)
    
    # check if there are category labels other than 'Kitchen' and 'Jewelry'
        # result: ['Kitchen' 'Jewelry' 'Ktchen']
    print("category items:",df_all["category"].drop_duplicates().to_numpy())
    # add a column 'label', 0 for Category of'Kitchen' or 'Ktchen', 1 for 'Jewelry' 
    # Define a lambda function for labeling
    label_function = lambda x: 0 if (x == 'Kitchen' or x == 'Ktchen') else 1 if (x == 'Jewelry') else -1
    # Apply the function to the 'category' column
    df_all['label'] = df_all['category'].apply(label_function)
    
    # if there are category labels other than 'Kitchen', 'Ktchen' or 'Jewelry', print them and delete the row
    # check if there are any rows with label of -1 and delete them
    rows_with_wrong_label = df_all[df_all['label'] == -1]
    if not rows_with_wrong_label.empty:
        print("Rows with label -1:")
        print(rows_with_wrong_label)
        # Remove the rows
        df_all = df_all[df_all['label'] != -1]
    else:
        print("All labels are valid. No rows with label -1 found.")

    # vectorize text of titles and reviews, and combine them
    vectorizer = CountVectorizer(ngram_range=(1,1), min_df=1, stop_words='english', lowercase=True)
    X_1 = vectorizer.fit_transform(df_all["title"].to_numpy())
    X_2 = vectorizer.fit_transform(df_all["review_text"].to_numpy())
    X = hstack([X_1, X_2]) # DO NOT use np.concatenate, X_1 and X_2 are sparse matrix
    print("\nShape of feature matrices:")
    print("X_title:", X_1.shape, " X_review:", X_2.shape, " combined X:", X.shape)

    y=df_all["label"].to_numpy()
    print("Shape of label array y:", y.shape)

    return X, y


"""
shuffle and split the dataset
X is feature matrix and y is label array
"""
def split_dataset(X, y, random_state=42):
    # First, combine X and y to shuffle them consistently
    Xy = vstack([X.T, y]).T

    # Shuffle and split the dataset into train (80%), and test (20%) sets
    Xy_train, Xy_test = train_test_split(Xy, test_size=0.2, random_state=random_state)

    # Separate the features and labels for each set
    X_train, y_train = Xy_train[:, :-1], Xy_train[:, -1].toarray().ravel()
    X_test, y_test = Xy_test[:, :-1], Xy_test[:, -1].toarray().ravel()

    return X_train, y_train, X_test, y_test
    

"""
train the model.
    1) Initialize the parameters of Logistic Regression model
    2) k-fold cross-validatio
    3) Train the model by full train set, and save the model
"""
def train(X_train, y_train, save_clf=True, K=5, C=1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42):
    # Initialize the Logistic Regression model
    clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=random_state)
    
    # Set up k-fold cross-validation (e.g., K=5)
    kfold = KFold(n_splits=K, shuffle=True, random_state=random_state)
    # Perform k-fold cross-validation and store the scores
    cv_scores = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    # Calculate the average and standard deviation of the cross-validation scores
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"Cross-Validation Scores(Accuracy): {cv_scores}")
    print(f"Average CV Accuracy: {cv_mean}")
    print(f"Standard Deviation of CV Accuracy: {cv_std}")

    # Train the model by full train set
    clf.fit(X_train, y_train)
    if save_clf:
        param_str=str(clf.C)+"-"+str(clf.penalty)+"-"+str(clf.solver)+"-"+str(clf.max_iter)+"-"+str(clf.random_state)
        dump(clf, './models/log_reg-'+param_str+'.model')
    return clf

"""
evaluate performance on the test set
"""
def test(clf, X_test, y_test):
    # predict labels
    y_pred = clf.predict(X_test)

    # calcullate the distribution of 2 label categories
    num_zeros = np.sum(y_test == 0)
    num_ones = np.sum(y_test == 1)
    print(f"Number of 0s (Kitchen) in y_test: {num_zeros}")
    print(f"Number of 1s (Jewelry) in y_test: {num_ones}")

    # calculate confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # The diagonal elements cm[0, 0] and cm[1, 1] give the correct predictions for each class
    accuracy_0 = cm[0, 0] / np.sum(cm[0, :])  # Accuracy for label 0 (Kitchen)
    accuracy_1 = cm[1, 1] / np.sum(cm[1, :])  # Accuracy for label 1 (Jewelry)
    print(f"Accuracy for label Kitchen: {accuracy_0}")
    print(f"Accuracy for label Jewelry: {accuracy_1}")

    #Overall Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Overall Accuracy: {accuracy}")



"""
main functiion
"""
if __name__ == "__main__":
    print("----------------------read and merge original data-----------------------")
    df_all = read_and_merge_data(n=4)
    print("\n----------------------clean and preprocess data-----------------------")
    X, y = clean_preprocess_data(df_all)
    X_train, y_train, X_test, y_test = split_dataset(X, y, random_state=42)
    print("\n----------------------train the model-----------------------")
    clf=train(X_train, y_train, save_clf=True, K=5, C=1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    print("\n----------------------test the model-----------------------")
    test(clf, X_test, y_test)