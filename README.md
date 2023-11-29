# Product-Category-Classifier

This project is to train and evaluate a classifier that predicts the category from the product title, rating and review text from a fictitious e-commerce dataset. I only use a logistic regression model and encode the features by bi-gram model. I focus on the data integration steps, the code readability and the reproducibility and practicality.

To train and test the classifier, you can simply install all libraries in `requirements.txt`, and run the code `classifier.py` by

        python <your-file-path>/Product-Category-Classifier/classifier.py

And the program will show the log of preprocess-train-test process.

* The trained model will be saved in `/models` folder.
* Folder `/dataset` includes all data.
* File `Q2_notebook.ipynb` is a draft of preprocessing and training experiments, which is not a formal solution of this task.
