import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import re
import nltk
from imblearn.over_sampling import SMOTE
from collections import Counter

dataset = pd.read_csv(r".\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
print('Original dataset samples:', Counter(dataset['Liked']))

# Apply SMOTE
smote = SMOTE(random_state=42)
# Reshape for SMOTE
X_temp = np.arange(len(dataset)).reshape(-1, 1)  # Using indices as temporary features
y = dataset['Liked'].values
# Fit SMOTE
X_res, y_res = smote.fit_resample(X_temp, y)
# Create balanced dataset
balanced_dataset = dataset.iloc[X_res.ravel()].copy()
balanced_dataset['Liked'] = y_res
# Print balanced class distribution
print('Balanced dataset samples:', Counter(balanced_dataset['Liked']))
# Update dataset for further processing
dataset = balanced_dataset.copy()

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
vectorizer=CountVectorizer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""## XGBoost"""
from xgboost import XGBClassifier
classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

import joblib
joblib.dump(classifier, 'classifier.pkl')
joblib.dump(cv, 'vectorizer.pkl')
