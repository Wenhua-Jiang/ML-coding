import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



data = pd.read_csv('spam_ham_dataset.csv')

#split the dataset for training and testing purposes
X_train, X_test, Y_train, Y_test = train_test_split(data['text'],
data['label_num'])


#transform the text data into vectors using CountVectorizer
cv = CountVectorizer()
vectorizer = cv.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)



model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)




predictions = model.predict(vectorizer.transform(X_test))
print("Accuracy:", 100 * sum(predictions == Y_test) / len(predictions), '%')



model.predict(vectorizer.transform(
[
"Hello Mike, I can came across your profile on Indeed, are you available for a short chat over the weekend.",
])
)


# In[ ]:




