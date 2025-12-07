import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from pickle import dump


data = pd.read_csv("Code_0.csv")

def clean_function(text):
    text = re.sub(r'<[^>]+>', ' ', text)             # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|@\S+', ' ', text) # Remove URLs and emails
    text = re.sub(r'\d+', ' ', text)                 # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()         # Normalize spaces
    return text

data["clean_text"] = data["Text"].apply(clean_function)

cv = CountVectorizer()
vector = cv.fit_transform(data["clean_text"])

features = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
target = data["Language"]

x_train,x_test,y_train,y_test = train_test_split(features.values,target,random_state=0)

model = MultinomialNB()
model.fit(x_train,y_train)

cr = classification_report(y_test,model.predict(x_test))
print(cr)

f = open("LangModel2.pkl","wb")
dump(model,f)
f.close()

f = open("CountVectorizer2","wb")
dump(cv,f)
f.close()

print("Done")