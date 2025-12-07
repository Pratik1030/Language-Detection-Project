from pickle import load
import re

f = open("LangModel2.pkl","rb")
model = load(f)
f.close()

f = open("CountVectorizer2","rb")
cv = load(f)
f.close()

def clean_function(text):
    text = re.sub(r'<[^>]+>', ' ', text)             # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|@\S+', ' ', text) # Remove URLs and emails
    text = re.sub(r'\d+', ' ', text)                 # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()         # Normalize spaces
    return text

text = input("Enter text: ")
clean_text = clean_function(text)
vector_text = cv.transform([clean_text])

result = model.predict(vector_text)
print("The lanuage is",result[0])