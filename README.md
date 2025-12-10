🌐 Language Detection ML Project

This project uses a Machine Learning model to automatically detect the language of a given text.
It currently supports the following languages:

English
Spanish
German
French
Italian
Dutch
Portuguese

The model is trained using a supervised learning approach with the help of the scikit-learn library.
A classification report was generated during evaluation, achieving an F1-score of 0.98, indicating highly accurate predictions.

This project uses Multinomial Naive Bayes, which is well-suited for text classification tasks.
Effective preprocessing techniques were applied to clean the text without altering the meaning of the language, ensuring high model performance.

Both the trained model and the CountVectorizer used during training are saved as .pkl files for easy loading and prediction.
📁 Project Structure

Language-Detection-Project/
│
├── model.pkl # Trained ML model
├── vectorizer.pkl # CountVectorizer used during training
├── predict.py # Script to load model + make predictions
├── train.py # (Optional) Code used to train the model
├── requirements.txt # Required Python dependencies
└── README.md # Project documentation

🧠 How It Works

This project follows a classic NLP + Machine Learning pipeline designed for efficient and accurate language detection:

1️⃣ Data Preprocessing
Raw text is cleaned using:
Lowercasing
Removing unnecessary characters
Normalizing text while preserving language characteristics
Preprocessing is intentionally minimal to avoid harming the identity of the language.

2️⃣ Feature Extraction with CountVectorizer
The cleaned text is transformed into a numerical representation using:
Bag-of-Words model (CountVectorizer)
Token frequency counts
Vocabulary learned from the training dataset
This step converts text into vectors that can be understood by ML models.

3️⃣ Model Training using Multinomial Naive Bayes
Multinomial Naive Bayes is used because:
It performs extremely well with word frequency data
It is fast and lightweight
It handles multi-class text classification effectively
The trained model achieved an F1-score of 0.98, demonstrating excellent performance.

4️⃣ Model Saving
Both the trained model and vectorizer are saved as .pkl files to ensure:
Easy loading
Consistent preprocessing
Same vocabulary during inference

5️⃣ Prediction
When a user provides a text input:
The vectorizer converts it to numerical features
The model predicts the most probable language
The predicted label is returned instantly

🚀 Future Improvements
🔹 1. Expand the number of supported languages
       Include widely spoken languages such as:
       Hindi
       Arabic
       Chinese
       Japanese
       Russian

🔹 2. Switch to more advanced NLP models
      Potential upgrades include:
      TF-IDF Vectorizer
      Logistic Regression / SVM
      LSTM or GRU-based neural networks
      Transformer-based models
      These may improve accuracy on more complex datasets.

🔹 3. Build a Web Interface
      Create a simple UI using:
      Flask
      FastAPI
      Streamlit
      Users can input text and get predictions instantly.

🔹 4. Deploy the Model
      Host the model on a cloud platform such as:
      Hugging Face Spaces
      Render
      Heroku
     AWS Lambda
     Deployment makes your model accessible to anyone.

🔹 5. Add Confidence Scores
      Enhance predictions by showing probability scores for each language.

🔹 6. Improve Dataset Quality
      Use:
      Larger datasets
      More balanced samples
      Augmentation techniques
      Better data leads to improved accuracy.


