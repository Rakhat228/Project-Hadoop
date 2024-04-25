import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
from io import BytesIO



def classify():
    
    df_test = df_test.drop(df_test[df_test['text'] == '[Music]'].index)
    tfidf = pickle.load(open('data/vectorizer.pkl', 'rb'))
    model = pickle.load(open('data/model.pkl', 'rb'))

    X_test = tfidf.transform(df_test['text'])
    y_pred = model.predict(X_test)
    y_pred_1 = (y_pred > 0.9096).astype(int)
    df_test2 = df_test
    df_test2['is_sponsorship'] = y_pred_1
    'Final output'
    df_test2

    def down():
        def to_excel(df_test2):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            df_test2.to_excel(writer, index=False, sheet_name='Sheet1') 
            writer.save()
            processed_data = output.getvalue()
            return processed_data
        df_xlsx = to_excel(df_test2)
        st.download_button(label='📥 Press to download',
                                        data = df_xlsx ,
                                        file_name = 'Final output.xlsx')
    zzz = st.success('Successfully done!', icon="✅")
    return (zzz, down())


dataset = st.file_uploader("UPLOAD TEXT FILE", type = ['csv'])
if dataset is not None:
    df_test = pd.read_csv(dataset, engine='python', sep='\|\|', header=None, delimeter = ',', skiprows=1, names=["ID", "Text"]).set_index('ID')
    st.write(df_test)

st.button('Classify', on_click=classify, disabled=False)

