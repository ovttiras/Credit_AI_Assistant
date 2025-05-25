import streamlit as st
import pandas as pd
import pickle
import numpy as np
from catboost import Pool

# Загрузка модели
with open('German_Credit_gradient_boosting_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Получаем имена признаков из модели
feature_names = model.feature_names_

# Функция для создания DataFrame из введенных данных
def create_input_df():
    # Категориальные признаки и их возможные значения
    cat_features = {
        'checking_status': ['<0', '0<=X<200', '>=200', 'no checking'],
        'credit_history': ['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit'],
        'purpose': ['radio/tv', 'education', 'furniture/equipment', 'new car', 'used car', 'business', 'domestic appliance', 'repairs', 'other', 'retraining'],
        'savings_status': ['no known savings', '<100', '100<=X<500', '500<=X<1000', '>=1000'],
        'employment': ['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'],
        'personal_status': ['male single', 'female div/dep/mar', 'male mar/wid', 'male div/sep'],
        'other_parties': ['none', 'co applicant', 'guarantor'],
        'property_magnitude': ['real estate', 'life insurance', 'car', 'no known property'],
        'other_payment_plans': ['bank', 'stores', 'none'],
        'housing': ['rent', 'own', 'for free'],
        'job': ['unskilled resident', 'unskilled non-res', 'skilled employee', 'high qualif/self emp/mgmt'],
        'own_telephone': ['none', 'yes'],
        'foreign_worker': ['yes', 'no']
    }

    # Собираем категориальные данные
    cat_data = {}
    for feature, options in cat_features.items():
        selected_value = st.selectbox(feature.replace('_', ' ').title(), options)
        cat_data[feature] = selected_value

    # Числовые признаки
    num_features = {
        'duration': float(st.number_input('Duration (months)', min_value=0, max_value=100, value=12)),
        'credit_amount': float(st.number_input('Credit Amount', min_value=0, value=2000)),
        'installment_commitment': float(st.number_input('Installment Rate (%)', min_value=1, max_value=4, value=3)),
        'residence_since': float(st.number_input('Residence Since (years)', min_value=1, max_value=4, value=2)),
        'age': float(st.number_input('Age (years)', min_value=18, max_value=100, value=30)),
        'existing_credits': float(st.number_input('Existing Credits', min_value=0, max_value=10, value=1)),
        'num_dependents': float(st.number_input('Dependents', min_value=0, max_value=10, value=1))
    }

    # Объединяем все данные
    input_data = {**cat_data, **num_features}  # Сначала категориальные, потом числовые
    df = pd.DataFrame([input_data])
    
    # Убедимся, что порядок столбцов соответствует модели
    if feature_names:
        df = df[feature_names]
    
    # Определяем категориальные и числовые признаки
    numeric_features = ['duration', 'credit_amount', 'installment_commitment', 
                       'residence_since', 'age', 'existing_credits', 'num_dependents']
    
    # Преобразуем типы данных
    for col in df.columns:
        if col in numeric_features:
            df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype(str)
    
    return df, numeric_features

# Интерфейс приложения
st.write("<h1 style='text-align: center; color: #FF7F50;'> Credit_AI_Assistant</h1>", unsafe_allow_html=True)
st.image('data/main.png')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
col1, col2, col3, col4 = st.columns(4)


with col1:
   st.image("data/artificial intelligence.png", width=125)
   st.text_area('Text to analyze', '''This application makes predictions based on  Machine learning model build on curated dataset.  The  model was developed using open-source CatBoost.  CatBoost is an open-source software library developed by Yandex. It provides a gradient boosting framework which, among other features, attempts to solve for categorical features using a permutation-driven alternative to the classical algorithm''', height=350, label_visibility="hidden" )


with col2:
   st.image("data/checklist.png", width=125)
   st.text_area('Text to analyze', '''We follow the best practices for model development and validation.  The machine learning model for predicting credit risk using the German Credit dataset. The dataset consists of various attributes such as age, sex, job, credit history, and more to predict whether a person has a good or bad credit risk. The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann''', height=350, label_visibility="hidden" )


with col3:
   st.image("data/strong.png", width=125)
   st.text_area('Text to analyze', '''The  model showed high predictive power with vigorous validation metrics for test set, achieving AUC-ROC, F1-score, Precision, Recall  from 78 to 88%. The target variable is a binary classification where the output indicates whether a person has good or bad credit risk.''', height=350, label_visibility="hidden" )


with col4:
   st.image("data/puzzle-piece.png", width=125)
   st.text_area('**Bioavailability Radar**', '''A benefit of using gradient boosting is that after the boosted trees are constructed, it is relatively straightforward to retrieve importance scores for each attribute.''', height=350, label_visibility="hidden" )

st.write('Enter client details to predict credit risk:')

# Создаем DataFrame с вводом данных
input_df, numeric_features = create_input_df()

# Определяем индексы категориальных признаков
cat_features = [i for i, col in enumerate(input_df.columns) if col not in numeric_features]

# Кнопка для предсказания
if st.button('Predict Credit Risk'):
    try:
        # Создаем Pool с явным указанием категориальных признаков
        input_pool = Pool(input_df, cat_features=cat_features)
        
        # Прогноз
        prediction = model.predict(input_pool)
        probability = model.predict_proba(input_pool)[0][1]
        
        # Отображение результата
        result = 'Good' if prediction[0] == 1 else 'Bad'
        st.success(f'Prediction: **{result}** (Probability: {probability:.2%})')
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Debug information:")
        st.write("Input DataFrame dtypes:")
        st.write(input_df.dtypes)
        st.write("Input DataFrame values:")
        st.write(input_df)
        st.write("Numeric features:", numeric_features)
        st.write("Categorical features indices:", cat_features)
        st.write("Column order:", list(input_df.columns))

# Для отладки: показать введенные данные
st.write('Input Data Preview:')
st.write(input_df)