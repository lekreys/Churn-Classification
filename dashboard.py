import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report



st.set_page_config(page_title="Churn Prediction"  , page_icon="📈")


def load_data():
    return pd.read_csv("clean_df.csv")

def preprocess_data(data):
    pivot_areacode = data.pivot_table(index="area code", values=["total day minutes"], aggfunc=[np.mean, np.amax, np.median])

    pivot_mean = pivot_areacode['mean'].reset_index()
    pivot_median = pivot_areacode['median'].reset_index()
    pivot_max = pivot_areacode['amax'].reset_index()

    pivot_mean.columns = ["area code", 'mean_total day minutes']
    pivot_median.columns = ["area code", 'median_total day minutes']
    pivot_max.columns = ["area code", 'max_total day minutes']

    pivot_clean = pd.merge(pivot_mean, pivot_median, on='area code', how='left')
    pivot_clean = pd.merge(pivot_clean, pivot_max, on='area code', how='left')

    clean_data = pd.merge(data, pivot_clean, on='area code', how='left')
    return clean_data

def train_model_LGBM(data):

    X = data.drop(["churn", "area code"], axis="columns")
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

    over = SMOTE()
    X_train_over, y_train_over = over.fit_resample(X_train, y_train)

    model = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, max_depth=-1,
                           min_child_samples=10, n_estimators=500, num_leaves=40, objective='binary',
                           subsample=0.8)
    model.fit(X_train_over, y_train_over)
    return model , X_train_over , y_train_over , X_test , y_test

def train_model_XGB(data):

    X = data.drop(["churn", "area code"], axis="columns")
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

    over = SMOTE()
    X_train_over, y_train_over = over.fit_resample(X_train, y_train)

    model = XGBClassifier(colsample_bytree=0.7, gamma=0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.7)

    model.fit(X_train_over, y_train_over)
    return model , X_train_over , y_train_over , X_test , y_test

def train_model_Bagging(data):

    X = data.drop(["churn", "area code"], axis="columns")
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

    over = SMOTE()
    X_train_over, y_train_over = over.fit_resample(X_train, y_train)

    model = BaggingClassifier()

    model.fit(X_train_over, y_train_over)
    return model , X_train_over , y_train_over , X_test , y_test


def matrix (cm):

        cm_matrix = pd.DataFrame(index=['actual Positive', 'actual negative'], columns=['predict positive', 'predict negative'], data=cm)
        sns.set(rc={'figure.figsize':(5, 5)})

        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        st.pyplot()

        TP = cm[0,0]
        TN = cm[1,1]
        FN = cm[0,1]
        FP = cm[1,0]

        st.subheader(' Accuracy : ')
        classification_scoring = ((TP + TN) / (float(TP+TN+FN+FP)))
        if classification_scoring >= 0.8:  # Misalnya, jika akurasi >= 80%, tampilkan ikon hijau
            st.success(classification_scoring)
        elif classification_scoring >= 0.6:  # Jika akurasi >= 60%, tampilkan ikon kuning
            st.warning(classification_scoring)
        else:  # Jika akurasi < 60%, tampilkan ikon merah
            st.error(classification_scoring)


        col1, col2 = st.columns(2)
        with col1:

            st.subheader('F1 Score  : ')
            f1 = 2 * (((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))))

            if f1 >= 0.8:  # Misalnya, jika akurasi >= 80%, tampilkan ikon hijau
                st.success(f1)
            elif f1 >= 0.6:  # Jika akurasi >= 60%, tampilkan ikon kuning
                st.warning(f1)
            else:  # Jika akurasi < 60%, tampilkan ikon merah
                st.error(f1)


        with col2:
            
            st.subheader('Precision  : ')
            precision = TP / (TP + FP)
            if precision >= 0.8:  # Misalnya, jika akurasi >= 80%, tampilkan ikon hijau
                st.success(precision)
            elif precision >= 0.6:  # Jika akurasi >= 60%, tampilkan ikon kuning
                st.warning(precision)
            else:  # Jika akurasi < 60%, tampilkan ikon merah
                st.error(precision)

        col1, col2 = st.columns(2)
        with col1:

            st.subheader(' Recall : ')
            recall = TP / (float(TP + FN))
            if recall >= 0.8:  # Misalnya, jika akurasi >= 80%, tampilkan ikon hijau
                st.success(recall)
            elif recall >= 0.6:  # Jika akurasi >= 60%, tampilkan ikon kuning
                st.warning(recall)
            else:  # Jika akurasi < 60%, tampilkan ikon merah
                st.error(recall)



        with col2:
            
            st.subheader('specificity  : ')
            specificity= TN / (TN + FP)
            if specificity >= 0.8:  # Misalnya, jika akurasi >= 80%, tampilkan ikon hijau
                st.success(specificity)
            elif specificity >= 0.6:  # Jika akurasi >= 60%, tampilkan ikon kuning
                st.warning(specificity)
            else:  # Jika akurasi < 60%, tampilkan ikon merah
                st.error(specificity)






def visual (dataa) :

    st.subheader("choose yout plot :")
    pilihan_plot = st.selectbox("" , options = ["Distribution" , "Scatter Plot" , "Heatmap" , "Box Plot" , "Violin Plot", "Bar Plot" , "Count Plot" ])
    st.markdown("----")


    if pilihan_plot == "Distribution" : 
        st.header("Distribution : ")
        select_col = st.selectbox("select variable you want : " , options=dataa.columns)

        sns.set_style("darkgrid")
        sns.distplot(dataa[select_col] , bins=40 , color="green")
        plt.title(select_col)
        st.pyplot()

    elif pilihan_plot == "Scatter Plot" :

        st.header("Scatter Plot : ")

        col1,col2 = st.columns(2)
        with col1:
           select_col1 = st.selectbox("select X you want : " , options=dataa.columns)
        with col2 :

           select_col2 = st.selectbox("select Y you want : " , options=dataa.columns)

        sns.set_style("darkgrid")
        sns.scatterplot(data =dataa , x=select_col1 , y=select_col2  , color="green")
        plt.title(f"{select_col1} vs {select_col2}")
        st.pyplot()
    elif pilihan_plot == "Heatmap":
      
      st.header("Heatmap : ")


      plt.figure(figsize=(10,10))
      sns.heatmap(dataa.corr() , annot=True , cmap='YlGnBu' , fmt='.2f' , annot_kws={"size" : 7})
      plt.xticks(fontsize=8)  # Ukuran teks sumbu x disetel ke 8
      plt.yticks(fontsize=8)
      plt.xticks(rotation=90)
      plt.title('Correlation Map')
      st.pyplot()

    elif pilihan_plot == "Box Plot":

        st.header("Box Plot : ")

        select_col = st.selectbox("select variable you want : " , options=dataa.columns)

        sns.boxplot(data = dataa , y=select_col , palette="Set3", linewidth=2, width=0.6)
        st.pyplot()

    elif pilihan_plot == "Violin Plot" :

        st.header("Violin Plot : ")

        select_col = st.selectbox("select variable you want : " , options=dataa.columns)

        sns.violinplot(data = dataa , y=select_col , palette="Set3", linewidth=2, width=0.6)
        
        st.pyplot()
    
    elif pilihan_plot == "Count Plot" :

        st.header("Count Plot : ")

        select_col = st.selectbox("select variable you want : " , options=["area code"])

        sns.countplot(data = dataa , y=select_col , color="red")
        st.pyplot()

    elif pilihan_plot == "Bar Plot" : 

        st.header("Bar Plot :")

        col1 , col2= st.columns(2)


        with col1 : 
            select_col1 = st.selectbox("select x variable you want : " , options=["churn" , "area code"])
        with col2 :
            select_col2 = st.selectbox("select y variable you want : " , options=['international plan', 'voice mail plan', 'number vmail messages', 'total day minutes', 'total day calls', 'total day charge', 'total eve minutes', 'total eve calls', 'total eve charge', 'total night minutes', 'total night calls', 'total night charge', 'total intl minutes', 'total intl calls', 'total intl charge', 'customer service calls'])
        select_col3 = st.selectbox("Hue : " , options=["None","churn" , "area code"])


        
        if select_col3 == "None":
         sns.barplot(x=select_col1 , y=select_col2 , data=dataa ,  palette="husl")
         st.pyplot()
        elif select_col3 == "churn": 
         sns.barplot(x=select_col1 , y=select_col2 , hue=select_col3, data=dataa ,  palette="husl")
         st.pyplot()
        elif select_col3 == "area code": 
         sns.barplot(x=select_col1 , y=select_col2 , hue=select_col3, data=dataa ,  palette="husl")
         st.pyplot()





# Function to display page 1
def page1():
    st.title('Churn Prediction Modeling')
    st.write('Welcome to our Churn Prediction Modeling app!')

    st.subheader("Dataset Overview")
    all_data = pd.read_csv("clean_df.csv")
    st.dataframe(all_data)

    st.markdown("In this process, the best results were obtained with oversampling method (SMOTE()) and Advanced Category pre-processing (total day minutes (mean, max, median).")

    st.subheader("Clean And Final Data")
    final = preprocess_data(load_data())
    st.dataframe(final)

    st.subheader("Top 3 Models")

    models = ["1. LGBMClassifier", "2. XGBClassifier", "3. BaggingClassifier"]
    model_selection = st.selectbox("Models:", models, index=0)

    # Defining models
    def LGBMClassifier_model():

        LGBMC_model_final = train_model_LGBM(final)
        model = LGBMC_model_final[0]

        lgbm_train_score = model.score(LGBMC_model_final[1], LGBMC_model_final[2])
        lgbm_test_score = model.score(LGBMC_model_final[3], LGBMC_model_final[4])

        st.subheader(f'Training set score : {lgbm_train_score}')
        st.subheader(f'Testing set score : {lgbm_test_score}')

        y_pred_final = model.predict(LGBMC_model_final[3])

        cm = confusion_matrix(LGBMC_model_final[4], y_pred_final)

        matrix(cm)

        

    def XGBClassifier_model():

        X = final.drop("churn", axis="columns")
        y = final["churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

        over = SMOTE()
        X_train_over, y_train_over = over.fit_resample(X_train, y_train)

        XG_model_final = XGBClassifier(colsample_bytree=0.7, gamma=0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.7)
        XG_model_final.fit(X_train_over, y_train_over)

        xg_train_score = XG_model_final.score(X_train_over, y_train_over)
        xg_test_score = XG_model_final.score(X_test, y_test)

        st.subheader(f'Training set score : {xg_train_score}')
        st.subheader(f'Testing set score : {xg_test_score}')

        y_pred_final = XG_model_final.predict(X_test)

        cm = confusion_matrix(y_test, y_pred_final)
        matrix(cm)

    def BaggingClassifier_model():
        
        X = final.drop("churn", axis="columns")
        y = final["churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

        over = SMOTE()
        X_train_over, y_train_over = over.fit_resample(X_train, y_train)

        bc_model_final = BaggingClassifier()
        bc_model_final.fit(X_train_over, y_train_over)

        bc_train_score = bc_model_final.score(X_train_over, y_train_over)
        bc_test_score = bc_model_final.score(X_test, y_test)

        st.subheader(f'Training set score : {bc_train_score}')
        st.subheader(f'Testing set score : {bc_test_score}')
        y_pred_final = bc_model_final.predict(X_test)

        cm = confusion_matrix(y_test, y_pred_final)
        matrix(cm)


    if model_selection == "1. LGBMClassifier":
        LGBMClassifier_model()
    elif model_selection == "2. XGBClassifier":
        XGBClassifier_model()
    elif model_selection == "3. BaggingClassifier":
        BaggingClassifier_model()

# Function to display page 2
def page2():


    st.title("Dataset Visualisasi Churn")
    st.write("Decide whether this is a representation or an overview of the dataset to be used for modeling, or your own dataset. The choice is yours.")

    type_vil = st.selectbox(  "The choice is yours : " , options=["Modelling Datasets" , "Your Data"])
    st.markdown("---")

    if type_vil == "Modelling Datasets" :

     if st.button("Display Data"):
        st.write(load_data())
    
     dataa = load_data()
     visual(dataa)

    elif type_vil == "Your Data" :
     
     st.subheader("Your Data :")
     upload = st.file_uploader("")

     if upload is not None:
        
        dataa = pd.read_csv(upload)

        if st.button("Display Data"):
         st.write(dataa)
        visual(dataa)
     
    
    
     

    
     
    













        





# Function to display page 3
def page3():
    st.title('Predict your data Manually')
    st.write("So you can predict churn by entering values ​​one by one manually")

    st.subheader("Area code : ")
    area_input = st.selectbox("Area Code : " ,options= [415 , 408 , 510])

    st.subheader("Plan : ")

    col1 , col2 = st.columns(2)

    with col1 :
        input_international = st.number_input("international plan : " , min_value=0)
    with col2 : 
        input_voice = st.number_input("Voice main plan : " , min_value=0)

    input_vmail = st.number_input("Number Vmail messages : " , min_value=0)

    st.subheader("Day : ")

    col1 , col2 , col3 = st.columns(3)


    with col1 :
        input_day_minutes = st.number_input("Total Day Minutes : " , min_value=0)
    with col2 : 
        input_day_calls = st.number_input("Total day calls : " , min_value=0)
    with col3 : 
        input_day_charge = st.number_input("Total day Charge : " , min_value=0)


    st.subheader("Eve : ")

    col1 , col2 , col3 = st.columns(3)


    with col1 :
        input_eve_minutes = st.number_input("Total eve Minutes : " , min_value=0)
    with col2 : 
        input_eve_calls = st.number_input("Total eve calls : " , min_value=0)
    with col3 : 
        input_eve_charge = st.number_input("Total eve Charge : " , min_value=0)

    st.subheader(" Customer Service Calls : ")
    input_service = st.number_input("Customer Service Calls : " , min_value=0)


    st.subheader("Night : ")

    col1 , col2 , col3 = st.columns(3)


    with col1 :
       input_night_minutes = st.number_input("Total night Minutes : " , min_value=0)
    with col2 : 
        input_night_calls = st.number_input("Total night calls : " , min_value=0)
    with col3 : 
       input_night_charge = st.number_input("Total night Charge : " , min_value=0)

    st.subheader("Intl : ")

    col1 , col2 , col3 = st.columns(3)


    with col1 :
       input_intl_minutes = st.number_input("Total intl Minutes : " , min_value=0)
    with col2 : 
        input_intl_calls = st.number_input("Total intl calls : " , min_value=0)
    with col3 : 
       input_intl_charge = st.number_input("Total intl Charge : " , min_value=0)
    

    st.markdown("---")


    st.header("Select your models : ")
    modell = st.selectbox("Models : " , options=["LGBMClassifier" , "XGBoost" , "BaggingClassifier"])

    st.header(f"Predict ( {modell} ) : ")

    final = preprocess_data(load_data())

    if modell == "LGBMClassifier":
        LGBM = train_model_LGBM(final)
        model = LGBM[0]
    elif modell == "XGBoost":
        XGB = train_model_XGB(final)
        model = XGB[0]
    elif modell == "BaggingClassifier" : 
        BAG = train_model_Bagging(final)
        model = BAG[0]


    if area_input == 415:
        mean, median, max = 181.5926, 180.7, 350
    elif area_input == 408:
        mean, median, max = 177.1754, 176.35, 322.5
    else:
        mean, median, max = 178.7876, 179.45, 345.3
    variable = [[input_international , input_voice ,input_vmail ,input_day_minutes , input_day_calls , input_day_charge ,input_eve_minutes, input_eve_calls , input_eve_charge , input_night_minutes , input_night_calls , input_night_charge , input_intl_minutes , input_intl_calls , input_intl_charge , input_service , mean , median ,max]]
    

    col1 , col2 = st.columns(2)

    with col1 :
     
     st.subheader("Probability Not Churn : " , )
     pred_not_churn = model.predict_proba(variable)[:,0]
     st.write("{:.1f}%".format(round(pred_not_churn[0] * 100) , 2))
     st.write(pred_not_churn[0])
    
    with col2 : 
        st.subheader("Probability Churn : ")

        pred_churn = model.predict_proba(variable)[:,1]
        st.write("{:.1f}%".format(round(pred_churn[0] * 100) , 2))
        st.write(pred_churn[0])

    if pred_churn > pred_not_churn :
        st.error("Churn")
    else : 
        st.success("Not Churn")

    
    



def page4():
    st.title("Upload Your File To Predict ")
    st.write("You can predict your data here without manually entering one by one, you just need to input or upload the file you want to predict.")

    st.header("Example Data : ")
    st.write(load_data().drop("churn" , axis="columns") )

    # Membuat komponen file uploader
    uploaded_file = st.file_uploader("Choose a file")

    # Jika pengguna mengunggah file
    if uploaded_file is not None:
        st.header("Your Data : ")
        data = pd.read_csv(uploaded_file)
        st.write(data)
        st.header( "Result  : ")

        clean = preprocess_data(data)
        
        LGBM = train_model_LGBM(preprocess_data(load_data()))
        model = LGBM[0]

        x= clean.drop("area code" , axis="columns")

        pred = model.predict(x)
        prob_churn = model.predict_proba(x)[:,1]
        prob_not_churn = model.predict_proba(x)[:,0]

        clean["Churn"] = pred
        clean["prob_churn"] = prob_churn
        clean["prob_not_churn"] = prob_not_churn

        clean["result"] = np.where(clean["Churn"] == 0 , "Not Churn" , "Churn")

        st.write(clean)

        c = 0
        nc = 0
        for i , val in clean.iterrows():
            if val["result"] == "Not Churn" :
                nc = nc + 1
            else :
                c = c + 1

        col1 , col2 = st.columns(2)

        with col1:
            st.subheader("Not Churn")
            st.success(nc)
            st.success("{:.2f} %".format(nc / len(clean) * 100))
        with col2:
            st.subheader("Churn")
            st.error(c)
            st.error("{:.2f} %".format(c / len(clean) * 100))








    
    


        

        


       

        


    
    



    




    










st.sidebar.title("Navigation")


# Sidebar navigation
st.sidebar.markdown("Explore different sections:")

# Menambahkan pemisah untuk membuat tampilan lebih terorganisir
st.sidebar.markdown("---")

# Menambahkan pilihan navigasi
selection = st.sidebar.radio(
    "",
    ["Modeling Churn Dataset", "Dataset Visualisasi Churn", "Predict Your Data ", "Input Your Datasets To Predict"]
)

# Action based on selection
if selection == "Modeling Churn Dataset":
    page1()
elif selection == "Dataset Visualisasi Churn":
    page2()
elif selection == "Predict Your Data ":
    page3()
elif selection ==  "Input Your Datasets To Predict":
    page4()
