import pandas as pd
import pyarrow as pa
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import NearMiss as nm
from imblearn.over_sampling import RandomOverSampler as ros
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix


st.set_page_config(layout="wide")

st.sidebar.image("Streamlit.png")

add_selectbox = st.sidebar.selectbox(
    "Pages",
    ("Home", "EDA", "Modelling")
)



if add_selectbox == "Home":
    st.title("HOMEPAGE")
    image1=Image.open('image.jpg')
    st.image(image1, width=800)
    hp_select = st.selectbox("Select dataset from dropdown menu", 
                            ("Loan Prediction", "Water Portability"))
    st.markdown(""" ### -- {0} -- """.format(hp_select))
    if hp_select == "Loan Prediction":
        st.markdown(""" 
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
        Mauris cursus mattis molestie a. Et netus et malesuada fames ac turpis egestas integer. 
        Tortor dignissim convallis aenean et tortor at. Nullam non nisi est sit amet facilisis magna etiam. 
        Ipsum faucibus vitae aliquet nec ullamcorper sit amet risus. Purus ut faucibus pulvinar elementum integer enim. 
        Tellus id interdum velit laoreet id donec ultrices tincidunt. Tincidunt arcu non sodales neque. 
        Donec massa sapien faucibus et. Nunc consequat interdum varius sit amet.
        Arcu bibendum at varius vel pharetra vel. Egestas sed tempus urna et pharetra pharetra massa massa. 
        Volutpat commodo sed egestas egestas fringilla phasellus. Eget gravida cum sociis natoque penatibus et magnis. 
        Ultricies leo integer malesuada nunc vel. Dignissim sodales ut eu sem integer vitae justo eget magna. 
        Varius sit amet mattis vulputate enim nulla aliquet porttitor lacus. Phasellus egestas tellus rutrum tellus pellentesque eu. 
        Faucibus et molestie ac feugiat sed lectus vestibulum. Gravida arcu ac tortor dignissim convallis aenean. 
        Venenatis cras sed felis eget velit aliquet sagittis. In est ante in nibh mauris cursus mattis. 
        Pellentesque sit amet porttitor eget dolor morbi non arcu risus. 
        Adipiscing vitae proin sagittis nisl rhoncus mattis rhoncus urna neque. 
        Vulputate eu scelerisque felis imperdiet proin fermentum leo vel. 
        Blandit libero volutpat sed cras ornare arcu dui vivamus arcu. 
        Et netus et malesuada fames ac turpis.
        """)
        df=pd.read_csv('loan_prediction.csv')
        st.dataframe(df.head())
        st.write(df.describe())
    elif hp_select == "Water Portability":
        st.markdown(""" 
        Lectus urna duis convallis convallis tellus id interdum. 
        Nullam vehicula ipsum a arcu cursus vitae congue mauris. 
        Non odio euismod lacinia at quis risus sed vulputate odio. 
        Nisl suscipit adipiscing bibendum est ultricies. 
        Mauris nunc congue nisi vitae suscipit tellus. 
        Sed lectus vestibulum mattis ullamcorper velit sed ullamcorper morbi tincidunt. 
        Curabitur gravida arcu ac tortor dignissim. Integer feugiat scelerisque varius morbi enim. 
        Quam elementum pulvinar etiam non quam lacus suspendisse faucibus interdum. 
        Amet cursus sit amet dictum sit. Tellus mauris a diam maecenas sed enim. 
        Ac feugiat sed lectus vestibulum. Rhoncus mattis rhoncus urna neque viverra justo nec ultrices dui. 
        Tincidunt tortor aliquam nulla facilisi. Lacinia quis vel eros donec.
        Suspendisse faucibus interdum posuere lorem ipsum dolor. 
        Ultricies integer quis auctor elit. Mattis molestie a iaculis at erat pellentesque adipiscing commodo. 
        Vulputate sapien nec sagittis aliquam malesuada bibendum arcu. Viverra ipsum nunc aliquet bibendum. 
        Neque convallis a cras semper auctor. Amet commodo nulla facilisi nullam. Placerat in egestas erat imperdiet sed. 
        Lacus sed turpis tincidunt id aliquet risus feugiat in ante. Dolor sit amet consectetur adipiscing elit ut aliquam. 
        Gravida dictum fusce ut placerat orci nulla pellentesque. Nec tincidunt praesent semper feugiat nibh sed pulvinar. 
        Sodales ut etiam sit amet nisl purus in mollis nunc. At elementum eu facilisis sed odio morbi quis commodo. 
        Blandit cursus risus at ultrices mi tempus imperdiet nulla.        
        """)
        df = pd.read_csv('water_potability.csv')
        st.dataframe(df.head())
        st.write(df.describe())

        



elif add_selectbox == "EDA":
    st.title("Exploratory Data Analysis")
    # image2=Image.open('image.jpg')
    # st.image(image2, width=400)

    st.markdown("""## Select dataset from dropdown menu""")
    eda_select = st.selectbox("Select dataset:", 
                            ("Loan Prediction", "Water Portability"))
    st.markdown(""" ### -- {0} -- """.format(eda_select))

    if eda_select == "Loan Prediction":
        df=pd.read_csv('loan_prediction.csv')
        # df=df.dropna()
        st.dataframe(df)
    else :
        df=pd.read_csv('water_potability.csv')
        # df=df.dropna()
        st.dataframe(df)



    # drop_mult_select = st.multiselect("""Select columns to drop""", list(df.columns))
    # df=df.drop(drop_mult_select,axis=1)

    # if drop_mult_select:
    #      st.dataframe(df)

    numeric_columns=df.select_dtypes(include="number").columns

    for i in numeric_columns:
        if df[i].nunique() < 20:
            df[i] = df[i].astype('object')

    numeric_columns=df.select_dtypes(include="number").columns
    categorical_columns=df.select_dtypes(include="object").columns


    st.markdown("""## Statistical Values""")
    st.write(df.describe().T)

    # st.markdown("""## Correlation Heatmap""")
    # fig, ax=plt.subplots(1,1,figsize=(10,8))
    # sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    # st.pyplot(fig)

    st.markdown("""## Checking for Outliers""")

    out_check=st.radio("Select the method to check for outliers", ("Show All Numeric Variables", "Choose Variables"))

    fig, ax = plt.subplots()
    if out_check == "Show All Numeric Variables":
        ax.boxplot(df[numeric_columns], patch_artist=True, labels=numeric_columns, 
                    showfliers=False, showmeans=True, meanline=True, medianprops={'color':'red'})
        st.pyplot(fig)
    else:
        out_sel_col = st.selectbox("Select Column", numeric_columns)
        ax.boxplot(df[out_sel_col])
        st.pyplot(fig)


    st.markdown("""## Checking for Imbalance""")

    st.markdown("""#### Choosing a target variable:""")
    
    target_sel = df.iloc[:, -1]
    st.markdown("""##### Default Target variable is "{0}" """.format(target_sel.name))

    X = df.drop(target_sel.name, axis=1)

    

    change_var = st.checkbox("Change Target Variable")
    if change_var:
        target_sel = st.selectbox("Select Target Variable",df.columns)

    
    st.write(df.groupby(target_sel).size())

    if df.groupby(target_sel).size().min() /  df.groupby(target_sel).size().max() < 0.25:
        st.markdown("""##### The target variable is imbalanced""")
        st.markdown("""{0}""".format(df.groupby(target_sel).size().max() / df.groupby(target_sel).size().min()))
    else:
        st.markdown("""##### The target variable is balanced""")
        st.markdown("""Balance Result is: {0}%""".format(round(df.groupby(target_sel).size().min() /  df.groupby(target_sel).size().max() * 100), 2))


    st.bar_chart(df.groupby(target_sel).size())


    column1, column2, column3 = st.columns(3)


    with column1:
        st.markdown("""#### Checking Null Values""")
        st.write(df.isnull().sum())
    

    with column2:
        st.markdown("""#### Choosing Imputation Method""")
        drop_null = st.radio("Drop or Fill Null Values", ("Drop", "Fill"))
        if drop_null == "Drop":
            df=df.dropna()
        else:
            impute_method_cat = st.radio("Categorical", ('Backfill', 'Fill', "Mode"))
            impute_method_num = st.radio("Numerical", ('Median', "Mode",  'Mean'))


    with column3:
        st.markdown("""#### Feature Engineering""")

        clean_outliers = st.checkbox("Clean outliers")

        drop_unn_cols = st.checkbox("Drop Unnecessary Columns")
        if drop_unn_cols:
            drop_mult_select = st.multiselect("""Select columns to drop""", list(df.columns))
            df=df.drop(drop_mult_select,axis=1)

    numeric_columns=df.select_dtypes(include="number").columns
    categorical_columns=df.select_dtypes(include="object").columns

    submit_button = st.button("Submit Choices")

    
    if submit_button:
        if drop_null == "Fill":
            if impute_method_cat == "Backfill":
                df[categorical_columns] = df[categorical_columns].fillna(method='backfill')
            elif impute_method_cat == "Fill":
                df[categorical_columns] = df[categorical_columns].fillna(method='ffill')
            elif impute_method_cat == "Mode":
                if impute_method_cat == "Mode":
                    for col in categorical_columns:
                        df[col] = df[col].fillna(df[col].mode())
            elif impute_method_num == "Median":
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            elif impute_method_num == "Mode":
                for col in categorical_columns:
                    df[col] = df[col].fillna(df[col].mode())
            elif impute_method_num == "Mean":
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

            df.dropna(inplace=True)        

        df.reset_index(drop=True, inplace=True)

        if clean_outliers:
            def outlier_treatment(col):
                q1 = col.quantile(0.25)
                q3 = col.quantile(0.75)
                iqr = q3-q1
                lower_bound = q1 - 1.5*iqr
                upper_bound = q3 + 1.5*iqr
                return lower_bound, upper_bound

            for col in numeric_columns:
                lower, upper = outlier_treatment(df[col])
                df[col] = df[col].clip(lower, upper)


        
        st.markdown("""#### Successfully Submitted""")

        # st.bar_chart(df.groupby(df.iloc[:,-1]).size())

        document="cleaned_data"
        if document in os.listdir():
            os.remove(document)
            pickle.dump(df, open(document,'wb'))
        else:
            pickle.dump(df, open(document,'wb'))

            

        # st.dataframe(df)






elif add_selectbox == "Modelling":
    st.title("Modelling")
    # image3=Image.open('image.jpg')
    # st.image(image3, width=400)

    document="cleaned_data"
    if document not in os.listdir():
        st.markdown("""#### Please run the EDA step first""")
        
    else:
        df = pickle.load(open(document,'rb'))

        st.dataframe(df)

        X = df.drop(df.columns[-1], axis=1)
        y = df.iloc[:, -1]

        mod_col_1, mod_col_2 = st.columns(2)
        with mod_col_1:
            st.title("Scaling/Encoding")

            choose_method = st.radio("Choosing Columns for Scaling/Encoding methods by default", ("Yes", "No"))
        
        with mod_col_2:
            st.title("Sampling Methods")
            under_samp = st.checkbox("Under Sampling")
            if under_samp:
                over_samp = st.checkbox("Over Sampling", disabled=True)
            else:
                over_samp = st.checkbox("Over Sampling")

        if choose_method == "Yes":
            scl_enc_1, scl_enc_2 = st.columns(2)

            with scl_enc_1:
                st.markdown("""#### Select Scaling Method""")
                scl_select = st.radio("Scaling", ("Standard", "Robust", "MinMax"))

            with scl_enc_2:
                st.markdown("""#### Select Encoding Method""")
                enc_select = st.radio("Encoders", ("LabelEncoder", "OneHotEncoder"))

        else:
            column_enc_1, column_enc_2, column_enc_3 = st.columns(3)

            with column_enc_1:
                st.markdown("""#### Choosing Columns for OneHotEncoder:""")
                ohe_col = st.multiselect("Select Columns for OHE", X.select_dtypes(include="object").columns)
                X.drop(ohe_col, axis=1, inplace=True)

            with column_enc_2:
                st.markdown("""#### Choosing Columns for LabelEncoder:""")
                lab_col = st.multiselect("Select Columns for LE", X.select_dtypes(include="object").columns)
                X.drop(lab_col, axis=1, inplace=True)
            
            with column_enc_3:
                st.markdown("""#### Choosing Scaling Method for Numeric Columns:""")
                scl_col = st.selectbox("Select Method", ("Standard", "Robust", "MinMax"))


        process_button = st.button("Data Processing")

        
        numeric_columns = X.select_dtypes(include="number").columns
        categorical_columns = X.select_dtypes(include="object").columns


        if process_button:
            y = LabelEncoder().fit_transform(y)

            if choose_method == "Yes":
                if scl_select == "Standard":
                    scaler = StandardScaler()
                    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
                elif scl_select == "Robust":
                    scaler = RobustScaler()
                    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
                elif scl_select == "MinMax":
                    scaler = MinMaxScaler()
                    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])


                if enc_select == "OneHotEncoder":
                    for i in categorical_columns:
                        ohe = OneHotEncoder(categories='auto')
                        feature_arr = ohe.fit_transform(np.array(X[i]).reshape(-1, 1)).toarray()
                        feature_labels = ohe.categories_
                        # st.dataframe(pd.DataFrame(feature_arr, columns=feature_labels[0]))
                        # st.markdown("""feature_labels: {}""".format(feature_labels[0]))
                        feature_labels = np.array(feature_labels).ravel()
                    
                        features = pd.DataFrame(feature_arr, columns= [i + '_' + str(fl) for fl in feature_labels])
                        # st.dataframe(features)
                        
                        X.drop(i, axis=1, inplace=True)
                        X = pd.concat([X, features], axis=1)

                        # ohe_data = OneHotEncoder().fit_transform(np.array(X[i]).reshape(-1, 1)).toarray()
                        # ohe_transform = pd.DataFrame(ohe_data, columns = [i + '_' + str(c) for c in X[i].unique()])
                        # X.drop(i, axis=1, inplace=True)
                        # X = pd.concat([X, ohe_transform], axis=1)
                    # st.dataframe(X)
                        
                elif enc_select == "LabelEncoder":
                    for col in categorical_columns:
                        X[col] = LabelEncoder().fit_transform(X[col])

            elif choose_method == "No":
                X = df.drop(df.columns[-1], axis=1)
                if ohe_col:
                    for i in ohe_col:
                        ohe = OneHotEncoder(categories='auto')
                        feature_arr = ohe.fit_transform(np.array(X[i]).reshape(-1, 1)).toarray()
                        feature_labels = ohe.categories_
                        # st.dataframe(pd.DataFrame(feature_arr, columns=feature_labels[0]))
                        # st.markdown("""feature_labels: {}""".format(feature_labels[0]))
                        feature_labels = np.array(feature_labels).ravel()
                    
                        features = pd.DataFrame(feature_arr, columns= [i + '_' + str(fl) for fl in feature_labels])
                        # st.dataframe(features)
                        
                        X.drop(i, axis=1, inplace=True)
                        X = pd.concat([X, features], axis=1)


                        # ohe_data = OneHotEncoder().fit_transform(np.array(X[i]).reshape(-1, 1)).toarray()
                        # ohe_transform = pd.DataFrame(ohe_data, columns = [i + '_' + str(c) for c in X[i].unique()])
                        # X.drop(i, axis=1, inplace=True)
                        # X = pd.concat([X, ohe_transform], axis=1)

                if lab_col:
                    for col in X[lab_col].columns:
                        X[col] = LabelEncoder().fit_transform(X[col])
                
                if scl_col:
                    if scl_col == "Standard":
                        scaler = StandardScaler()
                        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
                    elif scl_col == "Robust":
                        scaler = RobustScaler()
                        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
                    elif scl_col == "MinMax":
                        scaler = MinMaxScaler()
                        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

            X = pd.DataFrame(X)
            # st.dataframe(new_X)
            y = pd.DataFrame(y)
            # st.dataframe(new_y)

            if under_samp:
                undersample = nm(version=1, n_neighbors=3)
                X, y = undersample.fit_resample(X, y)
                # df = pd.concat([X, y], axis=1)
            
            if over_samp:

                oversample = ros(random_state=42)
                X, y = oversample.fit_resample(X, y)
                # df = pd.concat([X, y], axis=1)


            new_data = pd.concat([X, y], axis=1)
            # st.dataframe(new_data)

            st.markdown("""#### Successfully Processed Data""")
            # st.dataframe(pd.DataFrame(X))

            document="prepared_data"
            if document in os.listdir():
                os.remove(document)
                pickle.dump(new_data, open(document,'wb'))
            else:
                pickle.dump(new_data, open(document,'wb'))

        
        document="prepared_data"
        if document in os.listdir():
            df = pickle.load(open(document,'rb'))
            st.dataframe(df)

            X = df.drop(df.columns[-1], axis=1)
            y = df.iloc[:, -1]

            st.title("Modeling")
            model_col_1, model_col_2, model_col_3 = st.columns(3)

            with model_col_1:
                random_state = st.number_input("Random State", value=42)
            with model_col_2:
                percentage = st.number_input("Percentage of Data", value=80)
            with model_col_3:
                model_select = st.selectbox("Select Models", ("XGBoost", "Logistic Regression", "Random Forest", "Decision Tree", "Naive Bayes", "SVM"))

            result = st.markdown("#### Random State: {0} \n #### Percentage: {1}% \n #### Model: {2}".format(random_state, percentage, model_select))

            run_model_button = st.button("Run Model")

            if run_model_button:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage/100, random_state=random_state)

                if model_select == "XGBoost":
                    model = XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_select == "Logistic Regression":
                    model = LogisticRegression(random_state=random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_select == "Random Forest":
                    model = RandomForestClassifier(random_state=random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_select == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_select == "Naive Bayes":
                    model = GaussianNB()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_select == "SVM":
                    model = SVC(random_state=random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                st.markdown("#### Accuracy: {0}".format(accuracy))


                # classification_report = classification_report(y_test, y_pred)
                # st.markdown("#### Classification Report: \n {0}".format(classification_report))

                # confusion_matrix = confusion_matrix(y_test, y_pred)
                # st.markdown("#### Confusion Matrix: \n {0}".format(confusion_matrix))
