import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import neo4j
from neo4j import GraphDatabase
# from neo4j.exceptions import ServiceUnavailable

# from py2neo import Graph
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, roc_curve, roc_auc_score, mean_squared_error
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

load_dotenv()

URI = os.getenv('NEO4J_URI')
AUTH = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
QUERY = """
        MATCH (m:Movie) 
        WHERE none(k in ['imdbRating','budget','runtime','year','revenue','imdbVotes','released'] WHERE m[k] IS NULL)

        RETURN m.movieId as id,m.title as title,m.budget as budget,m.countries[0] as country,
        m.imdbId as imdbId,m.imdbRating as rating,m.imdbVotes as votes,
        m.languages[0] as language,m.plot as plot,m.poster as poster,m.released as released,m.revenue as revenue,
        m.runtime as runtime,m.tmdbId as tmdbId,
        m.url as url,m.year as year,[(m)-[:IN_GENRE]->(g) | g.name][0] as genre
        LIMIT $rows
        """

    
@st.cache_data(ttl=300, max_entries=100)
def read_data(query, rows=1):
    # with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    #     records, summary, keys = driver.execute_query(query, {"rows":rows})
    #     return pd.DataFrame(records, columns=keys)
    
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(query, {"rows": rows})
            keys = result.keys()
            records = [list(record.values()) for record in result]
            return pd.DataFrame(records, columns=keys)



def predict(input):
    st.sidebar.header("Prediction")
    data = input
    predict_column = st.sidebar.selectbox("Select value to predict", data.columns, index=data.columns.get_loc("budget"))

    # Preprocess the 'genre' column using one-hot encoding
    encoder = OneHotEncoder()
    encoder.fit(data[["genre"]])

    genres_encoded = encoder.transform(data[["genre"]]).toarray()

    # Create a DataFrame from the one-hot encoded genres and set column names
    genres_encoded_df = pd.DataFrame(genres_encoded, columns=encoder.get_feature_names_out(["genre"]))

    # Merge the one-hot encoded genres back into the original DataFrame
    # keep rating, genres_encoded, year
    data = input[["rating","year",predict_column]]
    data = data.join(genres_encoded_df) # .drop(columns=["genre","title","country","id"], axis=1)

    st.write(data.head())

    # Define the features (X) and target (y)
    X = data.drop(predict_column, axis=1)
    y = data[predict_column]

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model using the training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    st.text("Mean squared error: {}".format(mse))

    # Make sample predictions
    sample_data = [
        {"rating": 7.5, "year": 2000, "genre": "Action"},
        {"rating": 8.2, "year": 1995, "genre": "Drama"},
    ]
    sample_df = pd.DataFrame(sample_data)
    st.write(sample_df)

    st.header("2000 Action Movies")
    st.write(input[(input["year"] == 2000) & (input["genre"] == "Action")].head())
    st.header("1995 Dramas")
    st.write(input[(input["year"] == 1995) & (input["genre"] == "Drama")].head())

    # Preprocess the sample data (one-hot encoding for genre)
    sample_genres_encoded = encoder.transform(sample_df[["genre"]]).toarray()
    sample_genres_encoded_df = pd.DataFrame(sample_genres_encoded, columns=encoder.get_feature_names_out(["genre"]))
    sample_df = sample_df.join(sample_genres_encoded_df).drop("genre", axis=1)

    sample_predictions = model.predict(sample_df)
    st.write("Sample predictions: {}".format(sample_predictions))

    
def trace(data):
    st.sidebar.header("Visualisations")

    st.header("Importez un fichier EXCEL :")
    # data_file = "/Users/mh/Downloads/movieData/movies.csv"
    data_file = st.file_uploader("Fichier EXCEL", type=["xlsx"])

    if data_file is not None:
        data = pd.read_excel(data_file,sheet_name="Sheet1")
        # data = pd.DataFrame(records, columns=keys)
        st.write("Vue d’ensemble des données :")
        st.write(data)

        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choisissez un type de tracé :", plot_options)

        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Choisissez l'axe x :", data.columns)
            y_axis = st.sidebar.selectbox("Choisissez l'axe y :", data.columns)
            st.write("Bar plot :")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=100))
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Choisissez l'axe x :", data.columns)
            y_axis = st.sidebar.selectbox("Choisissez l'axe y :", data.columns)
            st.write("Nuage de points:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=100))
            st.pyplot(fig)

        elif selected_plot == "Histogram":

            column = st.sidebar.selectbox("Choisissez une variable :", data.columns)
            bins = st.sidebar.slider("Nombre de bins", 5, 100, 20)
            st.write("Histogramme :")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
             # Add annotations
            for rect in ax.patches:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f"{height:.0f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
            # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=100))
            st.pyplot(fig)


        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Choisissez une variable :", data.columns)
            st.write("Box plot :")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            st.pyplot(fig)

def var_ab_nan(data):

    st.sidebar.header("Visualisations")

    # st.header("Importez un fichier CSV")
    data_file = "C:/Users/gilbe/OneDrive/Bureau/chagpt-coding-main/chagpt-coding-main/streamlit-eda-ml/dfa.xlsx"
    # data_file = st.file_uploader("Fichier CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_excel(data_file,sheet_name="Sheet1")

        st.write("Vue d’ensemble des données :")
        st.write(data)

        plot_options = ["Données abérrantes", "Données manquantes"]
        selected_plot = st.sidebar.selectbox("Choisissez un type de données :", plot_options)

        if selected_plot == "Données abérrantes":

            # Sélectionner uniquement les variables quantitatives
            variables_quantitatives = data.select_dtypes(include=['int', 'float'])
            # Calculer le pourcentage de valeurs aberrantes par variable
            pourcentage_aberrant = variables_quantitatives.apply(lambda x: (x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) | (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))).mean() * 100
            
            df_pourcentage_aberrant = pd.DataFrame({'Variable': pourcentage_aberrant.index, 'Pourcentage (%)': pourcentage_aberrant.values})
        
            # Afficher le pourcentage de valeurs aberrantes par variable
            st.write("Pourcentage de valeurs aberrantes par variable en pourcentage (%) :")
            st.write(df_pourcentage_aberrant)

            # Tracer un histogramme avec seaborn
            st.set_option('deprecation.showPyplotGlobalUse', False) 
            ax = sns.barplot(x='Variable', y='Pourcentage (%)', data=df_pourcentage_aberrant)
            plt.xticks(rotation=90)
            plt.title("Pourcentage de valeurs aberrantes par variable")
            plt.xlabel("Variable")
            plt.ylabel("Pourcentage (%)")

            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=8)
            st.pyplot()

        elif selected_plot == "Données manquantes":

        # Calcul du nombre de données manquantes par attributs
            def get_nan_df(df):
            # '''
            #   Fonction permettant de calculer le nombre et le taux de données manquantes
            #   dans un dataframe

            #   INPUT
            #   -----
            #   df : dataframe

            #   OUTPUT
            #   -----
            #   dataframe avec la liste des attributs ('attribut'), le nombre de données manquantes ('nan_counts')
            #   et le taux de données manquantes pour l'attribut ('nan_rate)
            # '''
                col_names, nan_count, nan_rate = [], [], []
                for col in df.columns:
                    col_names.append(col)
                    count_nan = df[col].isna().sum()
                    nan_count.append(count_nan)
                    nan_rate.append(count_nan/df[col].shape[0])
            
                df_nan = pd.DataFrame(list(zip(col_names, nan_count, nan_rate)), columns=['variable', 'nan_counts', 'nan_rate (%)'])
                return df_nan.sort_values('nan_rate (%)',ascending=False).reset_index(drop=True)

            df_nan = get_nan_df(data)

            # Afficher le nombre de valeurs manquantes par variable
            st.write("Valeurs manquantes par variable :")
            st.write(df_nan.iloc[:,:-1])

            # Tracer un histogramme avec seaborn
            ax = sns.barplot(x='variable', y='nan_counts', data=df_nan)
            plt.xticks(rotation=90)
            # plt.yticks([int(x) for x in range(min(df_nan['nan_counts']), max(df_nan['nan_counts'])+1, 50)])
            plt.title("Valeurs manquantes par variable")
            plt.xlabel("Variable")
            plt.gca().set_ylim(bottom=0)

            for p in ax.patches:
                height = p.get_height()
                if height > 0 :
                    ax.annotate(f"{height():.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom',rotation=0, fontsize=8)
            st.pyplot()
        
            # Afficher le pourcentage de valeurs manquantes par variable
            st.write("Pourcentage de valeurs manquantes par variable :")
            st.write(df_nan.iloc[:,[0,2]])

            # Tracer un histogramme avec seaborn
            ax = sns.barplot(x='variable', y='nan_rate (%)', data=df_nan)
            plt.xticks(rotation=90)
            plt.title("Pourcentage de valeurs manquantes par variable")
            plt.xlabel("Variable")
            plt.ylabel("Pourcentage (%)")

            plt.ylim(0, 100)
            plt.yticks(np.arange(0, 101, 5))

            plt.axhline(y=50, color='red', linestyle='--')

            for p in ax.patches:
                height = p.get_height()
                if height > 0 :
                    ax.annotate(f"{height():.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom',rotation=0, fontsize=8)
            st.pyplot()

            # Suppression des attributs avec plus de 50% de données manquantes
            NAN_RATE_MAX = 0.5
            attr_many_nan = list(df_nan.loc[df_nan['nan_rate (%)'] > NAN_RATE_MAX, 'variable'])
            st.write('Les attributs avec plus de 50% de données manquantes sont : ',pd.DataFrame(attr_many_nan))
            st.write(attr_many_nan)

            data.drop(attr_many_nan, axis=1, inplace=True)
            # data.dtypes

            # Supprimer les accidents dont on a moins de 33% des informations
            ROW_NAN_RATE = 0.33
            percent_row = data.count(axis=1)/ len(data.columns)
            index_row = percent_row[percent_row < ROW_NAN_RATE].index
            # print(index_row)
            if len(index_row) > 0:
                for i in range(len(index_row)):
                    data.drop([index_row[i]], inplace=True)


            def fill_missing_values(df):
                '''
                Fonction permettant de remplacer les données manquantes dans un dataframe
                
                INPUT
                -----
                df : dataframe

                OUTPUT
                -----
                dataframe rempli avec la valeur médiane pour les attributs quantitatifs et la valeur la plus fréquence 
                pour les attributs catégoriques
                '''
                for col in df.columns:
                    if df[col].dtype == 'float64':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif df[col].dtype == 'object':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                return df


            def fill_missing_values_by_cat(df, output='gravite'):
                '''
                Fonction permettant de remplacer les données manquantes dans un dataframe par selon la catégorie de la variable de sortie

                INPUT
                -----
                df : dataframe
                output : colonne contenant les catégories

                OUTPUT
                -----
                dataframe rempli avec la valeur médiane pour les attributs quantitatifs et la valeur la plus fréquence 
                pour les attributs catégoriques selon la classe de l'observation
                '''
                df_category = df_data.groupby(output)    
                df_by_output = [df_category.get_group(x) for x in df_category.groups]
                L = []
                for df_group in df_by_output:
                    d = df_group.copy()
                    L.append(fill_missing_values(d))

                df = pd.concat(L, axis=0)
                return df

            data = fill_missing_values(data) # Finalement on utilisera pas le remplissage selon la classe car les résultats sont moins bons.
            df_nan = get_nan_df(data)
            st.write("Pourcentage de valeurs manquantes par variable (actualisé) :")
            st.write(df_nan)

    
def var_ab_nan2(data):

    data_file = "C:/Users/gilbe/OneDrive/Bureau/chagpt-coding-main/chagpt-coding-main/streamlit-eda-ml/dfa.xlsx"
    
    if data_file is not None:
        data = pd.read_excel(data_file,sheet_name="Sheet1")

        # Calcul du nombre de données manquantes par attributs
        def get_nan_df(df):
           
            col_names, nan_count, nan_rate = [], [], []
            for col in df.columns:
                col_names.append(col)
                count_nan = df[col].isna().sum()
                nan_count.append(count_nan)
                nan_rate.append(count_nan/df[col].shape[0])
        
            df_nan = pd.DataFrame(list(zip(col_names, nan_count, nan_rate)), columns=['variable', 'nan_counts', 'nan_rate (%)'])
            return df_nan.sort_values('nan_rate (%)',ascending=False).reset_index(drop=True)

        df_nan = get_nan_df(data)

        # Suppression des attributs avec plus de 50% de données manquantes
        NAN_RATE_MAX = 0.5
        attr_many_nan = list(df_nan.loc[df_nan['nan_rate (%)'] > NAN_RATE_MAX, 'variable'])
        
        data.drop(attr_many_nan, axis=1, inplace=True)
        # data.dtypes

        # Supprimer les accidents dont on a moins de 33% des informations
        ROW_NAN_RATE = 0.33
        percent_row = data.count(axis=1)/ len(data.columns)
        index_row = percent_row[percent_row < ROW_NAN_RATE].index
        # print(index_row)
        if len(index_row) > 0:
            for i in range(len(index_row)):
                data.drop([index_row[i]], inplace=True)


        def fill_missing_values(df):
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
            return df


        def fill_missing_values_by_cat(df, output='gravite'):
                df_category = df_data.groupby(output)    
                df_by_output = [df_category.get_group(x) for x in df_category.groups]
                L = []
                for df_group in df_by_output:
                    d = df_group.copy()
                    L.append(fill_missing_values(d))

                df = pd.concat(L, axis=0)
                return df

        data = fill_missing_values(data) # Finalement on utilisera pas le remplissage selon la classe car les résultats sont moins bons.
        df_nan = get_nan_df(data)
        
    return data


def var_interet(data):

    st.sidebar.header("Visualisations")

    # st.header("Importez un fichier CSV")
    data_file = "C:/Users/gilbe/OneDrive/Bureau/chagpt-coding-main/chagpt-coding-main/streamlit-eda-ml/dfa.xlsx"
    # data_file = st.file_uploader("Fichier CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_excel(data_file,sheet_name="Sheet1")
        # data = pd.DataFrame(records, columns=keys)
        st.write("Vue d’ensemble des données :")
        st.write(data)

        plot_options = ["Histogramme empilé","Histogramme groupé","Courbe sans marques","Courbe avec marques","Histogramme empilé 100%"]
        selected_plot = st.sidebar.selectbox("Choisissez un type de graphe :", plot_options)

        if selected_plot == "Histogramme empilé":

                selected_variable = st.sidebar.selectbox("Choisissez une variable :", data.columns)
                selected_elements = st.sidebar.multiselect("Selectionnez les modalités à afficher :", data['gravite'].unique())

                filtered_data = data[data['gravite'].isin(selected_elements)]
                
                bins = st.sidebar.slider("Number of bins", 5, 100, 20)
                st.write("Histogramme empilé :")
                fig, ax = plt.subplots()
                sns.histplot(filtered_data, x=selected_variable, hue='gravite', bins=bins, multiple='stack', ax=ax)
                for rect in ax.patches:
                        height = rect.get_height()
                        if height > 0:
                            ax.annotate(f"{height:.0f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom")
                            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
                st.pyplot(fig)


        elif selected_plot == "Histogramme groupé":

                selected_variable = st.sidebar.selectbox("Choisissez une variable :", data.columns)
                selected_elements = st.sidebar.multiselect("Selectionnez les modalités à afficher :", data['gravite'].unique())

                filtered_data = data[data['gravite'].isin(selected_elements)]

                st.write("Histogramme groupé :")
                fig, ax = plt.subplots()
                sns.countplot(x=selected_variable, hue='gravite', data=filtered_data, ax=ax)
                for rect in ax.patches:
                    height = rect.get_height()
                    if height > 0:
                        ax.annotate(f"{height:.0f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom")
                # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                plt.xticks(rotation=45)
                ax.set_xlabel(selected_variable)
                ax.set_ylabel("Count")
                ax.legend(title='gravite')
                st.pyplot(fig)

        elif selected_plot == "Courbe avec marques":

                selected_variable = st.sidebar.selectbox("Choisissez une variable :", data.columns)
                selected_elements = st.sidebar.multiselect("Selectionnez les modalités à afficher :", data['gravite'].unique())

                filtered_data = data[data['gravite'].isin(selected_elements)]

                st.write("Courbe avec marques :")
                fig, ax = plt.subplots()
                for element in selected_elements:
                    sns.kdeplot(filtered_data[filtered_data['gravite'] == element][selected_variable], ax=ax, label=element, marker='+')
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
                legend_labels = filtered_data['gravite'].unique()
                ax.legend(title='gravite', labels=legend_labels)
                st.pyplot(fig)

        elif selected_plot == "Courbe sans marques":

                selected_variable = st.sidebar.selectbox("Choisissez une variable :", data.columns)
                selected_elements = st.sidebar.multiselect("Selectionnez les modalités à afficher :", data['gravite'].unique())

                filtered_data = data[data['gravite'].isin(selected_elements)]

                st.write("Courbe sans marques :")
                fig, ax = plt.subplots()
                for element in selected_elements:
                    sns.kdeplot(filtered_data[filtered_data['gravite'] == element][selected_variable], ax=ax, label=element, marker='')
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
                legend_labels = filtered_data['gravite'].unique()
                ax.legend(title='gravite', labels=legend_labels)
                st.pyplot(fig)

        elif selected_plot == "Histogramme empilé 100%":

                selected_variable = st.sidebar.selectbox("Choisissez une variable :", data.columns)
                selected_elements = st.sidebar.multiselect("Selectionnez les modalités à afficher :", data['gravite'].unique())

                filtered_data = data[data['gravite'].isin(selected_elements)]

                bins = st.sidebar.slider("Number of bins", 5, 100, 20)
                st.write("Histogramme empilé 100% :")
                fig, ax = plt.subplots()
                sns.histplot(filtered_data, x=selected_variable, hue='gravite', multiple='fill', bins=bins, ax=ax)
                for rect in ax.patches:
                    height = rect.get_height()
                    if height > 0:
                        ax.annotate(f"{height*100:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom")
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

                ax.set_ylabel('Pourcentage (%)')
                vals = ax.get_yticks()
                ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

                legend_labels = filtered_data['gravite'].unique()
                ax.legend(title='gravite', labels=legend_labels)
                st.pyplot(fig)


def model(data):

    st.sidebar.header("Visualisations")

    data = var_ab_nan2(data)

    # st.header("Importez un fichier CSV")
    # data_file = "C:/Users/gilbe/OneDrive/Bureau/MDSMS 1/Regression logistique/Reg_Lo_Projet/Sujet 1/framingham.csv"
    # data_file = st.file_uploader("Fichier CSV", type=["csv"])

    # if data_file is not None:
        # data = pd.read_csv(data_file)
        # data = pd.DataFrame(records, columns=keys)
    st.write("Vue d’ensemble des données :")
    st.write(data)

    from sklearn import preprocessing
    def split_dataset(df, test_size, output_col='gravite', encoding='OneHot', set_seed=None):
        '''
        Fonction pour diviser le dataset en set d'entrainement et set de test.

        INPUTS
        -----
        df : un dataframe
        test_size : le % de taille du set de test
        output_col : la colonne correspondant à la variable de sortie
        encoding : 'None' pour pas d'encodage. 'Onehot' pour un encodage one-hot. 'LabelEncoder' pour un encodage numérique. 
        set_seed : Seed permettant la reproductibilité du résultat

        OUTPUTS
        -----
        X : dataframe contenant l'ensemble des prédicteurs
        Y : dataframe contenant uniquement la variable de sortie
        X_train : dataframe d'entrainement contenant les prédicteurs
        Y_train : dataframe d'entrainement contenant la variable de sortie
        X_test : dataframe de test contenant les prédicteurs
        Y_test : dataframe de test contenant la variable de sortie
        '''
        if encoding == 'LabelEncoder':
            cat_col = df.select_dtypes(include="object")
            Label_encoder = preprocessing.LabelEncoder()
            for col in cat_col:
                if col != output_col:
                    df[col] = Label_encoder.fit_transform(df[col])

        Y = df[output_col]
        X = df.loc[:, df.columns != output_col]

        if encoding == 'OneHot':
            X = pd.get_dummies(X,drop_first=True)

        # Division du dataset en set d'entrainement et de test
        if set_seed is not None:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=set_seed)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)
    
        #X_train, X_test = pd.get_dummies(X_train), pd.get_dummies(X_test)
        return X, Y, X_train, X_test, Y_train, Y_test

    X, Y, X_train, X_test, Y_train, Y_test = split_dataset(data, 0.30, 'gravite',encoding='OneHot', set_seed=204)


    plot_options = ["Decision Trees","Random Forests","Logistic Regression","Convolutional Neural Network (CNN)"]
    selected_plot = st.sidebar.selectbox("Choix du modèle :", plot_options)

    if selected_plot == "Decision Trees":
        st.sidebar.write()
        bins = st.sidebar.slider("profondeur maximale :", 2, 15, 2)
        
        # Réalisation d'une validation croisée afin d'obtenir la profondeur d'arbre optimale
        def cross_val() :
            kf = KFold(n_splits=5)

            depth = []
            #X_one_hot_data = pd.get_dummies(X,drop_first=True)
            for i in range(2,bins+1,1):
                st.write("... CV avec une profondeur max de =", i,"...")
                decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=i, min_samples_leaf=1, class_weight='balanced',random_state=2)
                # Perform 5-fold cross validation 
                scores = cross_val_score(decision_tree,X,Y, cv=kf, scoring='precision_micro',n_jobs=4)
                depth.append((i,scores.mean()))
            st.write("...Terminé...")

            depth_score = pd.DataFrame(depth, columns=['depth', 'precision (%)'])
            depth_score['precision (%)'] = depth_score['precision (%)'] * 100
            st.write(depth_score)

            # Affichage du résultat de la validation croisée
            st.write("Graphe de la Validation croisée en fonction de la profondeur de l'arbre :")
            fig, ax = plt.subplots()
            plt.plot(depth_score['depth'],depth_score['precision (%)'], '-o')
            ax.set_xlabel("Depth")
            ax.set_ylabel("Precision (%)")
            ax.set_title("Validation croisée : precision en fonction de la profondeur de l'arbre")
            st.pyplot(fig)

            depth_optim, depth_optim_precision = depth_score[depth_score['precision (%)'] == depth_score['precision (%)'].max()]['depth'].values[0], depth_score[depth_score['precision (%)'] == depth_score['precision (%)'].max()]['precision (%)'].values[0]

            st.write("La valeur optimale pour la profondeur est de", depth_optim,"avec une précision de :", round(depth_optim_precision, 2), "%")
            # return depth_optim, depth_optim_precision

    
            # def dt_model(depth_optim, depth_optim_precision) :
            # Ajustement d'un arbre de classification optimal et affichage des résulats de l'entrainement et de la validation
            best_model_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=depth_optim, min_samples_leaf=1, class_weight='balanced', random_state=200)
            clf = best_model_tree.fit(X_train, Y_train)
            class_levels = Y.unique()
            # class_levels[0], class_levels[1], class_levels[2], class_levels[3] = class_levels[1], class_levels[2], class_levels[0], class_levels[3]

            # Predictions
            Y_test_pred = clf.predict(X_test)
            # Calcul de la probabilité des classes prédites
            Y_test_pred_prob = clf.predict_proba(X_test)

            with st.expander("Matrice de confusion & Rapport de classification"):

                st.write("\n--------------- Validation des performances du modèle ---------------------\n")
                # Afficher les resultats
                st.write("Matrice de Confusion : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(Y_test, Y_test_pred), annot=True, fmt="g", yticklabels=class_levels, xticklabels=class_levels)
                ax.set_title("Matrix de  Confusion du modèle d'Arbre de décision")
                ax.set_ylabel('Valeurs réelles')
                ax.set_xlabel('Valeurs prédites')
                st.pyplot(fig)

                # Rapport de classification
                cr = classification_report(Y_test, Y_test_pred, output_dict=True)
                df1 = pd.DataFrame(cr).iloc[:-1,:]
                st.write("Rapport de classification : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                ax = sns.heatmap(df1, annot=True, cmap="Blues", cbar=True,fmt=".5f")
                ax.set_title("Rapport de classification du modèle d'Arbre de décision")
                ax.set_ylabel('Score')
                ax.set_xlabel('Classe/Métrique')
                ax.xaxis.set_ticks_position('top')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

                st.pyplot(fig)
                st.write(pd.DataFrame(cr))


            with st.expander("Area Under the Curve : ROC-AUC"):

                roc_auc = roc_auc_score(Y_test,Y_test_pred_prob,multi_class="ovo")
                st.write("L'aire sous la courbe  (ROC-AUC) est de : ", roc_auc)

                # label_encoder = LabelEncoder()

                # data2 = data.copy()

                # for column in data2.select_dtypes(include=["object"]):
                #     data2[column] = label_encoder.fit_transform(data2[column])
                
                # X_train2, X_test2, Y_train2, Y_test2 = train_test_split(data2.drop("gravite", axis=1), data2["gravite"], test_size=0.3)

                # best_model_tree.fit(X_train2, Y_train2)

                # Y_test_pred2 = best_model_tree.predict(X_test2)

                # st.write(Y_test2)
                # st.write(Y_test_pred2)

                # # Calcul de l'aire sous la courbe ROC
                # # roc_auc = roc_auc_score(Y_test2, Y_test_pred2, multi_class="ovr")

                # # Tracé de la courbe ROC pour le modèle de régression logistique
                # from sklearn.metrics import auc
                # fpr, tpr, thresholds = roc_curve(Y_test2, Y_test_pred2)
                # roc_auc = auc(fpr, tpr)
                # roc = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (ROC_AUC = {:0.15f})'.format(roc_auc))
                # roc_fill = go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 176, 246, 0.2)', name='AUC (Area Under Curve)')


                # layout = go.Layout(title="Courbe ROC (Receiver Operating Characteristic) du modèle d'arbres de décision",
                #                 xaxis=dict(title='Taux de Faux Positifs'),
                #                 yaxis=dict(title='Taux de Vrais Positifs')
                #                 )

                # fig = go.Figure(data=[roc,roc_fill], layout=layout)
                # st.plotly_chart(fig)

                # st.write("L'aire sous la courbe ROC (ROC-AUC) est de : ", roc_auc)

            with st.expander("Evaluation du modèle : Métriques"):

                st.write("Precision: ", round(precision_score(Y_test, Y_test_pred, average="macro")*100, 2), "%")
                st.write("Accuracy: ", round(accuracy_score(Y_test, Y_test_pred)*100, 2), '%\n')
                st.write("Recall : ", round(recall_score(Y_test, Y_test_pred, average="macro")*100, 2), '%\n')
                st.write("F1-Score: ", round(f1_score(Y_test, Y_test_pred, average="macro")*100, 2), '%\n')

                summary1 = {'model': ['DT'],
                        'F1-Score': [round(f1_score(Y_test, Y_test_pred, average="macro")*100, 2)],
                        'Accuracy' : [round(accuracy_score(Y_test, Y_test_pred)*100, 2)],
                        'precision' : [round(precision_score(Y_test, Y_test_pred, average="macro")*100, 2)],
                        'recall' : [round(recall_score(Y_test, Y_test_pred, average="macro")*100, 2)]
                        }

                dff1 = pd.DataFrame(summary1)
            
                # radar des metriques du modèle
                st.write("Radar des metriques : ")
                cell_lbl = str(dff1.iloc[:, 0].values[0])
                fig = go.Figure(data=go.Scatterpolar(
                r = dff1.iloc[0, 1:].to_numpy(),
                theta=dff1.iloc[:,1:].columns,
                name = cell_lbl,
                mode='lines+markers+text',
                text=dff1.iloc[0, 1:].to_numpy(),
                textfont=dict(color="black",size=14),
                textposition='top center',
                fill='toself'
                ))

                fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    color="black"
                    ),
                ),
                showlegend=True,
                legend=dict(
                title="Radar des métriques du modèle d'arbres de décision",
                x=0,  # Centrer le texte du titre horizontalement
                y=100  # Centrer le texte du titre verticalement
                ),
                width=800, 
                height=800,
                )

                st.plotly_chart(fig)
            # return dff1
        
        if st.sidebar.button("Effectuer une validation croisée"):
            st.write("Réalisation d'une validation croisée afin d'obtenir la profondeur d'arbre optimale")
            # depth_optim, depth_optim_precision = cross_val()
            cross_val()


    elif selected_plot == "Random Forests":
            
        # Création d'une grille de paramètres pour la validation croisée
        param_grid = {
            'max_depth': [],
            'n_estimators': [],
            # 'max_features': [int(np.sqrt(X.shape[1]))],
            'criterion' : [], 
        }

        # Utiliser Streamlit pour sélectionner les valeurs
        selected_max_depth = st.sidebar.multiselect('Sélectionnez la/les valeur(s) de la profondeur maximale :', range(5,16,5))
        selected_n_estimators = st.sidebar.multiselect("Sélectionnez la/les valeur(s) du nombre d'arbres :", range(100,1001,100))
        selected_criterion = st.sidebar.multiselect("Sélectionnez le/les critère(s) de selection :", ['gini','entropy'])


        # Mettre à jour les hyperparamètres avec les valeurs sélectionnées
        param_grid['max_depth'] = selected_max_depth
        param_grid['n_estimators'] = selected_n_estimators
        param_grid['criterion'] = selected_criterion

            # colonne_array = df['colonne1'].to_numpy()

        # st.write(pd.DataFrame(param_grid['max_depth']))

        def gridsearchcv_rf() :
            '''
            Réalisation d'une validation croisée afin d'obtenir la profondeur d'arbre optimale
            et le nombre d'arbres optimal pour Random Forest

            --> PEUT PRENDRE BEAUCOUP DE TEMPS
            Un long test a été fait avec 
            param_grid = {
                'max_depth': [i for i in range(5,20,5)],
                'n_estimators': [i for i in range(100,600,100)]
            }
            >>>> Résultat : 
            La valeur optimale pour la profondeur est de 15 avec 300 arbres et une précision de : 68.22 %

            Etant donné le temps de calcul qui peut être long, nous avons fait le choix
            de fixer le nombre max de prédicteurs à sqrt(nombre_total_de_predicteurs).
            '''

            # Création du modèle de Random Forest
            rf = RandomForestClassifier(bootstrap=True, min_samples_leaf=1, class_weight='balanced',random_state=204)
            # GridSearch avec une validation croisée à 5 folds
            grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = 4, verbose=3)    
            
            # Ajustement sur les données d'entrainement
            grid_search.fit(X_train, Y_train)  

            # Affichage des paramètres optimaux obtenus
            best_rf_depth, ntrees, best_criterion = grid_search.best_params_['max_depth'], grid_search.best_params_['n_estimators'], grid_search.best_params_['criterion']
            rf_cv_best_accuracy = grid_search.best_score_
            st.write("Random Forests : La valeur optimale pour la profondeur est de", best_rf_depth,"avec",ntrees,"arbres et pour critère de selection ", "''",grid_search.best_params_['criterion'],"''"," pour une exactitude de : ", round(rf_cv_best_accuracy*100, 2), "%")     

            # Ajustement d'un modèle Random Forest optimal
            best_rf = RandomForestClassifier(max_depth=best_rf_depth, n_estimators=ntrees, criterion=best_criterion, bootstrap=True, min_samples_leaf=1, class_weight='balanced', random_state=204)
            rf_clf = best_rf.fit(X_train, Y_train)

            class_levels = Y.unique()
            # class_levels[0], class_levels[1], class_levels[2], class_levels[3] = class_levels[1], class_levels[2], class_levels[0], class_levels[3]

            # Prédictions
            Y_test_pred = rf_clf.predict(X_test)
            # Calcul de la probabilité des classes prédites
            Y_test_pred_prob = rf_clf.predict_proba(X_test)

            with st.expander("Matrice de confusion & Rapport de classification"):

                st.write("\n--------------- Validation des performances du modèle ---------------------\n")
                # Afficher les resultats
                st.write("Matrice de Confusion : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(Y_test, Y_test_pred), annot=True, fmt="g", yticklabels=class_levels, xticklabels=class_levels)
                ax.set_title("Matrix de  Confusion du modèle de forets aléatoires")
                ax.set_ylabel('Valeurs réelles')
                ax.set_xlabel('Valeurs prédites')
                st.pyplot(fig)

                # Rapport de classification
                cr = classification_report(Y_test, Y_test_pred, output_dict=True)
                df1 = pd.DataFrame(cr).iloc[:-1,:]
                st.write("Rapport de classification : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                ax = sns.heatmap(df1, annot=True, cmap="Blues", cbar=True,fmt=".5f")
                ax.set_title("Rapport de classification du modèle de forets aléatoires")
                ax.set_ylabel('Score')
                ax.set_xlabel('Classe/Métrique')
                ax.xaxis.set_ticks_position('top')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

                st.pyplot(fig)
                st.write(pd.DataFrame(cr))


            with st.expander("Area Under the Curve : ROC-AUC"):

                roc_auc = roc_auc_score(Y_test,Y_test_pred_prob,multi_class="ovo")
                st.write("L'aire sous la courbe  (ROC-AUC) est de : ", roc_auc)

               
                # st.write("Aire sous la courbe ROC (ROC-AUC) :")
                # # Calcul de l'aire sous la courbe ROC
                # roc_auc = roc_auc_score(Y_test, Y_test_pred,multi_class="ovo")

                # # Tracé de la courbe ROC pour le modèle de régression logistique
                # fpr, tpr, thresholds = roc_curve(Y_test, Y_test_pred)
                # roc = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (ROC_AUC = {:0.15f})'.format(roc_auc))
                # roc_fill = go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 176, 246, 0.2)', name='AUC (Area Under Curve)')


                # layout = go.Layout(title="Courbe ROC (Receiver Operating Characteristic) du modèle de forets aléatoires",
                #                 xaxis=dict(title='Taux de Faux Positifs'),
                #                 yaxis=dict(title='Taux de Vrais Positifs')
                #                 )

                # fig = go.Figure(data=[roc,roc_fill], layout=layout)
                # st.plotly_chart(fig)

                # st.write("L'aire sous la courbe ROC (ROC-AUC) est de : ", roc_auc)

            with st.expander("Evaluation du modèle : Métriques"):

                st.write("Precision: ", round(precision_score(Y_test, Y_test_pred, average="macro")*100, 2), "%")
                st.write("Accuracy: ", round(accuracy_score(Y_test, Y_test_pred)*100, 2), '%\n')
                st.write("Recall : ", round(recall_score(Y_test, Y_test_pred, average="macro")*100, 2), '%\n')
                st.write("F1-Score: ", round(f1_score(Y_test, Y_test_pred, average="macro")*100, 2), '%\n')

                summary2 = {'model': ['RF'],
                        'F1-Score': [round(f1_score(Y_test, Y_test_pred, average="macro")*100, 2)],
                        'Accuracy' : [round(accuracy_score(Y_test, Y_test_pred)*100, 2)],
                        'precision' : [round(precision_score(Y_test, Y_test_pred, average="macro")*100, 2)],
                        'recall' : [round(recall_score(Y_test, Y_test_pred, average="macro")*100, 2)]
                        }

                dff2 = pd.DataFrame(summary2)
            
                # radar des metriques du modèle
                st.write("Radar des metriques : ")
                cell_lbl = str(dff2.iloc[:, 0].values[0])
                fig = go.Figure(data=go.Scatterpolar(
                r = dff2.iloc[0, 1:].to_numpy(),
                theta=dff2.iloc[:,1:].columns,
                name = cell_lbl,
                mode='lines+markers+text',
                text=dff2.iloc[0, 1:].to_numpy(),
                textfont=dict(color="black",size=14),
                textposition='top center',
                fill='toself'
                ))

                fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    color="black"
                    ),
                ),
                showlegend=True,
                legend=dict(
                title="Radar des métriques du modèle de forets aléatoires",
                x=0,  # Centrer le texte du titre horizontalement
                y=100  # Centrer le texte du titre verticalement
                ),
                width=800, 
                height=800,
                )

                st.plotly_chart(fig)


        if st.sidebar.button("Effectuer une validation croisée"):
            st.write("Réalisation d'une validation croisée afin d'obtenir le modèle de forets aléatoires optimal")
            gridsearchcv_rf()


    elif selected_plot == "Logistic Regression":
            
        # Création d'une grille de paramètres pour la validation croisée
        param_grid = {
            'penalty': [],
            'C': [],
            'solver': [],
            'max_iter' : [100, 300, 500, 1000,2000, 3000,5000]
        }

        # Utiliser Streamlit pour sélectionner les valeurs
        selected_penalty = st.sidebar.multiselect('Sélectionnez le/les type(s) de régularisation :', ['l1','l2','elasticnet'])
        selected_c = st.sidebar.multiselect("Sélectionnez la/les valeur(s) de la force de régularisation :", [0.01, 0.1, 1, 10])
        selected_solver = st.sidebar.multiselect("Sélectionnez le/les algorithme(s) d'optimisation :", ['saga','sag','lbfgs','newton-cg','liblinear'])
        selected_m_i = st.sidebar.multiselect("Sélectionnez la/les valeur(s) du nombre maximum d'itérations :", [100, 300, 500, 1000,2000, 3000,5000])
        

        # Mettre à jour les hyperparamètres avec les valeurs sélectionnées
        param_grid['penalty'] = selected_penalty
        param_grid['C'] = selected_c
        param_grid['solver'] = selected_solver
        param_grid['max_iter'] = selected_m_i


            # colonne_array = df['colonne1'].to_numpy()

        # st.write(pd.DataFrame(param_grid['max_depth']))

        def gridsearchcv_lr() :
     
            # Définition de notre Modèle de Regression Logistique, Entrainement et Prédiction sur les données d'entrainement et de test
            model = LogisticRegression(multi_class="multinomial")


            grid_search = GridSearchCV(model, param_grid = param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, Y_train)

            st.write("Meilleurs hyperparamètres : ", grid_search.best_params_)
            st.write("Meilleure performance :", grid_search.best_score_)

            class_levels = Y.unique()
            # class_levels[0], class_levels[1], class_levels[2], class_levels[3] = class_levels[1], class_levels[2], class_levels[0], class_levels[3]

            # Prédictions
            Y_test_pred = grid_search.predict(X_test)
            # Calcul de la probabilité des classes prédites
            Y_test_pred_prob = grid_search.predict_proba(X_test)

            with st.expander("Matrice de confusion & Rapport de classification"):

                st.write("\n--------------- Validation des performances du modèle ---------------------\n")
                # Afficher les resultats
                st.write("Matrice de Confusion : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(Y_test, Y_test_pred), annot=True, fmt="g", yticklabels=class_levels, xticklabels=class_levels)
                ax.set_title("Matrix de  Confusion du modèle de regression logistique")
                ax.set_ylabel('Valeurs réelles')
                ax.set_xlabel('Valeurs prédites')
                st.pyplot(fig)

                # Rapport de classification
                cr = classification_report(Y_test, Y_test_pred, output_dict=True)
                df1 = pd.DataFrame(cr).iloc[:-1,:]
                st.write("Rapport de classification : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                ax = sns.heatmap(df1, annot=True, cmap="Blues", cbar=True,fmt=".5f")
                ax.set_title("Rapport de classification du modèle de regression logistique")
                ax.set_ylabel('Score')
                ax.set_xlabel('Classe/Métrique')
                ax.xaxis.set_ticks_position('top')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

                st.pyplot(fig)
                st.write(pd.DataFrame(cr))


            with st.expander("Area Under the Curve : ROC-AUC"):

                # Calcul de l'aire sous la courbe ROC
                roc_auc = roc_auc_score(Y_test,Y_test_pred_prob,multi_class="ovo")
                st.write("L'aire sous la courbe  (ROC-AUC) est de : ", roc_auc)

                # Tracé de la courbe ROC pour le modèle de régression logistique
                # from sklearn.metrics import auc
                # fpr, tpr, thresholds = roc_curve(Y_test, Y_test_pred_prob)
                # roc_auc = auc(fpr, tpr)
                # roc = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (ROC_AUC = {:0.15f})'.format(roc_auc))
                # roc_fill = go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 176, 246, 0.2)', name='AUC (Area Under Curve)')


                # layout = go.Layout(title="Courbe ROC (Receiver Operating Characteristic) du modèle de regression logistique",
                #                 xaxis=dict(title='Taux de Faux Positifs'),
                #                 yaxis=dict(title='Taux de Vrais Positifs')
                #                 )

                # fig = go.Figure(data=[roc,roc_fill], layout=layout)
                # st.plotly_chart(fig)


            with st.expander("Evaluation du modèle : Métriques"):

                st.write("Precision: ", round(precision_score(Y_test, Y_test_pred, average="macro")*100, 2), "%")
                st.write("Accuracy: ", round(accuracy_score(Y_test, Y_test_pred)*100, 2), '%\n')
                st.write("Recall : ", round(recall_score(Y_test, Y_test_pred, average="macro")*100, 2), '%\n')
                st.write("F1-Score: ", round(f1_score(Y_test, Y_test_pred, average="macro")*100, 2), '%\n')

                summary3 = {'model': ['LR'],
                        'F1-Score': [round(f1_score(Y_test, Y_test_pred, average="macro")*100, 2)],
                        'Accuracy' : [round(accuracy_score(Y_test, Y_test_pred)*100, 2)],
                        'precision' : [round(precision_score(Y_test, Y_test_pred, average="macro")*100, 2)],
                        'recall' : [round(recall_score(Y_test, Y_test_pred, average="macro")*100, 2)]
                        }

                dff3 = pd.DataFrame(summary3)
            
                # radar des metriques du modèle
                st.write("Radar des metriques : ")
                cell_lbl = str(dff3.iloc[:, 0].values[0])
                fig = go.Figure(data=go.Scatterpolar(
                r = dff3.iloc[0, 1:].to_numpy(),
                theta=dff3.iloc[:,1:].columns,
                name = cell_lbl,
                mode='lines+markers+text',
                text=dff3.iloc[0, 1:].to_numpy(),
                textfont=dict(color="black",size=14),
                textposition='top center',
                fill='toself'
                ))

                fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    color="black"
                    ),
                ),
                showlegend=True,
                legend=dict(
                title="Radar des métriques du modèle de regression logistique",
                x=0,  # Centrer le texte du titre horizontalement
                y=100  # Centrer le texte du titre verticalement
                ),
                width=800, 
                height=800,
                )

                st.plotly_chart(fig)


        if st.sidebar.button("Effectuer une validation croisée"):
            st.write("Réalisation d'une validation croisée afin d'obtenir le modèle de regression logistique optimal")
            gridsearchcv_lr()


    elif selected_plot == "Convolutional Neural Network (CNN)" :
        st.sidebar.write("Choisissez le nombre d'epoques :")
        bins = st.sidebar.slider("nombre d'époques maximal :", 10, 50, 10)

        class_levels = Y.unique()
        class_levels[0], class_levels[1], class_levels[2], class_levels[3] = class_levels[1], class_levels[2], class_levels[0], class_levels[3]


        def cnn() :
            
            # Encodage de la variable cible
            label_encoder = LabelEncoder()
            Y_encoded = label_encoder.fit_transform(Y)
            Y_encoded_categorical = to_categorical(Y_encoded)

            # Diviser les données en ensembles d'entraînement et de test
            y_train_encoded, y_test_encoded = train_test_split( Y_encoded_categorical, test_size=0.3, random_state=42)

            # Standardiser les données
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_scaled = scaler.fit_transform(X)

            # Construire le modèle
            from keras import regularizers
            # from keras import KerasClassifier
            # from scikeras.wrappers import KerasClassifier
            # from keras.wrappers.scikit_learn import KerasClassifier

            # depth = []
            #X_one_hot_data = pd.get_dummies(X,drop_first=True)
            # for i in range(10,bins+1,1):
                # st.write("... CV avec une epoch max de =", i,"...")

            # def create_model() :
            model = Sequential()
            model.add(Dense(64, activation="relu", input_shape=(X_scaled.shape[1],), kernel_regularizer=regularizers.l2(0.01)))
            model.add(Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
            model.add(Dense(Y_encoded_categorical.shape[1], activation="softmax", kernel_regularizer=regularizers.l2(0.01)))  # Utilisation de softmax pour la classification multinomiale
            # Compiler le modèle
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                
            def keras_cv(model, X, y, epochs):
                kfold = KFold(n_splits=5, shuffle=True)
                losses = []
                accuracies = []
                for train, test in kfold.split(X, y):
                    model.fit(X[train], y[train], epochs=epochs, batch_size=32, verbose=0)
                    loss, acc = model.evaluate(X[test], y[test], verbose=0)
                    losses.append(loss)
                    accuracies.append(acc)

                return losses, accuracies
                
            
            results = []
            for epochs in range(1, bins+1):
                losses, accuracies = keras_cv(model, X_scaled, Y_encoded_categorical, epochs)
                mean_loss = np.mean(losses)
                mean_accuracy = np.mean(accuracies)
                results.append((epochs, mean_loss, mean_accuracy))

            # Afficher les résultats
            for epochs, mean_loss, mean_accuracy in results:
                st.write(f"epoch : {epochs},¨   | mean loss : {mean_loss},   | mean accuracy : {mean_accuracy}")

            # scores = cross_val_score(model,X,Y,epochs=i, batch_size=32, cv=kf, scoring='precision_micro',n_jobs=4)
            # depth.append((i,scores.mean()))
            st.write("...Terminé...")

            # depth_score = pd.DataFrame(depth, columns=['depth', 'precision (%)'])
            # depth_score['precision (%)'] = depth_score['precision (%)'] * 100
            # st.write(depth_score)

            # Affichage du résultat de la validation croisée
            epochs_values = range(1, bins+1)
            # Convertir results en un tableau NumPy
            results = np.array(results)
            st.write("Graphe de la Validation croisée en fonction de la profondeur de l'arbre :")
            fig, ax = plt.subplots()
            plt.plot(epochs_values, results[:, 2],'-o', label='Accuracy')
            plt.plot(epochs_values, results[:, 1],'-o', label='loss')


            for i, loss in enumerate(results[:, 1]):
                plt.text(epochs_values[i],loss, f"{loss:.4f}", ha='center', va='top', fontdict={'fontsize':7})

            for i, acc in enumerate(results[:, 2]):
                plt.text(epochs_values[i],acc, f"{acc:.4f}", ha='center', va='top', fontdict={'fontsize':7})

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

            xticks = list(epochs_values)
            plt.xticks(xticks, [str(x) if x in xticks else '' for x in range(min(xticks), max(xticks)+1)])

            # Récupérer uniquement les valeurs entières de l'axe des abscisses
            xticks = [int(x) for x in plt.xticks()[0]]

        
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss / Accuracy")
            ax.set_title("Validation croisée : (Loss / Accuracy) en fonction du nombre d'époques")
            ax.legend()
            st.pyplot(fig)

            # best_accuracy_index = np.argmax(results[:, 2])

            # optimal_epochs = epochs_values[best_accuracy_index]
            # best_loss = results[best_accuracy_index, 1]
            # best_accuracy = results[best_accuracy_index, 2]


            
            # depth_optim, depth_optim_precision = depth_score[depth_score['precision (%)'] == depth_score['precision (%)'].max()]['depth'].values[0], depth_score[depth_score['precision (%)'] == depth_score['precision (%)'].max()]['precision (%)'].values[0]

            # st.write("Le nombre optimal d'epoch est de", optimal_epochs ,"avec un loss de :", best_loss, "et une exactitude de :", best_accuracy)

            model.fit(X_train_scaled, y_train_encoded, epochs=bins, batch_size=32)
            
            
            # Prédictions sur l'ensemble de test
            y_test_pred = model.predict(X_test_scaled)
            predicted_classes = np.argmax(y_test_pred, axis=1)

            # class_levels = Y.unique()
            # class_levels[0], class_levels[1], class_levels[2], class_levels[3] = class_levels[1], class_levels[2], class_levels[0], class_levels[3]
            # class_levels[0], class_levels[1], class_levels[2], class_levels[3] = class_levels[1], class_levels[2], class_levels[0], class_levels[3]
            # Predictions
            # Y_test_pred = clf.predict(X_test)
            # Calcul de la probabilité des classes prédites
            # Y_test_pred_prob = clf.predict_proba(X_test)

            with st.expander("Matrice de confusion & Rapport de classification"):

                st.write("\n--------------- Validation des performances du modèle ---------------------\n")
                # Afficher les resultats
                st.write("Matrice de Confusion : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(np.argmax(y_test_encoded, axis=1), predicted_classes), annot=True, fmt="g", yticklabels=class_levels, xticklabels=class_levels)
                ax.set_title("Matrix de  Confusion du modèle de reseaux de neurones")
                ax.set_ylabel('Valeurs réelles')
                ax.set_xlabel('Valeurs prédites')
                st.pyplot(fig)

                # Rapport de classification
                cr = classification_report(np.argmax(y_test_encoded, axis=1), predicted_classes, output_dict=True)
                df1 = pd.DataFrame(cr).iloc[:-1,:]
                st.write("Rapport de classification : \n")
                # plt.figure(figsize=(5, 4))
                fig, ax = plt.subplots()
                ax = sns.heatmap(df1, annot=True, cmap="Blues", cbar=True,fmt=".5f")
                ax.set_title("Rapport de classification du modèle de reseaux de neurones")
                ax.set_ylabel('Score')
                ax.set_xlabel('Classe/Métrique')
                ax.xaxis.set_ticks_position('top')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

                st.pyplot(fig)
                st.write(pd.DataFrame(cr))


            with st.expander("Area Under the Curve : ROC-AUC"):

                roc_auc = roc_auc_score(np.argmax(y_test_encoded, axis=1), y_test_pred,multi_class="ovo")

                st.write("L'aire sous la courbe ROC (ROC-AUC) est de : ", roc_auc)
                # Calcul de l'aire sous la courbe ROC

                # # Tracé de la courbe ROC pour le modèle de régression logistique
                # fpr, tpr, thresholds = roc_curve(np.argmax(y_test_encoded, axis=1), predicted_classes)
                # roc = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (ROC_AUC = {:0.15f})'.format(roc_auc))
                # roc_fill = go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 176, 246, 0.2)', name='AUC (Area Under Curve)')


                # layout = go.Layout(title="Courbe ROC (Receiver Operating Characteristic) du modèle de reseaux de neurones",
                #                 xaxis=dict(title='Taux de Faux Positifs'),
                #                 yaxis=dict(title='Taux de Vrais Positifs')
                #                 )

                # fig = go.Figure(data=[roc,roc_fill], layout=layout)
                # st.plotly_chart(fig)

                # st.write("L'aire sous la courbe ROC (ROC-AUC) est de : ", roc_auc)

            with st.expander("Evaluation du modèle : Métriques"):

                st.write("Precision: ", round(precision_score(np.argmax(y_test_encoded, axis=1), predicted_classes, average="weighted")*100, 2), "%")
                st.write("Accuracy: ", round(accuracy_score(np.argmax(y_test_encoded, axis=1), predicted_classes)*100, 2), '%\n')
                st.write("Recall : ", round(recall_score(np.argmax(y_test_encoded, axis=1), predicted_classes, average="weighted")*100, 2), '%\n')
                st.write("F1-Score: ", round(f1_score(np.argmax(y_test_encoded, axis=1), predicted_classes, average="weighted")*100, 2), '%\n')

                summary4 = {'model': ['CNN'],
                        'F1-Score': [round(f1_score(np.argmax(y_test_encoded, axis=1), predicted_classes, average="weighted")*100, 2)],
                        'Accuracy' : [round(accuracy_score(np.argmax(y_test_encoded, axis=1), predicted_classes)*100, 2)],
                        'precision' : [round(precision_score(np.argmax(y_test_encoded, axis=1), predicted_classes, average="weighted")*100, 2)],
                        'recall' : [round(recall_score(np.argmax(y_test_encoded, axis=1), predicted_classes, average="weighted")*100, 2)]
                        }

                dff4 = pd.DataFrame(summary4)
            
                # radar des metriques du modèle
                st.write("Radar des metriques : ")
                cell_lbl = str(dff4.iloc[:, 0].values[0])
                fig = go.Figure(data=go.Scatterpolar(
                r = dff4.iloc[0, 1:].to_numpy(),
                theta=dff4.iloc[:,1:].columns,
                name = cell_lbl,
                mode='lines+markers+text',
                text=dff4.iloc[0, 1:].to_numpy(),
                textfont=dict(color="black",size=14),
                textposition='top center',
                fill='toself'
                ))

                fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    color="black"
                    ),
                ),
                showlegend=True,
                legend=dict(
                title="Radar des métriques du modèle  de reseaux de neurones",
                x=0,  # Centrer le texte du titre horizontalement
                y=100  # Centrer le texte du titre verticalement
                ),
                width=800, 
                height=800,
                )

                st.plotly_chart(fig)


        if st.sidebar.button("Evaluer le modèle"):
            st.write("Evaluation du modèle de reseau de neurones")
            cnn()

def corr_mtx(data) :
    # st.header("Importez un fichier EXCEL :")
    # data_file = "C:/Users/gilbe/OneDrive/Bureau/chagpt-coding-main/chagpt-coding-main/streamlit-eda-ml/dfa.xlsx"
    data_file = st.file_uploader("Fichier EXCEL", type=["xlsx"])

    if data_file is not None:
        data = pd.read_excel(data_file,sheet_name="Sheet1")
        # data = pd.DataFrame(records, columns=keys)
        st.write("Vue d’ensemble des données :")
        st.write(data)

        # Sélectionner les variables quantitatives
        qt_var = data.select_dtypes(include=[np.number])

        # Supprimer les colonnes du DataFrame ayant moins de 4 valeurs uniques
        # for col in qt_var.columns:
        #     if qt_var[col].nunique() <= 4:
        #         qt_var.drop(col, axis=1, inplace=True)

        # qt_var = data

        # st.write(qt_var)

        # Créer la matrice de corrélation pour les variables quantitatives
        corr_matrix = qt_var.corr()

        with st.expander("Matrice"):

            # Afficher la matrice de corrélation avec Seaborn
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', ax=ax)
            ax.set_title('Matrice de corrélation')
            st.pyplot(fig)

        with st.expander("Nuage de points"):
            sns.set(style='ticks')
            plot = sns.pairplot(qt_var)
            fig = plot.fig
            st.pyplot(fig)

def test_stat(data) :

    st.sidebar.header("Visualisations")

    data = var_ab_nan2(data)

    # st.header("Importez un fichier CSV")
    # data_file = "/Users/mh/Downloads/movieData/movies.csv"
    # data_file = st.file_uploader("Fichier CSV", type=["csv"])

    if data is not None:
        # data = pd.read_csv(data_file)

        st.write("Test d'indépendance sur les variables qualitatives et quantitatives :")

        st.write("Le test d'indépendance de la variable ''gravité'' sur les variables qualitatives et les variables quantitatives peut aider dans le projet de prédiction de la gravité de l'accident de plusieurs manières :",unsafe_allow_html=True)
        st.write("1. Variables qualitatives : En effectuant le test d'indépendance du chi carré sur les variables qualitatives par rapport à la variable ''gravité'', nous pouvons déterminer s'il existe une relation statistiquement significative entre ces variables. Si une telle relation est identifiée, cela indique que la variable qualitative peut être un bon prédicteur de la gravité de l'accident. Cela peut aider à identifier les caractéristiques spécifiques des accidents qui sont fortement associées à une gravité élevée, ce qui peut être exploité lors de la construction du modèle de prédiction.",unsafe_allow_html=True)
        st.write("2. Variables quantitatives : L'analyse de variance (ANOVA) est utilisée pour comparer la moyenne des variables quantitatives entre les différentes catégories de la variable ''gravité''. Si une différence significative est observée, cela suggère que les variables quantitatives peuvent également être des indicateurs importants de la gravité de l'accident. Cela signifie que des valeurs spécifiques de ces variables peuvent être associées à une gravité plus élevée, ce qui peut aider à améliorer les performances du modèle de prédiction.")
        st.write("En combinant les résultats du test d'indépendance du chi carré pour les variables qualitatives et l'analyse de variance (ANOVA) pour les variables quantitatives, nous obtenons des informations sur les variables qui sont potentiellement les plus pertinentes pour prédire la gravité de l'accident. Ces variables peuvent être utilisées comme caractéristiques ou prédicteurs lors de la construction du modèle de prédiction.")

        with st.expander("Test d'indépendance sur les variables qualitatives"):
            import scipy.stats as stats

            # Extraire les variables qualitatives et quantitatives
            variables_qualitatives = data.select_dtypes(include='object').columns
            # variables_qualitatives1 = data.select_dtypes(include='object').columns
            # variables_qualitatives2 = data.select_dtypes(include='number').columns[data.select_dtypes(include='number').nunique() <= 4]

            # variables_qualitatives = variables_qualitatives1.union(variables_qualitatives2)

            # colonne_a_retirer = 'gravite'

            # if colonne_a_retirer in variables_qualitatives:
            #     variables_qualitatives = variables_qualitatives.drop(colonne_a_retirer)

            variables_quantitatives = data.select_dtypes(include='number').columns

            # Test du chi carré pour les variables qualitatives
            chi2_results = {}
            for variable in variables_qualitatives:
                contingency_table = pd.crosstab(data[variable], data['gravite'])
                chi2, p_value, *_ = stats.chi2_contingency(contingency_table)
                chi2_results[variable] = {'Chi2': chi2, 'p-value': p_value}
            # Affichage des résultats
            st.write("Résultats du test du chi carré pour les variables qualitatives :")
            for variable, result in chi2_results.items():
                st.write(f"Variable : {variable}")
                st.write(f"Chi2 : {result['Chi2']}")
                st.write(f"p-value : {result['p-value']}")
                st.write()

        with st.expander("Test d'indépendance sur les variables quantitatives"):
            # Analyse de variance (ANOVA) pour les variables quantitatives
            anova_results = {}
            for variable in variables_quantitatives:
                groups = []
                for gravite in data['gravite'].unique():
                    groups.append(data[data['gravite'] == gravite][variable])
                anova, p_value = stats.f_oneway(*groups)
                anova_results[variable] = {'ANOVA': anova, 'p-value': p_value}
            st.write("Résultats de l'analyse de variance (ANOVA) pour les variables quantitatives :")
            for variable, result in anova_results.items():
                st.write(f"Variable : {variable}")
                st.write(f"ANOVA : {result['ANOVA']}")
                st.write(f"p-value : {result['p-value']}")
                st.write()

        with st.expander("Conclusion des tests"):
            st.title("Conclusion des tests :")
            st.write("Tous les tests **(test d'indépendance du chi carré pour les variables qualitatives et analyse de variance (ANOVA) pour les variables quantitatives)** indiquent que toutes les variables, que ce soit qualitatives ou quantitatives, sont significatives par rapport à la variable ''gravité'', cela suggère qu'elles sont toutes potentiellement importantes pour prédire la gravité des accidents. De plus, les différentes caractéristiques et mesures contenues dans nos variables peuvent fournir des informations utiles pour déterminer la gravité des accidents. En utilisant toutes ces variables ensemble, nous avons avez une gamme complète de facteurs qui peuvent influencer la prédiction de la gravité des accidents. Enfin, l' interprétation suggère que toutes les variables, qu'elles soient qualitatives ou quantitatives, fournissent des informations précieuses pour améliorer la prédiction de la gravité des accidents. En intégrant toutes ces variables dans le modèle de prédiction, nous avons une meilleure chance de capturer les relations complexes qui influencent la gravité des accidents.")


pages = {"Tracés et Graphes":trace,"Données abérrantes/manquantes":var_ab_nan,"Histogrammes sur la variable d'interet":var_interet,"Modèles":model,"Matrice de corrélation":corr_mtx,"Tests d'indépendance":test_stat}

def main():
    st.title("Prédiction du dégré de gravité d'un accident de la circulation : ''Streamlit Web-App''")
    st.sidebar.title("Menu")
    selected_page = st.sidebar.selectbox("Choisissez une page :", options=list(pages.keys()))

    rows = st.sidebar.number_input("Nombre de lignes :", min_value=1, max_value=5000, value=50, step=1)
    data = read_data(QUERY, rows)

    pages[selected_page](data)

if __name__ == "__main__":
    main()

