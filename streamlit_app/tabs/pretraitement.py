import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from PIL import Image
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

title = "Affichage 2D & Tests statistiques"
sidebar_name = title


def run():

    df = pd.read_csv("assets/bank.csv", sep=",", header=0)
    df_1 = df.copy()

    # Suppresion de 'duration' & 'campaign'
    df_1 = df_1.drop('duration', axis = 1)
    df_1 = df_1.drop('campaign', axis = 1)

# Suppresion des 2 lignes incohérentes : 
    df_1 = df_1[(df_1['poutcome'] != 'unknown') | (df_1['previous'] == 0)]

    st.title(title)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Affichage 2D", "Tests Statistiques"])
    
    with tab1:

        st.header('ACP : Analyse des composantes principales')
        st.write(''' Afin d'avoir une meilleure représentation du jeu de données, nous allons opter pour une analyse en composante principale, l'ACP permet de transformer des variables qui sont corrélées en variables dé-corrélées. Cette analyse permet de réduire le nombre de variables du jeu de données pour simplifier les observations tout en conservant un maximum d'informations. 
    Pour cela, il est nécessaire de n’avoir que des variables numériques. Nous transformons donc les variables catégorielles en variables indicatrices. Nos données ne sont pas sur la même base, on met tous les champs à la même échelle en centrant et réduisant les données (MinMaxScaler)
    Cette analyse permet de réduire le nombre de variables du jeu de données pour simplifier les observations tout en conservant un maximum d'informations.''' )
        
        st.image(Image.open("assets/PCA_1.png"), use_column_width=False, caption='ACP',  width=400)
    
        st.write(''' En projetant les 2 premières composantes, nous atteignons 24% d'inertie expliquée. Les données sont très mélangées et il est vraiment difficile de distinguer des zones départageant le deposit.
    On peut tout de même voir que les points rouges sont majoritaires en bas à gauche et les points bleus le sont en haut à droite du graphique. ''')

        st.header('Autre méthode de réduction de dimension')
        st.write('ISOMAP & TNSE : ')
        st.image(Image.open("assets/PCA_2.png"), use_column_width=False, caption='ISOMAP & TNSP',   width=750)


        st.write(''' On a testé deux autres méthodes : Isomap et TNSE . Nous parvenons au même constat que pour le PCA. Il y a bien certaines zones regroupant uniquement ou majoritairement du rouge. 
    Mais on a également énormément de zones où les points rouges et bleus sont mélangés. 
    Ce que montre ces 3 graphiques, c’est qu’il sera très difficile de départager l’ensemble des données. Nous ne sommes pas sur un sujet simple et nous ne pourrons donc pas espérer de résultat avoisinant les 100%. ''')

    with tab2:

        st.header("Analyse des variables catégorielles à l'aide du test du 𝜒2")
        st.write('Le tableau suivant nous donne les informations suivantes : statistique du test, p-value, \
    degré de liberté, et V de Cramer (coefficient de corrélation du 𝜒2)')

        from scipy.stats import chi2_contingency
        df_1_cat = df_1.select_dtypes('object')
        def V_Cramer(table, N):
            stat_chi2 = chi2_contingency(table)[0]
            k = table.shape[0]
            r = table.shape[1]
            phi_2 = max(0,(stat_chi2)/N - ((k - 1)*(r - 1)/(N - 1)))
            k_b = k - (np.square(k - 1) / (N - 1))
            r_b = r - (np.square(r - 1) / (N - 1))   
            return np.sqrt(phi_2 / min(k_b - 1, r_b - 1))

        dico = {}
        for col in df_1_cat.columns:
            table = pd.crosstab(df_1_cat[col], df_1['deposit'])
            res = chi2_contingency(table)
            dico[col] = [res[0], res[1], res[2], V_Cramer(table, df_1.shape[0])]
        
        
        stats = pd.DataFrame.from_dict(dico).transpose()
        stats = stats.rename(columns={0:'chi 2', 1:'p-value', 2:'DoF', 3:'V de Cramer'})
        st.dataframe(stats)
        st.write(''' les p-values sont toutes sous la barre des 5%, on rejette donc l'hypothèse nulle H0 'les 2 variables testées sont indépendantes'.\n\
    De même, le V de Cramer présente des valeurs insuffisantes notamment pour les variables 'marital', 'default' ''')
        
        st.header("Analyse des variables numériques à l'aide du test ANOVA")

        st.write(''' Une ANOVA ('analyse de variance') est utilisée pour déterminer si oui ou non les moyennes de trois groupes indépendants ou plus sont égales. \n\
    Une ANOVA utilise les hypothèses nulles et alternatives suivantes :  \n\
    - H0 : Toutes les moyennes de groupe sont égales.  \n\
    - HA : Au moins une moyenne de groupe est différente des autres. ''')

        st.write(''' Comprendre la valeur P dans ANOVA :  \n Si cette valeur de p est inférieure à α = 0,05, \n\
    nous rejetons l'hypothèse nulle de ANOVA et \n concluons qu'il existe une différence statistiquement \n\
    significative entre les moyennes des trois groupes. \n\
    Sinon, si la valeur de p n'est pas inférieure à α = 0,05, nous ne rejetons pas l'hypothèse nulle \n\
    et concluons que nous n'avons pas suffisamment de preuves pour dire qu'il existe une différence statistiquement significative entre les moyennes des trois groupes. ''')
        
        data_anova = {'F':[13.587852,73.92161, 35.584154, 261.977062, 222.509262],
            'PR(>F)':[0.000229,9.212031e-18,2.516763e-09,2.925198e-58,7.740962e-50]}
        df_anova = pd.DataFrame(data_anova, index=['age','balance','days','pdays','previous'])
        st.dataframe(df_anova) 
        st.write(''' Les p-values sont toutes sous la barre des 5%, on rejette donc l'hypothèse nulle H0, de même, le V de Cramer présente des valeurs insuffisantes la plupart des variables, on peut en conclure qu’il est difficile de mettre en évidence des features indispensables.''' )

        st.header(''' Correlation entre variables catégoriques : ''')
        # Création d'un dataframe avec les variables catégoriques :
        df_1_cat = df_1.select_dtypes('object')
        df_1_cat.columns
        from itertools import product

        # Création d'un dataframe avec les variables catégoriques :
        df_1_cat = df_1.select_dtypes('object')
        df_1_cat.columns
        # Split de df_1_cat en 2 parties : 
        df_1_cat_var1 = ('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit')

        #  Chi-2 Test :
        ## Création de toutes les combinaison Creating all possible combinations between the above two variables list
        cat_var_prod = list(product(df_1_cat_var1,df_1_cat_var1, repeat = 1))

        ## Création d'une liste result et la remplir avec les p values issues du chi2 test : 
        import scipy.stats as ss

        result = []
        for i in cat_var_prod:
            if i[0] != i[1]:
                result.append((i[0],i[1],list(ss.chi2_contingency(pd.crosstab(df_1_cat[i[0]], df_1_cat[i[1]])))[1]))

        chi_test_output = pd.DataFrame(result, columns = ['var1', 'var2', 'coeff'])
        ## Using pivot function to convert the above DataFrame into a crosstab
        chi_test_output.pivot(index='var1', columns='var2', values='coeff')
        # Heatmap : 
        st.image(Image.open("assets/corr_cat.png"), use_column_width=False, caption='Heatmap Chi-2',  width=800)
        #fig = plt.figure(figsize=(15,8))
        #sns.heatmap(chi_test_output.pivot(index='var1', columns='var2', values='coeff'), annot=True, cmap='Blues')        
        #st.write(fig)
        st.write(''' Conclusions :\n Pour rappel : Chaque test statistique dispose de ce qu’on appelle la p-value.\n\ 
        On peut la voir comme une valeur référence pour décider du rejet ou non de l’hypothèse nulle.\n\ 
        Si cette dernière est en-dessous de 5% alors on rejette l’hypothèse nulle (« les deux variables testées sont indépendantes ») \n\ ''')
        st.write(''' Ici p-value > 0,05 pour les variables suivantes: ''')
        st.write(''' - Education / default''')
        st.write(''' - Education / housing ''') 
        st.write(''' - Loan / contact''') 
        st.write(''' - Marital / default''')
        st.write('''Ces variables sont indépendantes ''')
        
        st.header(''' Correlation entre variables numériques : ''')
        st.write(''' Le coefficient de Pearson est un indice reflétant une relation linéaire entre deux variables continues. Le coefficient de corrélation varie entre -1 et +1, 0 reflétant une relation nulle entre les deux variables, une valeur négative (corrélation négative) signifiant que lorsqu'une des variable augmente, l'autre diminue ; tandis qu'une valeur positive (corrélation positive) indique que les deux variables varient ensemble dans le même sens.
    La valeur de r obtenue est une estimation de la corrélation entre deux variables continues dans la population. Dès lors, sa valeur fluctuera d'un échantillon à l'autre. On veut donc savoir si, dans la population ces deux variables sont réellement corrélées ou pas. On doit donc réaliser un test d'hypothèse.

    - H0: Pas de corrélation entre les deux variables : ρ = 0 
    - HA: Corrélation entre les deux variables : ρ ≠ 0

    Faisons apparaitre la heatmap des corrélations entre les valeurs numériques : ''')
        st.image(Image.open("assets/corr_num.png"), use_column_width=False, caption='Heatmap Pearson',  width=800)
        #fig = plt.figure(figsize=(16, 6))
        #sns.heatmap(df_1.corr(), vmin=-1, vmax=1, annot=True, cmap='Blues')
        #st.write(fig)

        st.header(''' Correlation entre valeurs catégoriques test de Kendall : ''')
        st.write('''le tau de Kendall est une statistique qui mesure l’association entre deux variables.
    Plus spécifiquement, le tau de Kendall mesure la corrélation de rang entre deux variables.
    Le tau s’interprête de la même façon que les autres coeffcients de corrélation. Sa valeur est comprise entre -1 et 1 : s’il s’approche de 1, on peut supposer l’existence d’une corrélation positive (variation dans le même sens), 
    s’il tend vers -1, on peut dire qu’il existe une corrélation négative et si le tau est proche de 0, il est fort probable qu’il n’y ait aucune liaison entre les 2 variables.  ''')

        from scipy.stats import kendalltau
        corr, _ = kendalltau(df_1['balance'], df_1['job'])
        st.write('- Coefficient de corrélation Kendall entre balance et job : %.5f' % corr)
        corr, _ = kendalltau(df_1['day'], df_1['month'])
        st.write(''' - Coefficient de corrélation Kendall entre day et month : %.5f''' % corr)
        st.write(''' - Grande probabilité qu'il n'y ait aucune liaison entre les variables ''')

