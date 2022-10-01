import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import sys
sys.path.append(r'C:\Program Files (x86)\IronPython 2.7\DLLs')
sys.path.append(r'C:\Program Files (x86)\IronPython 2.7\Lib')
from datetime import datetime
import locale
import shap
import joblib

title = "Démonstration"
sidebar_name = "Démonstration"


def run():

    st.title(title)
    
    st.markdown('---')

    st.markdown(
        """
        ### Testons le modèle ! Le client va-t-il souscrire ?
        """
    )
    
    

    model_RF_final = pickle.load(open(os.getcwd()+'/assets/RF_FINAL_model.sav', 'rb'))

    classes = {0:'Pas souscrit',1:'Souscrit'}
    class_labels = list(classes.values())
    st.markdown('**Objectif** : Donner les caractéristiques de l\'individu afin de savoir s\'il va souscrire au dépôt à terme ou non.')
    st.markdown('Le modèle va prédire si le client va : **souscrire ou non** ')
    
    month_jan = 0
    month_feb = 0
    month_mar = 0
    month_apr = 0
    month_may = 0
    month_jun = 0
    month_jul = 0
    month_aug = 0
    month_sep = 0
    month_oct = 0
    month_nov = 0
    month_dec = 0
    contact_cellular = 0
    contact_telephone = 0
    contact_unknown = 0
        
    def predict_class():

        data = np.array([[age, marital, education, balance, housing, loan, day, pdays, poutcome_success, contact_cellular, contact_telephone ,
                 contact_unknown, month_apr, month_aug, month_dec, month_feb, month_jan, month_jul,  month_jun, month_mar , month_may , month_nov,  month_oct, month_sep]])

        result = model_RF_final.predict(data) 
        #st.write("The predicted class is ",result)
        if result[0] == 1:
            st.markdown("Le modèle prédit que le client va **probablement souscrire au dépôt de compte.** ")
        else:
            st.markdown("Le modèle prédit que le client ne va **probablement pas souscrire au dépôt de compte.**")
        result_proba = pd.DataFrame(model_RF_final.predict_proba(data)).T
        
        #ax = sns.barplot(result_proba ,[0,1], palette="winter", orient='h')
        #ax.set_yticklabels(class_labels,rotation=0)
        #plt.title("Probabilities of the Data belonging to each class")
        #for index, value in enumerate(probs):
        #    plt.text(value, index,str(value))
        colors = ["#FF0B04", "#1fd655"]
        fig = plt.figure()
        plt.figure(figsize=(3,2))
        sns.set_palette(sns.color_palette(colors))
        ax = sns.barplot(result_proba[0] ,result_proba.index, orient='h')
        ax.set_yticklabels(class_labels,rotation=0)
        plt.title("Probablilités que l'individu va souscrire ou non", fontsize=5)
        for container in ax.containers:
            ax.bar_label(container, fontsize=5)
        plt.xlabel('Probabilité', fontsize=5)
        plt.ylabel('Classe', fontsize=5)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        st.pyplot()



        explainer = shap.TreeExplainer(model_RF_final)
        
        #shap_values_specific = explainer.shap_values(data)
        shap_values = joblib.load(os.getcwd()+'/assets/explainer_choosen_instance.bz2')
        shap_values_specific = explainer.shap_values(data)
        shap.initjs()
        fig = shap.force_plot(explainer.expected_value[1], shap_values_specific[1], data,
                        feature_names = ['age', 'marital', 'education', 'balance', 'housing', 'loan', 'day', 'pdays', 'poutcome_success', 'contact_cellular', 'contact_telephone' ,
                 'contact_unknown', 'month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul',  'month_jun', 'month_mar' , 'month_may' , 'month_nov',  'month_oct', 'month_sep'], 
                              matplotlib=True, show=False ,figsize=(16,5))
        st.pyplot(fig)
        plt.clf()
        
    #préparation des liste 
    list_age = list(range(18,100))
    list_marital = ['célibataire', 'divorcé(e)', 'marié(e)']
    list_education = ['primaire', 'secondaire', 'tertiaire']
    list_housing = ['oui', 'non']
    list_loan = ['oui', 'non']

    list_month = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet',
                  'août', 'septembre', 'octobre', 'novembre', 'decembre']

    list_psucess = ['oui', 'non']
    list_contact = ['cellular', 'telephone', 'unknown']
    list_last_day = list(range(1,32))
    list_last_month = ['janvier', 'février', 'mars', 'avril', 'mai','juin','juillet',
                       'août', 'septembre', 'octobre', 'novembre', 'decembre']
    
    
    st.markdown("**Veuillez entrer les caractéristiques de votre client :**")
    
    # les features pour prédire l'individu
    col1, col2, col3 = st.columns(3)
    age = col1.select_slider('L\'age du client', list_age)
    marital = col2.selectbox('La situation maritale du client', list_marital)
    
    if marital == 'célibataire':
        marital = 1
    elif marital == 'divorcé(e)':
        marital = 2
    elif marital == 'marié(e)':
        marital = 3
    
    education = col3.selectbox('Niveau d\'éducation', list_education)
    
    if education == 'primaire':
        education = 1
    elif education == 'secondaire' : 
        education = 2
    elif education == 'tertiaire' :
        education = 3
    
    balance = col1.text_input('Le solde du client (nombre entier) ', '0')
    
    housing = col2.selectbox('A-t-il un crédit immobilier en cours ?', list_housing)
    if housing == 'oui':
        housing = 1
    elif housing == 'non' : 
        housing = 0
        
    loan = col3.selectbox('A-t-il un prêt personnel en cours ?', list_loan)
    if loan == 'oui':
        loan = 1
    elif loan == 'non' : 
        loan = 0
    
    contact = col1.selectbox('Moyen de communication', list_contact)
    if contact == 'cellular':
        contact_cellular = 1
    elif contact == 'telephone' : 
        contact_telephone  = 1
    elif contact == 'unknown' : 
        contact_unknown = 1

    date_camp = st.selectbox('Quand sera-t-il contacté pour la souscription ?', ['Aujourd\'hui', 'Choisir la date'])
    if 'Aujourd\'hui' in date_camp:
        locale.setlocale(locale.LC_TIME, "fr_FR")
        month = datetime.today().strftime("%B")
        if month == 'janvier':
            month_jan = 1
        elif month == 'février' : 
            month_feb  = 1
        elif month == 'mars' : 
            month_mar = 1
        elif month == 'avril' : 
            month_apr = 1
        elif month == 'mai' : 
            month_may = 1
        elif month == 'juin' : 
            month_jun = 1
        elif month == 'juillet' : 
            month_jul = 1
        elif month == 'août' : 
            month_aug = 1
        elif month == 'septembre' : 
            month_sep = 1
        elif month == 'octobre' : 
            month_oct = 1
        elif month == 'novembre' : 
            month_nov = 1
        elif month == 'decembre' : 
            month_dec = 1
        day = datetime.today().strftime("%d")
        
    else:
        col1, col2 = st.columns(2)
        month = col1.selectbox('Mois de contact ?', list_month)
        if month == 'janvier':
            month_jan = 1
            list_day = list(range(1,32))
        elif month == 'février' : 
            month_feb  = 1
            list_day = list(range(1,30))
        elif month == 'mars' : 
            month_mar = 1
            list_day = list(range(1,32))
        elif month == 'avril' : 
            month_apr = 1
            list_day = list(range(1,31))
        elif month == 'mai' : 
            month_may = 1
            list_day = list(range(1,32))
        elif month == 'juin' : 
            month_jun = 1
            list_day = list(range(1,31))
        elif month == 'juillet' : 
            month_jul = 1
            list_day = list(range(1,32))
        elif month == 'août' : 
            month_aug = 1
            list_day = list(range(1,32))
        elif month == 'septembre' : 
            month_sep = 1
            list_day = list(range(1,31))
        elif month == 'octobre' : 
            month_oct = 1
            list_day = list(range(1,32))
        elif month == 'novembre' : 
            month_nov = 1
            list_day = list(range(1,31))
        elif month == 'decembre' : 
            month_dec = 1
            list_day = list(range(1,32))

        day = col2.select_slider('Le jour du mois ?', list_day)
        
    phrase_jour_camp = f"""
    <body>
        <font size="4"
          color="#DC4442">
        <p style="text-align:center"><i> Le client sera contacté le {day} {month}. </i></p>
         </font>
    </body>
    """
    
    st.markdown(phrase_jour_camp, unsafe_allow_html=True)
    
    camp_prec = st.selectbox('A-t-il été déjà contacté lors d\'une précédente campagne ?', ['Veuillez choisir', 'oui', 'non'])
    
    col1, col2, col3 = st.columns(3)
    
    if 'oui' in camp_prec: #si le client a déjà été contacté
        #last_month = col1.selectbox('Mois du dernier contact ?', list_last_month)
        #last_day = col2.selectbox('Jour du dernier contact ?', list_last_month)
        
        pdays = col1.text_input('Contacté il y a x jour(s)', '0', placeholder = 0)
    
        poutcome_success = col2.selectbox('A-t-il souscrit lors de la dernière campagne ?', list_psucess)
        if poutcome_success == 'oui':
            poutcome_success = 1
        elif poutcome_success == 'non' : 
            poutcome_success = 0
            
    else:
        pdays = -1
        poutcome_success = 0

        
    if st.button("Prédire"):
        predict_class()