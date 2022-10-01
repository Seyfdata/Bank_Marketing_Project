import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

title = "Analyse exploratoire des données"
sidebar_name = title

def run():
    
    st.title(title)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Jeu de données", "Visualisation des données", "Aller plus loin"])
    
    with tab1 :
    
        st.header('Présentation du jeu de données')

        df = pd.read_csv("assets/bank.csv", sep=",", header=0)
        AgGrid(df)
        st.header('Explication des variables :')
        st.text("    1 - age : âge du client \n\
    2 - job : métier du client \n\
    3 - marital : statut marital du client \n\
    4 - education : niveau d'étude du client \n\
    5 - default : si le client a un crédit impayé \n\
    6 - balance : somme d'argent sur le compte bancaire \n\
    7 - housing : si le client a un emprûnt immobilier \n\
    8 - loan : si le client à un crédit en cours \n\
    9 - contact : type de contact (cellular, phone, unknown) \n\
    10 - day : le jour du mois où le client a été contacté \n\
    11 - month : le mois où le client a été contacté \n\
    12 - duration : le temps en ligne avec le client \n\
    13 - campaign : combien de fois le client a été contacté au cours de cette campagne \n\
    14 - pdays : combien de jours se sont écoulés depuis le dernier contact \n\
    15 - previous : combien de fois le client a été contacté au cours de la \n\
                    précedente campagne \n\
    16 - poutcome : résultat de la précédente campagne \n\
    17 - deposit :  résultat de la présente campagne")


    with tab2: 
        st.header('Visualisation des données')


        st.markdown(
        """
        * **Distribution de la variable cible 'deposit**
        
        """
        )        
        st.image(Image.open("assets/distri_deposit.png"), use_column_width=False, caption='Distribution de la variable cible deposit',  width=400)
        st.write(''' Une variable cible plutôt équilibrée''')
        
        st.markdown(
        """
        * **Aspect trimestriel des campagnes**
        
        """
        ) 
        st.image(Image.open("assets/trimestres.png"), use_column_width=False, caption='Aspect trimestriel des campagnes',  width=600)
        st.write(''' Il y a eu 2 grosses campagnes d'appels, autour de 90 et 180 jours qui ont un eu très bon succès. On a également deux campagnes d'appels plus discrètes, autour de 270 jours et de 360 jours. La banque semble faire des campagnes par trimestre car 90, 180, 270 et 360 jours représentent respectivement 3, 6, 9 et 12 mois''')

        st.markdown(
        """
        * **Distribution de pdays en fonction de deposit**
        
        """
        ) 
        st.image(Image.open("assets/distri_pdays.png"), use_column_width=False, caption='Distribution de pdays en fonction de la variabe cible deposit',  width=800)
        st.write(''' Il y a beaucoup de clients avec pdays = -1. Ce sont les clients qui n'ont jamais été contactés auparavant. On constate que les clients déjà connus de la banque sont plus susceptibles de souscrire le DAT.''')
    
        st.markdown(
        """
        * **Distribution de poutcome en fonction de deposit**
        
        """
        ) 
        st.image(Image.open("assets/poutcome.png"), use_column_width=False, caption='Distribution de la variable cible deposit',  width=800)
        st.write(''' Il y a énormément d'unknown. Cela correspond à tous les nouveaux clients. C'est logique, nous ne pouvons pas avoir de résultat précédent, sur une personne que nous n'avons jamais appelée. Unknown correspond donc à previous = 0 ou pdays = -1''')
        st.write(''' - “Success” correspond aux clients ayant dit oui lors d'une précédente campagne. La campagne actuelle connaît un franc succès pour ces clients (91%) ''')
        st.write(''' - “Failure” et “other” sont assez proches. “Failure” correspond à un refus du client lors de la précédente campagne. “Other” correspond à un client qui a fait un retour positif mais un autre produit que le DAT. On pourra envisager de regrouper “other” et “faillure”, car les volumes sont très faibles.''')

        st.markdown(
        """
        * **Distribution des crédits en fonction de la variable cible**
        
        """
        ) 
        st.image(Image.open("assets/credit.png"), use_column_width=False, caption="Distribution des crédits ensemble en fonction de deposit",  width=700)
        st.write(''' Avoir deux crédits en cours réduit les chances de souscrire au DAT, mais l'écart n'est pas très significatif pour autant (avec un seul). ''')

        st.markdown(
        """
        * **Distribution de l'âge en fonction de la variable cible**
        
        """
        ) 
        st.image(Image.open("assets/distri_age.png"), use_column_width=False, caption="Distribution de l'âge en fonction de la variable cible",  width=600)
        st.write('''  Les clients de 19-29 ans et les 60+ ont plus tendance à dire oui. En revanche pour les 30-59 ans, c'est le non qui l'emporte.''')

        st.markdown(
        """
        * **Distribution de la balance en fonction de deposit**
        
        """
        ) 
        st.image(Image.open("assets/distri_balance.png"), use_column_width=False, caption="Distribution de la balance en fonction de deposit",  width=600)
        st.write(''' On observe moins de souscriptions pour les plus faibles balances. Cela s'équilibre à partir de 1000 euros.''')


    with tab3: 
 
        st.markdown(
        """
        * **Distribution de duration en fonction de deposit**
        
        """
        ) 
        st.image(Image.open("assets/distri_duration.png"), use_column_width=False, caption="Distribution de duration en fonction de deposit",  width=600)
        st.write(''' On observe une tendance très nette. Plus la durée de l'appel est élevée, plus le client a de chance de souscrire. La durée semble avoir un impact très fort sur deposit. Cependant, nous ne pourrons pas utiliser cette donnée dans le cadre du modèle prédictif. En effet, la durée n'est connue qu'une fois que l'appel est effectué. Or, nous souhaitons justement prédire si c'est pertinent ou non de le contacter. ''')

        st.markdown(
        """
        * **Distribution de campaign en fonction de deposit**
        
        """
        ) 
        st.image(Image.open("assets/distri_campaign.png"), use_column_width=False, caption="Distribution de campaign en fonction de deposit",  width=600)
        st.write(''' De prime abord, on pourrait croire que plus on appelle, moins on est efficace. Cependant, ‘campaign’ = 1 montre uniquement les clients ayant fait un seul appel. Pourtant un client avec ‘campaign’ = 2 a refusé la proposition lors du premier appel. On va retravailler les données pour avoir une meilleure vision du taux d'acceptation. 
                     Comme pour ‘duration’, ce champ ne sera pas utilisé pour le modèle prédictif, car nous ne connaissons pas cette donnée tant que la campagne n'est pas terminée. On pourra tout de même étudier le taux de retour en fonction du nombre d'appels et le prendre en compte lors des futures campagnes.
            ''')
        
        st.markdown(
        """
        * **Distribution des métiers en fonction de la variable cible**
        
        """
        ) 
        st.image(Image.open("assets/job.png"), use_column_width=False, caption="Distribution des métiers en fonction de deposit",  width=700)
        st.write(''' 75% des étudiants souscrivent au DAT, suivi des retraités avec 66%, il semble donc que les personnes qui ne font pas partie de la population active sont plus susceptibles de souscrire un DAT. On peut aussi noter que certains emplois comme col-bleu ou entrepreneur ont un résultat en deçà de la moyenne.''')

        st.markdown(
        """
        * **Distribution de la variable education en fonction de la variable cible**
        
        """
        ) 
        st.image(Image.open("assets/education.png"), use_column_width=False, caption="Distribution de la variable education en fonction de deposit",  width=700)
        st.write(''' Nous constatons que plus le niveau d'éducation est élevé, plus on a de chance de souscrire un DAT.''')

        st.markdown(
        """
        * **Distribution de la variable credit en fonction de la variable job**
        
        """
        ) 
        st.image(Image.open("assets/distri_credit_job.png"), use_column_width=False, caption="Distribution de la variable credit en fonction de job",  width=700)
        st.write(''' Il y a plus de 80% des retraités et des étudiants qui n’ont aucun crédit en cours. Cela explique en partie pourquoi ce sont les deux job qui ont le meilleur taux de souscription. ''')
