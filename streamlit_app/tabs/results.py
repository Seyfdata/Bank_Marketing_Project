import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image

# Scoring
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, make_scorer ,roc_curve
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
import joblib

title = "Résultats du modèle"
sidebar_name = title


def run():
    # Option pour cacher le warning
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.title(title)
    
    st.markdown("---")

    # charger les données
    X_train = pickle.load(open(os.getcwd()+'/assets/data_X_train.sav', 'rb'))
    X_test = pickle.load(open(os.getcwd()+'/assets/data_X_test.sav', 'rb'))
    y_train = pickle.load(open(os.getcwd()+'/assets/data_y_train.sav', 'rb'))
    y_test = pickle.load(open(os.getcwd()+'/assets/data_y_test.sav', 'rb'))
    df = pickle.load(open(os.getcwd()+'/assets/full_data.sav', 'rb'))
    
    model_RF_final = pickle.load(open(os.getcwd()+'/assets/RF_FINAL_model.sav', 'rb'))
    
    explainer = shap.TreeExplainer(model_RF_final)
    shap_values = joblib.load(os.getcwd()+'/assets/explainer_choosen_instance.bz2')
    shap.initjs()
    
    probs_test = model_RF_final.predict_proba(X_test)
    y_pred = model_RF_final.predict(X_test)
    
    tab1, tab2, tab3 = st.tabs(["Score des métriques", "Interprétabilité du modèle", "Limites du dataset et recommandations pour le futur"])
    
    with tab1:
    
        st.markdown(
            """
            Dans cette partie, nous allons vous présenter les résultats de notre modèle final.

            Vu précedemment, le modèle final créé nous donne de meilleurs scores, malgré un plus bas recall. Cela nous nous dit qu'il faut appeler plus de clients, quitte à faire moins de réussites.
            
            Si nous comparons la validation score avec le trainset, nous voyons bien l'existence de l'overfitting, nous avons des métriques élevés avec notre trainset: on a un une precision et recall qui sont environ inférieur à 7% sur le score de validation.
            """
    )
        
        st.image(Image.open("assets/metrics_validcross_trainset.png"), use_column_width=False)
    
        
        st.markdown(
        """
        * **Courbe ROC AUC**
        
        """
    )
    
    # Courbe de ROC du traintset
        probs = model_RF_final.predict_proba(X_train)

        # calcul du roc auc
        fpr, tpr, seuils = roc_curve(y_train, probs[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        # affichage
        fig = plt.figure()
        plt.figure(figsize=(3,2))
        fig.add_subplot(211)
        plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', color='grey', label='Prédiction parfaite (auc = 1.0)', lw=0.8)
        plt.plot([0, 1], [0, 1], color='black', lw=0.8, linestyle='--', label='Prédiction aléatoire (auc = 0.5)')
        plt.plot(fpr, tpr, color='skyblue', lw=1, label='Prédiction du trainset (auc = %0.2f)' % roc_auc)
        plt.xlim([-0.006, 1.0])
        plt.ylim([0.00, 1.006])
        plt.xlabel('Taux faux positifs', fontsize=5)
        plt.ylabel('Taux vrais positifs', fontsize=5)
        plt.title('Courbe ROC du RF final', fontsize=5)
        plt.legend(loc="lower right", fontsize=5)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)

    #Courbe ROC du testset


    # roc auc test
        fpr, tpr, seuils = roc_curve(y_test, probs_test[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        # figure
        fig.add_subplot(212)
        plt.plot(fpr, tpr, color='red', lw=1, label='Prédiction du testset (auc = %0.2f)' % roc_auc)
        #plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Prédiction aléatoire (auc = 0.5)')
        #plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', color='blue', label='Prédiction parfaite (auc = 1.0)')
        #plt.xlim([-0.006, 1.0])
        #plt.xlabel('Taux faux positifs', fontsize=5)
        #plt.ylabel('Taux vrais positifs', fontsize=5)
        plt.legend(loc="lower right", fontsize=5)
        #plt.xticks(fontsize=5)
        #plt.yticks(fontsize=5)
        plt.show();
        st.pyplot()

        st.markdown(
            """
            En affichant la courbe ROC, nous voyons que le modèle final donne de bons résultats sur les données test.

            * **Courbe des gain**
            """
        )

        tab1_1, tab1_2 = st.tabs(["Vision classique", "Version horizontale"])
    # Cumulative gain curve
    # on calcul le "score", difference entre proportion de deposit trouvé vs proportion de contact
        with tab1_1:
            #train
            # best line
            best_prediction_train = y_train.value_counts(normalize=True)[1]
            probs_train = model_RF_final.predict_proba(X_train)
            # prep dataset pour afficher la prediction du model
            train_df = pd.DataFrame(y_train)
            train_df['pred'] = probs_train[:,1]
            train_df = train_df.sort_values('pred', ascending=False)
            train_df['cumul_deposit'] = train_df['deposit'].cumsum()
            train_df['prop_deposit'] = train_df['cumul_deposit'] / train_df['cumul_deposit'].max()
            train_df['contact'] = [x for x in range(1,(train_df.shape[0])+1)]
            train_df['prop_contact'] = train_df['contact'] / train_df.shape[0]

            train_df['reussite'] = train_df['cumul_deposit'] / train_df['contact']
            train_df = train_df.join(X_train)

            train_df['RF Prediction'] = train_df['prop_deposit'] - train_df['prop_contact']

            
            
            # test : prep dataset pour afficher la prediction du model
            test_df = pd.DataFrame(y_test)

            # test : best line 
            best_prediction = y_test.value_counts(normalize=True)[1]

            test_df['pred'] = probs_test[:,1]
            test_df = test_df.sort_values('pred', ascending=False)
            test_df['cumul_deposit'] = test_df['deposit'].cumsum()
            test_df['prop_deposit'] = test_df['cumul_deposit'] / test_df['cumul_deposit'].max()
            test_df['contact'] = [x for x in range(1,(test_df.shape[0])+1)]
            test_df['prop_contact'] = test_df['contact'] / test_df.shape[0]

            test_df['reussite'] = test_df['cumul_deposit'] / test_df['contact']
            test_df = test_df.join(X_test)

            test_df['RF Prediction'] = test_df['prop_deposit'] - test_df['prop_contact']
            
            # figure
            plt.xlim(0,1)
            plt.ylim(0,1.05)
            plt.xlabel('Percentage of contact', fontsize=5)
            plt.ylabel('Percentage of target', fontsize=5)
            plt.grid()
            plt.xticks(np.arange(0, 1.1, step=0.1), fontsize=5)
            plt.yticks(np.arange(0, 1.1, step=0.1), fontsize=5)


            # random line
            plt.plot([0,1], [0,1], '--', color='black', label='Random prediction', linewidth=1)

            # best line
            best_prediction = y_test.value_counts(normalize=True)[1]
            plt.plot([0, best_prediction, 1], [0,1, 1], '--', color='grey', label='Perfect prediction')

            # prediction line train et test set
            plt.plot(train_df['prop_contact'], train_df['prop_deposit'], color='skyblue', label='RF prediction trainset', linewidth=1)
            plt.plot(test_df['prop_contact'], test_df['prop_deposit'], color='red', label='RF prediction testset', linewidth=1)

            plt.title('Courbe de gain cumulative du RF final', fontsize=5);

            # légende
            plt.legend(fontsize=5);
            st.pyplot()

            # random line
            plt.plot([0,1], [0,0], '--', color='black', label='Prédition aléatoire', linewidth=0.8)
            st.markdown(
                    """
                    Trainset : On peut voir qu’en contactant 50% des clients, on arrive à trouver plus de 80% de la target. Il ne
            faut pas oublier qu’on regarde le trainset, et que ce dernier est en overfitting. Il faut minorer le résultat affiché.

            Testset: Le point maximum se trouve autour de 45% de contact. Mais la courbe ne monte plus vraiment à
            partir de 30%.
                    """
                )
 
        with tab1_2:
            # perfect line
            test_df[test_df['contact'] == (test_df['deposit'] == 1).sum()]
            plt.plot([0, best_prediction_train, 1], [0,1-best_prediction_train, 0], '--', color='grey', label='Prédiction parfaite', linewidth=0.8)
            # info graphique
            plt.xlabel('Percentage of contact', fontsize=5)
            plt.ylabel('Percentage of target - Percentage of contact', fontsize=5)
            plt.grid()
            plt.xticks(np.arange(0, 1.1, step=0.1))
            plt.xlim(0,1)
            plt.ylim(-0.0001,0.55)
            # test: predicted line
            plt.plot(test_df['prop_contact'], test_df['RF Prediction'], color='red', label='Prediction du test_set', linewidth=1)
            # test : max predicted line
            max_predicted_vertical_test = test_df['RF Prediction'].max()
            max_predicted_horizontal_test = test_df[test_df['RF Prediction'] == max_predicted_vertical_test]['prop_contact']
            plt.scatter(max_predicted_horizontal_test, max_predicted_vertical_test, c='maroon', marker='+', s=100, label='Score max prediction testset', linewidth=1)
            # train : prediction line
            plt.plot(train_df['prop_contact'], train_df['RF Prediction'], color='skyblue', label='Prediction trainset', linewidth=1)
            # train : max predicted line
            max_predicted_vertical_train = train_df['RF Prediction'].max()
            max_predicted_horizontal_train = train_df[train_df['RF Prediction'] == max_predicted_vertical_train]['prop_contact']
            plt.scatter(max_predicted_horizontal_train, max_predicted_vertical_train, c='darkblue', marker='+', s=100, label='Score Max trainset', linewidth=1)
            plt.legend(fontsize=3)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            plt.title('Courbe de gain cumulative du RF final', fontsize=5);
            st.pyplot()
            st.markdown(
                    """
            Le problème de la courbe de gain est qu’il ne nous montre pas le seuil de probabilité à prendre pour considérer que le client va souscrire. Comme précisé auparavant, celle-ci est dépendante de la distribution de deposit: lors de la prochaine campagne, la distribution peut être totalement différente.
                    C’est pourquoi il est important de connaître le seuil de prédiction (Threshold).
                    """
                )

        ## precision recall
        st.markdown(
            """
            * **Courbe de Precision/Recall**

            La courbe de precision recall traduit le comportement global de notre modèle. Elle reflète bien un problème que rencontre la banque: le biais exploration/exploitation.

            """        
        )
        
        tab2_1, tab2_2 = st.tabs(["Courbe precision-recall du testset", "Courbe precision-recall du trainset"])
    
        with tab2_1:
        
            # precision recall curve
            precision_t, recall_t, threshold_t = precision_recall_curve(y_test, probs_test[:,1])

            # figure
            plt.figure(figsize=(10,8))
            plt.plot(threshold_t, precision_t[:-1], label='precision_t', linewidth=2)
            plt.plot(threshold_t, recall_t[:-1], label='recall_t', linewidth=2)

            # détail figure
            plt.grid()
            plt.xticks(np.arange(0, 1.1, step=0.1))
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel('Threshold / Probabilité')
            plt.ylabel('Score métrique')
            plt.title('Precision Recall Curve du modèle Random Forest sur le testset')

            plt.legend();
            st.pyplot()

            st.markdown(
                """
                Le graphique valide nos esimations, on peut reprendre ce graphique pour une future campagne si nous la répartition de la target est similaire.
                """)
             
        
        with tab2_2:
            # on calcule la precision recall curbe
            precision, recall, threshold = precision_recall_curve(y_train, probs_train[:,1])

            # affichage de recall et precision
            plt.figure(figsize=(10,8))
            plt.plot(threshold, precision[:-1], label='precision', linewidth=2)
            plt.plot(threshold, recall[:-1], label='recall', linewidth=2)

            # info graphique
            plt.grid()
            plt.xticks(np.arange(0, 1.1, step=0.1))
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel('Threshold / Probabilité')
            plt.ylabel('Score métrique')
            plt.title('Precision Recall Curve du modèle Random Forest sur le trainset')

            plt.legend();
            st.pyplot()
            st.markdown(
                """
                Sur le trainset, on voit par exemple qu’avec un seuil de 0.6, on a presque 90% de précision. Cela veut dire que 9 clients sur 10 souscrivent. On trouverait pour cet exemple 60% des clients prêts à souscrire.
                """)
            
        st.markdown('---')

        st.markdown(
            """
            * **Matrice de confusion et résultats final pour Threshold = 0.5**

            """
        )
        st.image(Image.open("assets/matrix_confusion_testset_final.png"), width=150, use_column_width=False)

        st.image(Image.open("assets/metrics_testset_final.png"), use_column_width=False)

       
    
    with tab2:
        
        st.markdown(
        """
        ### Interprétabilité globale
        
        
        """        
    )
    
        tab3_1, tab3_2, tab3_3 = st.tabs(["Importance moyenne des variables", "Importance détaillée des variables", "Arbre de décision sur le modèle final"])

        with tab3_1:
            col1, col2, col3 = st.columns([1,6,1])
            col2.image(Image.open("assets/Shap Value.png"), use_column_width=False)
            st.markdown(
            """
            Ce graphique nous montre l'impact moyen de chaque feature dans le modèle: c'est la moyenne en valeur absolue.
            La variable la plus importante serait la variable `contact`; il est important d'avoir une explication pour `contact_unknonwn`, suivi du précédent succès de la campagne `poutcome` puis de la détention d'un crédit immobilier `housing`.
            """        
        )
        with tab3_2:
            col1, col2, col3 = st.columns([1,6,1])
            col2.image(Image.open("assets/Shap Summary_plot.png"), use_column_width=False)
            st.markdown(
            """
            Pour chaque variable, nous voyons l'impact qu'elle a sur la prédiction du modèle.

            Avoir un précédent succès a un impact très positif sur la souscription, contrairement à `housing`qui a un impact néfatif pour la prédiction.

            Pour plusieurs features, nous sommes très partagés, on voit que c'est mélangé comme `age`.
            """        
        )
            
           
        with tab3_3:
            st.image(Image.open("assets/interpretabilite_decision_tree2.png"), width=1100, use_column_width=False)
            st.markdown(
                """
                L'arbre de décision basé sur le modèle pour comprendre comment le modèle s'est créé. Ca reste une méthode très explicite mais il est difficile d'avoir une vision au client et nous ne voyons pas clairement les variables les plus impactantes pour notre modèle.
                """
        )

        st.markdown('---')

        st.markdown("""
        ### Interprétabilité locale
        """)

        st.markdown(
            """
            Intérressons nous à quelques cas clients du testset.

            """
        )
        
        tab3_2_1, tab3_2_2, tab3_2_3 = st.tabs(["Cas client 1", "Cas client 2", "Cas client 3 "])
        with tab3_2_1:
            client = 1567
            st.write("Regardons le client ", client) 
            st.write(df.loc[[client]])

            specific_instance = X_test.loc[[client]]
            shap_values_specific = explainer.shap_values(specific_instance)
            shap.initjs()
            shap.force_plot(explainer.expected_value[1], shap_values_specific[1], specific_instance, matplotlib=True, show=False ,figsize=(16,5))
            st.pyplot()
            plt.clf()
            st.markdown(
                """
                Ce client devrait à 87% choisir de souscrire au dépôt de compte. Les élements ayant un effet positif seraient d'abord lié à son âge, suivi de l'appel en avril, et qu'il n'a pas de crédit immobilier.
                """
            )
        with tab3_2_2:
            client = 7679
            st.write("Regardons le client ", client) 
            st.write(df.loc[[client]])

            specific_instance = X_test.loc[[client]]
            shap_values_specific = explainer.shap_values(specific_instance)
            shap.initjs()
            shap.force_plot(explainer.expected_value[1], shap_values_specific[1], specific_instance, matplotlib=True, show=False ,figsize=(16,5))
            st.pyplot()
            plt.clf()
            st.markdown(
                """
                Ce client aurait 15% de chance de choisir de souscrire: ceci est expliqué par le moyen de communication inconnu, du mois de contact et de son solde de compte.
                """
            )
        with tab3_2_3:
            client = 8582
            st.write("Regardons le client ", client) 
            st.write(df.loc[[client]])

            specific_instance = X_test.loc[[client]]
            shap_values_specific = explainer.shap_values(specific_instance)
            shap.initjs()
            shap.force_plot(explainer.expected_value[1], shap_values_specific[1], specific_instance, matplotlib=True, show=False ,figsize=(16,5))
            st.pyplot()
            plt.clf()
            st.markdown(
                """
                Le modèle prédit que ce client a 90% de chance de souscrire: en effet il a déjà souscrit précédemment à un compte dépôt à terme, il n'a pas de crédit immobilier, c'est un jeune adulte avec une balance importante. Cependant il n'a pas souscrit.
                """
            )
        
    with tab3:
        
        st.markdown(
        """
        ### Limites du dataset et recommandations pour le futur
        
        """
        )
        
        tab4_1, tab4_2, tab4_3 = st.tabs(["Manque d'informations sur certaines données", "Manque de features", "Recommandations de contacts"])
        with tab4_1:
                st.markdown(
                    """
                    * Présence de la modalité de `unknown` sur des variables primordiales comme `contact`
                    * Comprendre les circonstances, dont les pics des `contact`, on émet des hypothèses mais sans confirmations du métier, on ne peut pas pousser l'exploitation
                    """
                )
                fig = plt.figure(figsize=(13, 4))
                palette = {
                    'yes': 'tab:green',
                    'no': 'tab:red'
                }
                sns.histplot(data=df[(df['pdays']>-1) & (df['pdays'] <=600)], x='pdays', hue='deposit', kde=True, binwidth=10, palette = palette)
                plt.xticks(ticks=np.arange(0,630,30))
                st.pyplot(fig)
                st.markdown(
                    """
                    * Variable cible `deposit`: est-ce toujours la même offre entre les différentes campagnes ?
                    * Connaître la politique de contacts pour être en norme

                    """
                )
        with tab4_2:
                st.markdown(
                    """
                    * Avoir l'année de campagne permettrait de connaître le contexte économique
                    * Avoir plus d'informations sur la/les dernière(s) campagne(s) des clients (duration, combien de campagnes différentes,...)
                    * Le solde du client n'est pas un indicateur suffisant, il faudrait d'autres informations (épargne, solvabilité et taux d'endettement...) qui peuvent jouer un rôle sur la suscription du dépôt de compte
                    * Un identifiant unique au client est nécessaire pour éviter les doublons

                    """
                )
                st.write(df.loc[(3213,2495,5092,3870),:])

        with tab4_3:
                st.markdown(
                    """
                    * Contacter uniquement les clients avec une prédiction positive de notre modèle
                    * Contacter les clients en début de mois
                    * Contacter les clients sur des mois favorables
                    """)
                st.image(Image.open("assets/recomm_mois_fav_call.png"), use_column_width=False)
                st.markdown(
                    """
                    * Limiter le nombre d'appels par campagne

                    """
                )