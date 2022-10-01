import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np


title = "Création de modèles \"simples\""
sidebar_name = "Création de modèles \"simples\""


def run():

    # titre + ligne
    st.title(title)
    st.markdown("---")
    
    # définition des onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stratégie", "Transformation des données", "Métrique ROC AUC", "Entraînements des modèles", "Modèle Random Forest"])
    
    
    ####################
    # Onglet Stratégie #
    ####################
    with tab1:
    
        # Intro
        st.header('Trouver un modèle efficace')
        st.markdown(
            """
            Maintenant que nous avons une bonne compréhension du dataset, on peut commencer la
    création d’un modèle de machine learning pour répondre à notre problématique. Il en existe une
    multitude ayant chacun des avantages et des inconvénients. Il est difficile de dire avec exactitude quel sera
    le modèle le plus efficace pour prédire les clients qui vont souscrire à un dépôt à terme.

    La stratégie choisie ici est donc, dans un premier temps, de lancer plusieurs algorithmes et
    observer ceux qui ont les meilleurs résultats. Nous pourrons ainsi sélectionner le meilleur modèle et
    chercher ensuite à l’optimiser. Nous devons entamer avant tout une phase de pré-processing simple pour
    que les modèles puissent tourner correctement.
            """
        )

        # Train test split
        st.header('Train test split')
        st.markdown(
            """
    Jusqu’à maintenant, nous avons travaillé sur le dataset complet. Cependant, dans le cadre de la
    création d’un modèle, nous devons utiliser une partie des données et garder la seconde pour nous assurer
    que notre prédiction fonctionne.
    On appelle cela le train test split. Nous allons découper le dataset de la façon suivante :
    - 80% des données iront dans le trainset. Le trainset va servir à entraîner le modèle, le paramétrer. Lors de son entraînement, le modèle ne verra que les données du trainset.
    - 20% des données iront dans le testset. Le testset sera mis complètement de côté dans ce projet. Comme notre modèle n’aura jamais vu les données du test set, on pourra donc vérifier si nos prédictions fonctionnent correctement sur des cas concrets.

            """
        )
        st.image(Image.open("assets/train_test_split_2.jpg"), use_column_width=False, caption='Schéma du train test split')
        
        
     
    #####################################
    # Onglet Transformation des données #
    #####################################
    with tab2:
        
        st.header('Pourquoi faut-il transformer les données ?')
        st.markdown(
            """
            Les données ne sont pas intérprétables par un modèle de machine learning en l'état. En effet, derrière ces modèles se cachent des formules mathématiques. Il n'est donc pas possible de faire des calculs avec des données comme 'yes' ou 'no'. Il faut transformer les données pour quelles soient interprétables par nos modèles.
            """)

       # Encodage des valeurs binaires
        st.header('Encodage des valeurs binaires')
        st.markdown(
            """
        L’encodage des valeurs binaires est simple. Il consiste à transformer chaque variable ayant
    uniquement 2 valeurs possibles, en 0 et 1. Dans ce dataset, ‘no’ sera égale à 0, tandis que ‘yes’ sera égal à 1. Il y a 4 colonnes concernées. Voici un avant/après :
            """
        )
        st.image(Image.open("assets/encodage_binaire.png"), use_column_width=False, caption='Avant / après l\'encodage binaire')

        # Création de dummy varialbles pour les colonnes catégoriques
        st.header('Créer des dummy variables pour les colonnes catégoriques')
        st.markdown(
            """
        En ce qui concerne les variables catégoriques comprenant plus de deux sorties, c’est un peu plus
    compliqué que pour l’encodage de données binaire.
    Ici, nous allons créer automatiquement une nouvelle variable binaire, pour chacune des valeurs
    disponibles. Voici un exemple avec la colonne poutcome (avant / après) :
            """
        )
        st.image(Image.open("assets/dummy_variable.png"), use_column_width=False, caption='Avant / après la création de dummy variable. Colonnes concernées : job, marital, eudcation, contact, month, poutcome.')

        # Features scaling des données numériques
        st.header('Features scaling des données numériques')
        st.markdown(
            """
        Les données numériques peuvent déjà être lues par un modèle de machine learning sans être
    modifié. Cependant, les données ne sont pas toujours à la même échelle, et cela peut avoir un impact très
    négatif lors de l'entraînement du modèle. Typiquement, l’échelle de l’âge qui va de 18 à 95 ans, n’est pas
    du tout la même que celle de balance (-6847 jusqu’à 66653 sur le trainset).
    Il existe plusieurs méthodes pour mettre les données numériques sur une même échelle. Nous utilisons le
    StandardScaler pour le moment.
            """
        )
        st.image(Image.open("assets/feature_scaling.png"), use_column_width=False, caption='Avant / après l\'utilisation du StandardScaler')

        # Information du trainset
        st.header('Information du trainset')
        st.markdown(
            """
        On avait initialement 17 colonnes. On a supprimé 'Duration' et 'Campaign' car nous ne sommes pas supposé avoir accès a ces informations au moment de la prédiction. On a également mis de côté 'Deposit' qui est notre target. Nous avions donc 14 colonnes. Suite au pre-processing, nous avons les 46 colonnes ci-dessous :
            """
        )
        st.image(Image.open("assets/pre-processing_simple.png"), use_column_width=False, caption='Information sur le trainset après l\'application du pré-processing.')
        
        
    ###########################    
    # Onglet Métrique ROC AUC #
    ###########################
    with tab3:
    
        # Définition de la métrique principale : ROC AUC
        st.header('Définition de la métrique principale : ROC AUC')
        st.markdown(
            """
        La métrique ROC AUC est celle qui nous intéresse le plus, du moins dans le cadre de
    l'entraînement de nos modèles de machine learning. La courbe ROC (receiver operating characteristic) se
    base sur des probabilités que la target soit oui ou non, là où pour les autres métriques le modèle va
    prédire de manière catégorique un oui ou un non.

    Quand on utilise des probabilités, le modèle peut plus ou moins être sûr de lui. Il va par exemple
    prédire un oui à 90%, ou alors un oui à 51% en cas de doute. La courbe ROC permet de classer toutes les
    prédictions de la plus forte probabilité à la plus faible dans un graphique en 2D. En ordonnées, on a le taux
    de True Positive trouvé, et en abscisse le taux de False Positive trouvé.
    Voici à quoi ressemble une courbe ROC :
            """
        )
        st.image(Image.open("assets/exemple_roc_auc.png"), use_column_width=False, caption='Exemple de ROC AUC')
        st.markdown(
            """
        - La ligne bleu clair dans le coin supérieur gauche montre le ROC parfait. On trouve d’abord tous les éléments de la classe que l’on souhaite, puis on finit par trouver les éléments de la classe que l’on ne souhaite pas.

    - La ligne en tiret bleu montre une prédiction aléatoire. En jouant à pile ou face, on a va trouver aussi rapidement la classe recherchée que la classe non recherchée. C’est pourquoi le taux de vrai et faux positif avance au même rythme. C’est le pire modèle possible.

    - La courbe orange est un exemple de prédiction. Elle doit normalement toujours se situer entre la prédiction parfaite et la prédiction aléatoire. Si ce n’est pas le cas, il y a une erreur dans le code.

    AUC (Area under the curve) signifie l’aire sous la courbe. Le ROC AUC consiste donc à tracer une
    courbe, puis à mesurer la surface sous celle-ci. Un modèle parfait aura un aire de 100%, le modèle
    aléatoire de 50%. Pour maximiser ce score, il faut donc s’approcher au maximum des 100%.

    Notre but étant de sélectionner un panel réduit de client mais qui ont une très grande chance de
    souscrire, cette métrique est très utile, car elle permet de classer du mieux possible l’ordre des clients
    susceptibles de souscrire à l’offre.
    Maintenant que nous avons défini nos métriques utiles dans le cadre du projet, nous allons
    pouvoir entamer nos modèles de machines learning, à un détail près.    
            """
        )
        
    
    
    ####################################   
    # Onglet Entraînements des modèles #
    ####################################
    with tab4:
        
        # La cross validation
        st.header('La cross-validation')
        st.markdown(
            """
        Nous devons faire une cross-validation. Il s’agit de couper le dataset en plusieurs parties (en 4 dans notre projet). On va entraîner notre modèle 4 fois. A chaque fois, on va prendre 3 parties des données, et on va se servir pour essayer de prédire la dernière non utilisée. Cela permet de simuler des données de test.
            """
        )
        st.image(Image.open("assets/cross_validation.png"), use_column_width=False, caption='Schéma de la cross-validation')
        st.markdown(
            """
        On réalise donc cette opération 4 fois puis on fait la moyenne des résultats obtenus. On intègre donc dans une fonction cross_validate, le modèle, les données, nos métriques.    
            """
        )
        st.image(Image.open("assets/code-cross_validation.png"), use_column_width=False, caption='La cross validation utilisée pour les modèles')

        # Liste des modèles
        st.header('Liste des modèles')
        st.markdown(
            """
        Il ne nous reste plus qu’à créer des modèles simples (aucune personnalisation du modèle, utilisation des paramètres de base). On ne va pas détailler ici le principe de tous ces modèles, on le fera uniquement pour celui que l’on retient.
            """
        )
        st.image(Image.open("assets/liste_modele.png"), use_column_width=False, caption='Les 10 modèles testés lors de cette 1ère phase.')

        # Résultats
        st.header('Résultats')
        st.markdown(
            """
        Les 2 meilleurs modèles en se basant sur le ROC AUC sont Random Forest et SVM. C’est également ces 2 modèles qui obtiennent le meilleur score dans au moins l’une des métriques affichées.
            """
        )
        st.image(Image.open("assets/modèles_simples.png"), use_column_width=False, caption='Résumé des résultats. Le meilleur score de chaque métrique est affiché en vert.')
        st.markdown(
            """
        Sur ce projet, l’interprétabilité du modèle est fondamentale. C’est pourquoi nous allons mettre de
    côté le SVM, qui offre moins de possibilités à ce sujet. C’est également un modèle plus lent que le Random
    Forest. Nous décidons donc d’optimiser le Random Forest.
            """
        )
        
     
    ###############################   
    # Onglet Modèle Random Forest #
    ###############################
    with tab5:
    
        # Le modèle random forest
        st.header('Le modèle Random Forest')
        st.markdown(
            """
            Le Random Forest (Forêt aléatoire) fait appel à plusieurs Decision Tree en voici un exemple : 
            """
        )
        st.image(Image.open("assets/exemple-decision_tree.png"), use_column_width=False, caption='Exemple de Decision Tree')
    #     col1, col2, col3 = st.columns([1,6,1])
    #     col2.image(Image.open("assets/exemple-decision_tree.png"), use_column_width=False, caption='Exemple de Decision Tree')
        st.markdown(
            """
            On y ajoute également
    une composante aléatoire, en limitant le nombre de données pour chaque arbre. Le Random Forest va
    ensuite faire une moyenne des différents Decision Tree. Ce modèle va donc gagner en
    performance et réduire l’overfitting. En revanche, il va perdre un peu de son interprétabilité.
            """
        )