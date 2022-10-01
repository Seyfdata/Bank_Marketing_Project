import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, learning_curve


title = "Optimisation d'un Random Forest"
sidebar_name = title


def run():

    st.title(title)
    
    st.markdown('---')

    def modele_result(features,modele):
                # importation des données
            df = pd.read_csv("assets/bank.csv")
            # on découpe le dataframe en 2, un trainset et un testset
            trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
            
            def encodage(df):
                
                # gestion des valeurs binaire, oui/non ; vrai/faux
                # deposit, loan, housing....
                code = {'yes':1,
                        True:1,
                        'no':0,
                        False:0}
                
                for col in df.select_dtypes('boolean').columns:
                    df[col] = df[col].map(code)  
                
                for col in df.select_dtypes('object').columns:
                    if (len(df[col].unique()) == 2):
                        df[col] = df[col].map(code)
                        df[col] = df[col].astype('uint8')
                        
                # création de dummy variable pour les colonnes ayant plus de deux variables comme job, marital, poutcome...
                col_categoric = df.select_dtypes(['object', 'category']).columns
                df = df.join(pd.get_dummies(df[col_categoric], prefix=col_categoric))
                # on supprime les anciennes colonnes
                df = df.drop(col_categoric, axis=1)
                
                return df
                
                
            def feature_scaling(df, is_testset):
            
                    # scaler numeric
                col_numeric = df.select_dtypes(['int64']).columns
                    
                    # On scale les données sur le trainset
                    # Si trainset
                if is_testset == False:
                    df[col_numeric] = scaler.fit_transform(df[col_numeric])
                    # Si nouvelle données / testset
                if is_testset == True:
                    df[col_numeric] = scaler.transform(df[col_numeric])
                    
                return df
                
                    # appelle les fonctiones précédentes
            # un Dataframe et un boolean (si trainset : false, si autres true)
            def preprocessing(df, is_testset):
                
                df = encodage(df)
                df = feature_scaling(df, is_testset)

                #X = df.drop('deposit', axis=1)
                X = df[features]
                y = df['deposit']
                
                return X, y
            
              
                # on fait une copie des données
            train = trainset.copy()
            test = testset.copy()

            # scaler qui sera appliqué avec la fonction feature_scaling
            scaler = MinMaxScaler((0,1))
            
            

            # on applique la fonction de preprocessing pour générer X_train, y_train etc
            X_train, y_train = preprocessing(df=train, is_testset=False)
            X_test, y_test = preprocessing(df=test, is_testset=True)
    

            RF_clf = modele
            RF_clf.fit(X_train,y_train)

            y_pred = RF_clf.predict(X_test)
            
            classifiers = [
               ('Random Forest Classifier',modele)
              ]
            
            # on créé un dataframe pour accueillir les résultats
            models_scores_standard_scaler = pd.DataFrame(columns= ['Random Forest Classifier'],
                                   index=   ['Accuracy', 'Precision', 'Recall', 'F1','Roc_auc']
                                  )
            # pour chaque modèle 
            for clf_name, clf in classifiers:
    
                # on fait une cross validation sur les 5 métriques
                scores = cross_validate(clf, X_train, y_train,
                            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], 
                            cv=4, n_jobs=-1)
    
                # on enregistre la moyenne des score dans le dataframe
                models_scores_standard_scaler[clf_name] = [scores['test_accuracy'].mean(),
                                     scores['test_precision'].mean(),
                                     scores['test_recall'].mean(),
                                     scores['test_f1'].mean(),
                                     scores['test_roc_auc'].mean()]

            # on affiche les résultats, en vers les meilleurs score
            table = models_scores_standard_scaler

            df_report = pd.DataFrame(table).T
            performance = RF_clf.score(X_test, y_test)
            st.write('La performance du modele est de : ', round(performance,4))
            st.dataframe(df_report)
            

    
    # importation des données
    df = pd.read_csv('assets/bank.csv')
    
    st.header("Créer un modèle Random Forest performant")
    
    st.markdown('<div style="text-align: justify">Nous avions fait un pre-processing simple pour faire tourner des premiers modèles. Nous allons maintenant chercher à optimiser cette phase afin de mettre le modèle dans les meilleures dispositions lors de son entraînement.</div><br>', unsafe_allow_html=True)

    #st.text("choix par defaut : toutes les données")
    
    tab1, tab2 = st.tabs(["Amélioration du pre-processing", "Optimisation des hyperparamètres"])
    
    with tab1:
#if st.checkbox("Performance pre-processing"): 
        #st.header('Récap de performance')
        taba,tabd,tabb,tabc,tabe = st.tabs(['Selection des features','Feature engeneering', 'Gestion des outliers','Encodage',"Les résultats du pré-processing"])
        with taba :
            #st.text('Selection des features')
            st.header('Variance Threshold')
            st.markdown('<div style="text-align: justify">VarianceThreshold permet de conserver uniquement les features qui ont une variance minimum. Comme nous avons pu le voir dans la première partie, la feature default varie très peu.</div><br>', unsafe_allow_html=True)
            st.header('SelectKBest')
            st.markdown('<div style="text-align: justify">SelectKBest permet de déterminer les meilleures features en fonction de la target. Certaines features comme job_admin. job_self-employed ont tendance à améliorer le résultat du modèle lorsqu’elles sont supprimées. Les valeurs unknown sont aussi problématiques, job_unknown et education_unknown font partie des pires features pour SelectKBest.</div><br>', unsafe_allow_html=True)
            st.header('Traitement des unknown')
            st.image(Image.open("assets/features_unknown.png"), use_column_width=False)
            st.markdown('<div style="text-align: justify">Pour ‘poutcome’, cela concerne tous les clients dont c’est la première campagne. Il n’y a donc pas de résultat précédent. Nous pouvons donc laisser cette donnée en l’état.</div><br>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify">Pour contact, nous n’avons aucune piste. Nous constatons juste que la grande majorité des cas se trouve en mai ou en juin. Nous sommes obligé de conserver cette information car on avait constaté un très fort écart de souscription pour cette valeur inconnue.</div><br>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify">Pour ‘job’ et ‘education’, nous avons très peu de données manquantes. Nous avons décidé de remplacer la valeur unknown, pour compléter éducation, nous nous sommes basé sur le job du client et inversement.</div><br>', unsafe_allow_html=True)
            #st.markdown('<div style="text-align: justify"></div><br>', unsafe_allow_html=True)
            st.header('Selection manuelle')
            st.markdown('<div style="text-align: justify">On a tenté de supprimer chacune des colonnes une à une pour constater un impact positif, négatif ou neutre.</div><br>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify"> A la fin de ce premier process nous faisont le choix de supprimer ‘duration’ et ‘campaign’ car nous ne pouvons pas avoir cette information avant l’appel. Puis nous supprimons ‘previous’, ‘job’ et ‘default’, car cela améliore la performance du Random Forest.</div><br>', unsafe_allow_html=True)
            st.image(Image.open("assets/RF_keep_features.png"), use_column_width=False, caption='Liste des champs conservés')       
        with tabb :
            #st.text('Gestion des outliers')
            st.markdown('<div style="text-align: justify">Le Random Forest est très résistant aux outliers, mais nous avons tout de même réussi à améliorer le modèle en imposant une limite sur la balance. Nous avions vu que plus la balance augmente, plus on a de chance d’avoir deposit avec yes, mais aussi, qu’il y avait une espèce de seuil, qui fait qu’à partir d’un moment, avoir plus d’argent n’augmente pas les chances de souscrire. </div><br>', unsafe_allow_html=True)
        with tabc :
            st.markdown('<div style="text-align: justify">Pour l’encodage, nous reprenons simplement le code de la première phase de pre-processing. En gérant correctement les types de colonnes, nos modifications précédentes ont été prises en compte automatiquement.</div><br>', unsafe_allow_html=True)
        with tabd :
            #st.text('Feature engeneering')
            st.markdown('<div style="text-align: justify">Le feature engineering consiste à modifier ou créer de nouvelles variables à partir de l’existant. Nous avons testé énormément de possibilités. On va lister quelques tentatives.</div><br>', unsafe_allow_html=True)
            tabz, taby = st.tabs(["Ce qui a fonctionné", "Exemples de ce qui n'a pas fonctionné"])
            with tabz :
                st.markdown('- Conservation de "poutcome = success" seulement')
                st.markdown('- Classification education en 1-2-3')
                st.markdown('- Classification statut marital en 1-2-3')
            with taby :
                st.markdown('- Définir des classes : pour balance, age et day')
                st.markdown('- Regrouper : les actifs et inactifs, les grandes campagnes d’appel, pdays (en mois, en trimestre…), le contact telephone et cellular ensemble, les prêts housing et loan, les mois directement de 1 à 12 comme pour day')
        with tabe :
            #st.header("Les résultats du pré-processing")
            st.markdown('<div style="text-align: justify">Résultat : Nous avions 46 colonnes après le premier pré-processing. Nous en avons maintenant 24.</div><br>', unsafe_allow_html=True)
            st.image(Image.open("assets/columns_result_RF.png"), use_column_width=False)
            st.header("PCA après la seconde phase de pré-processing")
            st.image(Image.open("assets/PCA_result_RF.png"), use_column_width=False)
            st.header("Les résultats de l’optimisation du choix des features")
            st.markdown('<div style="text-align: justify">Nous avons toujours le même modèle basique de Random Forest. Juste en améliorant la phase de pré-processing, on a amélioré toutes les métriques. On gagne 1,12 % sur le ROC AUC qui est la métrique principale pour le moment.</div><br>', unsafe_allow_html=True)
            st.image(Image.open("assets/resultats_prepro_RF.png"), use_column_width=False)  
    
    with tab2:
        tabf, tabg = st.tabs(["Définition des hyperparametres", "Selection des hyperparametres"])
        with tabf :
            st.markdown('- `Criterion`: Dans un Decision Tree, lors de la séparation des données en deux branches, le modèle cherche à maximiser la séparation entre les deux classes recherchées. Criterion est la méthode de calcul appliquée pour choisir cette séparation.')
            st.markdown('- `Max_depth` : représente la profondeur maximale d’un arbre. Plus la profondeur est élevée et plus le modèle fera de l’overfitting. Par défaut, la valeur est illimitée, ce qui explique pourquoi nous avons de l’overfitting sur le graphique de la learning curve.')
            st.markdown('- `Max_features` : permet d’apporter de l’aléatoire au modèle. Pour chaque embranchement, on va prendre au hasard le nombre de features max défini préalablement. Parmi elles, le modèle va choisir la feature la plus utile. Cet hyperparamètre force le modèle à tester des features moins utiles. Plus max_features est élevé, plus le modèle fera de l’overfitting car on réduira la part d’aléatoire.')
            st.markdown('- `Min_samples_split` : la limite qui permet de découper un nœud en deux nouvelles branches. Plus on augmente la valeur, plus on réduit l’overfitting. En revanche, si on augmente trop cette valeur, on risque de fortement baisser les performances du modèle.')
            st.markdown('- `Min_samples_leaf` : fonctionne un peu comme l’hyperparamètres précédent. La différence réside sur le fait que la valeur vérifiée n’est plus celle de départ mais celle d’arrivée. Si min_samples_leaf était égal à 3, la case orange en bas à droite n’aurait pas pu être créée, car il n’y a que 2 données dedans. Augmenter min_samples_leaf réduit l’overffiting mais peut réduire la performance du modèle.')
            st.markdown('- `Class_weight` : permet d’équilibrer le poids entre les classes 1 et 0 de deposit. Notre target n’est pas parfaitement équilibré avec 52,5/47,5 en faveur du no. Class_weight permet d’augmenter le poids d’une des deux classes, afin qu’il soit plus punitif pour le modèle de se tromper sur une classe plutôt que l’autre. Il y a un critère ‘balanced’, qui permet d’avoir un ratio inversement proportionnel à notre distribution, ce qui permet de simuler une distribution équilibrée.')
            st.markdown('- `N_estimators` : le nombre d’arbres utilisés dans notre forêt aléatoire. Plus le chiffre est élevé et plus la performance augmente. Cependant, on atteint à partir un plateau un certain seuil. Ce n’est donc pas utile d’avoir une valeur trop élevée, d’autant que plus celle-ci est élevée, plus le modèle va prendre du temps à entraîner.')
            
        with tabg :
            #st.text('Selection des hyperparametres')
            st.markdown('<div style="text-align: justify">La fonction GridSearchCV, inclut directement une cross-validation qui est essentielle pour s’assurer de la performance du modèle sur des nouvelles données. Cela permet également de rentrer une liste complète d’hyperparamètres, afin que le modèle teste toutes les possibilités. C’est donc très efficace pour trouver la meilleure combinaison d’hyperparamètre. Le problème étant, que plus il y a de possibilité à tester, plus le résultat sera long à obtenir.</div><br>', unsafe_allow_html=True)
            st.header("Les deux meilleurs résultats de l’optimisation des hyperparamètres")
            st.markdown('<div style="text-align: justify">Ce qui nous intéresse pour juger le modèle est le validation score. Tout en ayant un train score le moins élévé possible (overfitting).</div><br>', unsafe_allow_html=True)
            st.image(Image.open("assets/Best_GSCV_RF_1.png"), use_column_width=False)
            st.image(Image.open("assets/Best_GSCV_RF_2.png"), use_column_width=False)
            st.header("Les résultats finaux de l’optimisation des hyperparamètres")
            st.markdown('<div style="text-align: justify">Nous avons encore amélioré le score du modèle. La seule métrique qui est moins bonne est le recall et ce n’est pas vraiment important. Cela veut juste dire que le précédent modèle préconise d’appeler plus de clients quitte à se tromper plus souvent.</div><br>', unsafe_allow_html=True)
            st.image(Image.open("assets/resultats_RF_gds.png"), use_column_width=False)
    