import streamlit as st


title = "Prédire le succès d'une campagne de marketing d'une banque"
sidebar_name = "Introduction"


def run():
    st.image("assets/image_intro.png")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
L’analyse des données marketing est une problématique très classique des sciences des données appliquées dans les entreprises de service. Pour ce jeu de données, nous avons des données personnelles sur des clients d’une banque qui ont été “télémarketés” pour souscrire à un produit que l’on appelle un "dépôt à terme”. Lorsqu’un client souscrit à ce produit, il place une quantité d’argent dans un compte spécifique et ne peut plus toucher à ses fonds avant l’expiration du terme. En échange, le client reçoit des intérêts de la part de la banque à la fin du terme.

Pour ce projet, nous avons d’abord effectué une analyse visuelle et statistique des facteurs pouvant expliquer le lien entre les données personnelles du client (âge, statut marital, quantité d’argent placée dans la banque...) et la variable cible : “Est-ce que le client a souscrit au dépôt à terme ?”.

Une fois l’analyse visuelle terminée, nous avons utilisé des techniques de machine learning pour déterminer à l’avance si un client va oui ou non souscrire au produit. Une fois cette prédiction réalisée, nous avons utilisé les techniques d’interprétabilité des modèles de machine learning pour expliquer à l’échelle d’un individu pourquoi il est plus susceptible de souscrire au produit ou non.

Le jeu de données est téléchargeable au lien suivant: https://www.kaggle.com/janiobachmann/bank-marketing-dataset
        """
    )