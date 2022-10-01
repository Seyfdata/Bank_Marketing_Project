from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import introduction, analyse_exploratoire, pretraitement, modele_simple, random_forest, results, demo


st.set_page_config(
    page_title=config.TITLE,
    page_icon="assets/Bank_logo.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (introduction.sidebar_name, introduction),
        (analyse_exploratoire.sidebar_name, analyse_exploratoire),
        (pretraitement.sidebar_name, pretraitement),
        (modele_simple.sidebar_name, modele_simple),
        (random_forest.sidebar_name, random_forest),
        (results.sidebar_name, results),
        (demo.sidebar_name, demo),
    ]
)


def run():

    st.sidebar.image(
        "assets/logo_datascientest2.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Membres de l'équipe:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
