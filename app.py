# -*- coding: utf-8 -*-
"""
============================================================================
 Application Streamlit — Pilotage du Risque de Crédit (Al Barid Bank, PFA)
 Auteur : Walid BEN ABID — Filière IFA, FST Errachidia
----------------------------------------------------------------------------
 Cette application automatise l'intégralité de la chaîne quantitative du
 mémoire :
   1. Génération / chargement du portefeuille (8 000 contrats simulés)
   2. Analyse exploratoire interactive (chapitre 4)
   3. Tests statistiques & régression logistique (chapitre 5)
   4. Notation interne : score, rating, PD, LGD, EAD (chapitre 7)
   5. EL Bâle III, ECL IFRS 9, capital économique IRB (z = 3,090 — 99,9 %)
   6. Stress-test interactif (chapitre 7.10)
   7. Tableau de bord de pilotage à seuils (chapitre 8)
----------------------------------------------------------------------------
 CORRECTIONS INTÉGRÉES par rapport au mémoire initial :
   • Capital IRB au niveau réglementaire 99,9 % (z = 3,090) — comparateur
     pédagogique 99 % vs 99,9 % inclus ;
   • Cohérence figure/tableau ECL (une seule source de calcul) ;
   • Matrice de transition normalisée (chaque ligne somme à 100 %).
============================================================================
Lancement :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as st_stats
import statsmodels.formula.api as smf

# ============================================================================
# 0. CONFIGURATION GÉNÉRALE & CHARTE GRAPHIQUE
# ============================================================================

st.set_page_config(
    page_title="Risque de Crédit — Al Barid Bank (PFA)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette institutionnelle (bleu profond / or postal / états de risque)
BLEU   = "#0F3D68"   # bleu institutionnel principal
BLEU2  = "#1F6FB2"   # bleu secondaire
OR     = "#C9A227"   # accent or (Barid)
VERT   = "#1E7F4F"
ORANGE = "#D97B29"
ROUGE  = "#B03A2E"
GRIS   = "#5D6D7E"
SEQ    = [BLEU, BLEU2, VERT, ORANGE, ROUGE, OR, GRIS]

st.markdown(
    f"""
    <style>
      .main .block-container {{ padding-top: 1.2rem; }}
      h1, h2, h3 {{ color: {BLEU}; font-family: Georgia, 'Times New Roman', serif; }}
      section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, {BLEU} 0%, #0A2A49 100%);
      }}
      section[data-testid="stSidebar"] * {{ color: #F2F4F7 !important; }}
      div[data-testid="stMetric"] {{
          background: #F5F8FB; border-left: 5px solid {BLEU};
          padding: 12px 14px; border-radius: 6px;
      }}
      .badge-ok    {{ background:{VERT};   color:white; padding:3px 10px; border-radius:12px; font-weight:600; }}
      .badge-warn  {{ background:{ORANGE}; color:white; padding:3px 10px; border-radius:12px; font-weight:600; }}
      .badge-alert {{ background:{ROUGE};  color:white; padding:3px 10px; border-radius:12px; font-weight:600; }}
      .note {{
          background:#FBF6E7; border-left:5px solid {OR};
          padding:10px 14px; border-radius:6px; font-size:0.92rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

TYPE_ORDER = ["Immobilier", "Consommation", "Avance sur salaire",
              "CMLT", "Facilite de caisse", "Intelak"]
SEGMENTS_PART = ["Particulier - Fonctionnaire", "Particulier - Salarie prive",
                 "Particulier - Retraite"]
SEGMENTS_PRO  = ["Professionnel - TPE", "Auto-entrepreneur"]
REGIONS = ["Casablanca-Settat", "Rabat-Sale-Kenitra", "Marrakech-Safi",
           "Fes-Meknes", "Beni Mellal-Khenifra", "Tanger-Tetouan-Al Hoceima",
           "Souss-Massa", "Oriental", "Draa-Tafilalet", "Guelmim-Oued Noun"]
REGION_W = [0.20, 0.16, 0.12, 0.11, 0.09, 0.09, 0.08, 0.07, 0.05, 0.03]

RATING_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]
PD_CIBLE = {"A": 0.005, "B": 0.012, "C": 0.025, "D": 0.050,
            "E": 0.085, "F": 0.140, "G": 0.220, "H": 0.330}

# ============================================================================
# 1. GÉNÉRATION DU PORTEFEUILLE SIMULÉ (chapitres 3-4 du mémoire)
# ============================================================================

@st.cache_data(show_spinner="Génération du portefeuille simulé (n = 8 000)…")
def generer_portefeuille(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Reproduit le jeu de données simulé du mémoire (seed 42),
    calibré sur les ordres de grandeur publics du secteur bancaire marocain."""
    rng = np.random.default_rng(seed)

    # --- Type de crédit -----------------------------------------------------
    proba_type = [0.22, 0.34, 0.18, 0.10, 0.09, 0.07]
    type_credit = rng.choice(TYPE_ORDER, size=n, p=proba_type)

    # --- Segment de clientèle (conditionné au produit) ----------------------
    segment = np.empty(n, dtype=object)
    pro_mask = np.isin(type_credit, ["CMLT", "Facilite de caisse", "Intelak"])
    segment[pro_mask]  = rng.choice(SEGMENTS_PRO,  size=pro_mask.sum(),
                                    p=[0.55, 0.45])
    segment[~pro_mask] = rng.choice(SEGMENTS_PART, size=(~pro_mask).sum(),
                                    p=[0.42, 0.40, 0.18])

    # --- Géographie & agence -------------------------------------------------
    region = rng.choice(REGIONS, size=n, p=REGION_W)
    prefixe = {r: r.split("-")[0][:3].upper() for r in REGIONS}
    agence = np.array([f"AG-{prefixe[r]}-{rng.integers(1, 41):03d}"
                       for r in region])

    # --- Caractéristiques du contrat -----------------------------------------
    params_montant = {  # (mu, sigma) log-normaux — médianes réalistes en DH
        "Immobilier": (11.7, 0.55), "Consommation": (10.3, 0.60),
        "Avance sur salaire": (8.4, 0.45), "CMLT": (11.9, 0.65),
        "Facilite de caisse": (10.8, 0.55), "Intelak": (10.9, 0.50)}
    duree_map = {"Immobilier": (120, 300), "Consommation": (12, 84),
                 "Avance sur salaire": (1, 12), "CMLT": (36, 120),
                 "Facilite de caisse": (6, 24), "Intelak": (24, 84)}
    taux_map = {"Immobilier": (4.4, 0.6), "Consommation": (7.2, 1.0),
                "Avance sur salaire": (8.0, 0.8), "CMLT": (6.6, 0.9),
                "Facilite de caisse": (8.8, 1.0), "Intelak": (5.8, 0.7)}

    montant = np.array([rng.lognormal(*params_montant[t]) for t in type_credit])
    duree = np.array([rng.integers(*duree_map[t]) for t in type_credit])
    taux = np.array([max(2.0, rng.normal(*taux_map[t])) for t in type_credit])
    anciennete = np.minimum(rng.gamma(2.2, 55, n).astype(int), 420)

    date_octroi = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 4 * 365, n), unit="D")

    # --- Garantie (conditionnée au produit) ----------------------------------
    def tirer_garantie(t):
        if t == "Immobilier":
            return rng.choice(["Hypotheque", "Aucune"], p=[0.85, 0.15])
        if t == "CMLT":
            return rng.choice(["Nantissement", "Hypotheque",
                               "Garantie Tamwilcom", "Aucune"],
                              p=[0.35, 0.20, 0.25, 0.20])
        if t == "Intelak":
            return rng.choice(["Garantie Tamwilcom", "Aucune"], p=[0.80, 0.20])
        if t == "Facilite de caisse":
            return rng.choice(["Nantissement", "Aucune"], p=[0.30, 0.70])
        return rng.choice(["Aucune", "Nantissement"], p=[0.88, 0.12])
    garantie = np.array([tirer_garantie(t) for t in type_credit])

    secteurs = ["Commerce", "Services", "BTP", "Transport",
                "Artisanat", "Agriculture"]
    secteur = np.where(pro_mask,
                       rng.choice(secteurs, size=n,
                                  p=[0.30, 0.22, 0.16, 0.12, 0.11, 0.09]),
                       "N/A")

    # --- Capital restant dû ---------------------------------------------------
    avancement = rng.uniform(0.05, 0.95, n)
    capital_restant = montant * (1 - avancement)

    # --- Probabilité d'incident : ancrage empirique (chap. 4-5) --------------
    risque_base = {"Immobilier": 0.012, "Consommation": 0.040,
                   "Avance sur salaire": 0.014, "CMLT": 0.048,
                   "Facilite de caisse": 0.078, "Intelak": 0.028}
    fac_segment = {"Particulier - Fonctionnaire": 0.62,
                   "Particulier - Salarie prive": 1.00,
                   "Particulier - Retraite": 1.25,
                   "Professionnel - TPE": 1.30, "Auto-entrepreneur": 1.55}
    p_inc = np.array([risque_base[t] * fac_segment[s]
                      for t, s in zip(type_credit, segment)])
    p_inc *= np.where(anciennete < 12, 1.85, 1.0)          # OR ≈ 1,89
    p_inc *= np.where(garantie == "Aucune", 1.35, 0.85)    # OR ≈ 1,70
    p_inc = np.clip(p_inc, 0.002, 0.60)

    incident = rng.random(n) < p_inc
    jours = np.zeros(n, dtype=int)
    jours[incident] = rng.choice(
        [rng.integers(90, 180), 0], size=incident.sum()) * 0  # placeholder
    # Répartition des retards parmi les incidents : pré-douteux / douteux / compromis
    tirage = rng.random(incident.sum())
    j_inc = np.where(tirage < 0.45, rng.integers(90, 180, incident.sum()),
             np.where(tirage < 0.82, rng.integers(180, 360, incident.sum()),
                                     rng.integers(360, 720, incident.sum())))
    jours[incident] = j_inc
    retard_leger = (~incident) & (rng.random(n) < 0.10)
    jours[retard_leger] = rng.integers(1, 89, retard_leger.sum())

    def classer_bam(j):
        if j < 90:   return "Sain"
        if j < 180:  return "Pre-douteux"
        if j < 360:  return "Douteux"
        return "Compromis"

    df = pd.DataFrame({
        "id_credit": [f"CR{100000 + i}" for i in range(n)],
        "type_credit": type_credit, "segment_client": segment,
        "region": region, "agence": agence,
        "date_octroi": date_octroi,
        "montant_accorde": montant.round(0),
        "duree_mois": duree, "taux_interet": taux.round(2),
        "anciennete_client_mois": anciennete,
        "garantie": garantie, "secteur_activite": secteur,
        "capital_restant_du": capital_restant.round(0),
        "jours_retard_max": jours,
    })
    df["statut_bam"] = df["jours_retard_max"].apply(classer_bam)
    df["cds"] = (df["jours_retard_max"] >= 90).astype(int)
    df["nb_incidents"] = np.where(
        df["cds"] == 1, rng.integers(1, 5, n),
        np.where(df["jours_retard_max"] > 0, 1, 0))
    return df

# ============================================================================
# 2. NOTATION INTERNE : SCORE → RATING → PD → LGD → EAD (chapitre 7)
# ============================================================================

@st.cache_data(show_spinner="Calcul du dispositif de notation interne…")
def calculer_notation(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Score 300-900, rating A-H, PD 1 an, LGD (loi Bêta), EAD (CCF),
    exactement selon la construction du chapitre 7 du mémoire."""
    rng = np.random.default_rng(seed + 1)
    d = df.copy()
    n = len(d)

    # 1. Score de base par segment (ancre empirique chap. 5)
    base = {"Particulier - Fonctionnaire": 690,
            "Particulier - Salarie prive": 635,
            "Particulier - Retraite": 655,
            "Professionnel - TPE": 575, "Auto-entrepreneur": 555}
    score = d["segment_client"].map(base).astype(float)

    # 2. Bonus ancienneté (plafonné à 15 ans)
    score += 0.50 * np.minimum(d["anciennete_client_mois"], 180)

    # 3. Bonus garantie (cohérent LGD & OR régression)
    score += d["garantie"].map({"Hypotheque": 40, "Garantie Tamwilcom": 25,
                                "Nantissement": 12, "Aucune": 0}).astype(float)

    # 4. Ajustement type de crédit (cohérent CDS observé)
    score += d["type_credit"].map({"Immobilier": 18, "Consommation": 0,
                                   "Avance sur salaire": 12, "CMLT": -20,
                                   "Facilite de caisse": -30,
                                   "Intelak": -8}).astype(float)

    # 5. Malus taille du crédit (OR log-montant = 1,30)
    z = (np.log(d["montant_accorde"]) - np.log(d["montant_accorde"]).mean()) \
        / np.log(d["montant_accorde"]).std()
    score += -7.0 * np.clip(z, 0, None)

    # 6. Bruit idiosyncratique
    score += rng.normal(0, 45, n)
    d["score_credit"] = score.clip(300, 900).round(0).astype(int)

    # 7. Rating en 8 classes
    d["rating_interne"] = pd.cut(
        d["score_credit"],
        bins=[300, 500, 560, 610, 650, 690, 730, 780, 900],
        labels=["H", "G", "F", "E", "D", "C", "B", "A"],
        include_lowest=True).astype(str)

    # 8. PD 1 an par table d'ancrage + dispersion log-normale
    d["pd_1an"] = (d["rating_interne"].map(PD_CIBLE).astype(float)
                   * rng.lognormal(0, 0.20, n)).clip(0.002, 0.55)

    # 9. LGD par loi Bêta calée sur la garantie
    lgd_moy = d["garantie"].map({"Hypotheque": 0.26, "Garantie Tamwilcom": 0.37,
                                 "Nantissement": 0.48, "Aucune": 0.61}).astype(float)
    kappa = 14
    d["lgd"] = rng.beta(lgd_moy * kappa, (1 - lgd_moy) * kappa).clip(0.05, 0.90)

    # 10. EAD avec CCF (revolving)
    ccf = d["type_credit"].map({"Facilite de caisse": 0.65,
                                "Avance sur salaire": 0.50}).fillna(0.0)
    d["ead"] = d["capital_restant_du"] + ccf * (
        d["montant_accorde"] - d["capital_restant_du"]).clip(lower=0)

    # 11. Stage IFRS 9 (affectation vectorisée — ~50× plus rapide qu'un apply)
    cond3 = d["statut_bam"].isin(["Douteux", "Compromis"])
    cond2 = ((d["statut_bam"] == "Pre-douteux")
             | d["rating_interne"].isin(["F", "G", "H"])
             | (d["nb_incidents"] >= 2))
    d["stage_ifrs9"] = np.select([cond3, cond2], [3, 2], default=1)
    return d


@st.cache_data(show_spinner="Estimation du modèle logit (une seule fois)…")
def estimer_logit(df_in: pd.DataFrame):
    """Régression logistique du chapitre 5, mise en cache : l'estimation
    n'a lieu qu'une fois par jeu de données, puis l'affichage est instantané."""
    data = df_in.copy()
    data["log_montant"] = np.log(data["montant_accorde"])
    data["garantie_aucune"] = (data["garantie"] == "Aucune").astype(int)
    data["fonctionnaire"] = (
        data["segment_client"] == "Particulier - Fonctionnaire").astype(int)
    data["pro_autoentr"] = data["segment_client"].isin(SEGMENTS_PRO).astype(int)
    data["anciennete_courte"] = (data["anciennete_client_mois"] < 12).astype(int)

    model = smf.logit(
        "cds ~ log_montant + duree_mois + taux_interet + anciennete_courte + "
        "garantie_aucune + fonctionnaire + pro_autoentr",
        data=data).fit(disp=0)

    res = pd.DataFrame({
        "Coefficient": model.params.round(3),
        "Odds ratio": np.exp(model.params).round(3),
        "IC 95 % bas": np.exp(model.conf_int()[0]).round(3),
        "IC 95 % haut": np.exp(model.conf_int()[1]).round(3),
        "p-value": model.pvalues.round(4)})

    y, s = data["cds"].values, model.predict(data).values
    ordre = np.argsort(-s)
    y_ord = y[ordre]
    tpr = np.cumsum(y_ord) / y_ord.sum()
    fpr = np.cumsum(1 - y_ord) / (1 - y_ord).sum()
    auc = float(_trapz(tpr, fpr))
    return res, fpr, tpr, auc, float(model.prsquared), float(model.llr_pvalue)


@st.cache_data(show_spinner="Construction de la courbe de sensibilité…")
def courbe_stress(df_in: pd.DataFrame, pas: int = 20) -> pd.DataFrame:
    """Courbe EL/ECL/Capital pour des chocs de 0 à +200 %, calculée UNE fois
    puis mise en cache : le slider devient instantané."""
    ead = calculer_pertes(df_in, 0.0, 0.999)["ead"].sum()
    lignes = []
    for c in range(0, 201, pas):
        r = calculer_pertes(df_in, c / 100, 0.999)
        lignes.append({"choc": c,
                       "EL/EAD": r["el_1an"].sum() / ead * 100,
                       "ECL/EAD": r["ecl"].sum() / ead * 100,
                       "Capital/EAD": r["capital_eco"].sum() / ead * 100})
    return pd.DataFrame(lignes)


# Compatibilité NumPy 1.x / 2.x
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def pd_lifetime(pd1an: np.ndarray, duree_res_mois: np.ndarray) -> np.ndarray:
    """PD lifetime par composition de survies annuelles, seasoning 15 %/an,
    horizon effectif plafonné à 5 ans (simplification assumée du mémoire)."""
    t_eff = np.minimum(duree_res_mois / 12.0, 5.0)
    pd_lt = np.zeros_like(pd1an)
    survie = np.ones_like(pd1an)
    for k in range(5):
        dt = np.clip(t_eff - k, 0, 1)                # fraction d'année couverte
        pd_marg = np.clip(pd1an * (0.85 ** k), 0, 0.99)
        survie *= (1 - pd_marg) ** dt
    pd_lt = 1 - survie
    return np.clip(pd_lt, pd1an, 0.995)              # lifetime ≥ PD 1 an


@st.cache_data(show_spinner=False)
def calculer_pertes(d: pd.DataFrame, choc_pd: float = 0.0,
                    confiance: float = 0.999) -> pd.DataFrame:
    """EL Bâle III, ECL IFRS 9 par stage, capital économique IRB retail.
    ⚡ Mis en cache : chaque triplet (données, choc, confiance) n'est
    calculé qu'une fois par session.
    `choc_pd` : choc multiplicatif sur les PD (stress-test), ex. 0.80 = +80 %.
    `confiance` : niveau IRB (0.999 réglementaire → z = 3,090)."""
    d = d.copy()
    d["pd_stress"] = (d["pd_1an"] * (1 + choc_pd)).clip(0.002, 0.99)

    duree_res = (d["duree_mois"] * 0.55).clip(lower=6)   # durée résiduelle approx.
    d["pd_lifetime"] = pd_lifetime(d["pd_stress"].values, duree_res.values)

    # --- EL Bâle III (horizon 1 an) ------------------------------------------
    d["el_1an"] = d["pd_stress"] * d["lgd"] * d["ead"]

    # --- ECL IFRS 9 par stage --------------------------------------------------
    d["ecl"] = np.select(
        [d["stage_ifrs9"] == 1, d["stage_ifrs9"] == 2, d["stage_ifrs9"] == 3],
        [d["el_1an"],
         d["pd_lifetime"] * d["lgd"] * d["ead"],
         d["lgd"] * d["ead"]])                        # Stage 3 : PD = 1

    # --- Capital économique IRB retail (rho = 0,15) ---------------------------
    rho = 0.15
    z_conf = st_stats.norm.ppf(confiance)             # 3,090 à 99,9 %
    pdv = d["pd_stress"].clip(0.001, 0.999)
    pd_cond = st_stats.norm.cdf(
        (st_stats.norm.ppf(pdv) + np.sqrt(rho) * z_conf) / np.sqrt(1 - rho))
    d["capital_eco"] = d["lgd"] * d["ead"] * (pd_cond - pdv)
    return d

# ============================================================================
# 3. OUTILS D'AFFICHAGE
# ============================================================================

def fig_style(fig, h=430):
    fig.update_layout(
        height=h, template="plotly_white",
        font=dict(family="Georgia, serif", size=13, color="#22313F"),
        title_font=dict(size=17, color=BLEU),
        margin=dict(l=40, r=25, t=60, b=40),
        colorway=SEQ, legend=dict(orientation="h", y=-0.18))
    return fig


def badge(valeur, seuil_orange, seuil_rouge, sens=">"):
    """Retourne un badge HTML selon les seuils du tableau 8.1."""
    if sens == ">":
        if valeur > seuil_rouge:   return '<span class="badge-alert">ROUGE</span>'
        if valeur > seuil_orange:  return '<span class="badge-warn">ORANGE</span>'
    else:
        if valeur < seuil_rouge:   return '<span class="badge-alert">ROUGE</span>'
        if valeur < seuil_orange:  return '<span class="badge-warn">ORANGE</span>'
    return '<span class="badge-ok">VERT</span>'

# ============================================================================
# 4. CHARGEMENT DES DONNÉES (simulées ou CSV réel)
# ============================================================================

st.sidebar.markdown("## 🏦 Risque de Crédit\n#### Al Barid Bank — PFA IFA")
st.sidebar.markdown("---")

source = st.sidebar.radio("Source des données",
                          ["Portefeuille simulé",
                           "Importer un CSV réel"])
taille = st.sidebar.select_slider(
    "Taille de l'échantillon simulé", options=[2000, 4000, 8000], value=8000,
    help="8 000 = configuration du mémoire · 2 000/4 000 = affichage plus "
         "rapide sur machine ou connexion lente (mêmes conclusions).")

if source == "Importer un CSV réel":
    up = st.sidebar.file_uploader("CSV au format du dictionnaire (chap. 3)",
                                  type="csv")
    if up is not None:
        df_base = pd.read_csv(up, parse_dates=["date_octroi"])
        df_base["cds"] = (df_base["jours_retard_max"] >= 90).astype(int)
        if "statut_bam" not in df_base.columns:
            df_base["statut_bam"] = pd.cut(
                df_base["jours_retard_max"], [-1, 89, 179, 359, 10**6],
                labels=["Sain", "Pre-douteux", "Douteux", "Compromis"]).astype(str)
        if "nb_incidents" not in df_base.columns:
            df_base["nb_incidents"] = df_base["cds"]
    else:
        st.sidebar.info("En attente d'un fichier — portefeuille simulé utilisé.")
        df_base = generer_portefeuille(n=taille)
else:
    df_base = generer_portefeuille(n=taille)

df = calculer_notation(df_base)

page = st.sidebar.radio("Navigation", [
    "🏠 Accueil & synthèse",
    "🗂️ Données du portefeuille",
    "🔍 Analyse exploratoire",
    "📐 Tests statistiques & logit",
    "⭐ Notation interne & simulateur",
    "💰 EL · ECL IFRS 9 · Capital IRB",
    "⚡ Stress-test interactif",
    "🎛️ Tableau de bord de pilotage",
])
st.sidebar.markdown("---")
st.sidebar.caption("Données **simulées** à des fins pédagogiques — "
                   "aucune donnée réelle d'Al Barid Bank.\n\n"
                   "Capital IRB au niveau **99,9 %** (z = 3,090), "
                   "conformément à Bâle III.")

# Calcul de base (scénario sans choc, 99,9 %)
base = calculer_pertes(df, choc_pd=0.0, confiance=0.999)
EAD_T = base["ead"].sum()
EL_T, ECL_T, CAP_T = base["el_1an"].sum(), base["ecl"].sum(), base["capital_eco"].sum()

# ============================================================================
# PAGE — ACCUEIL
# ============================================================================
if page.startswith("🏠"):
    st.title("Pilotage du risque de crédit — Al Barid Bank")
    st.markdown(
        "Application décisionnelle accompagnant le **mémoire de PFA** : elle "
        "rejoue l'intégralité de la chaîne quantitative — de l'exploration du "
        "portefeuille jusqu'au capital économique — sur données simulées ou "
        "sur un extrait réel importé au même format.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Contrats", f"{len(base):,}".replace(",", " "))
    c2.metric("Taux de CDS (nombre)", f"{base['cds'].mean()*100:.2f} %")
    c3.metric("EL / EAD (1 an)", f"{EL_T/EAD_T*100:.2f} %")
    c4.metric("ECL / EAD (IFRS 9)", f"{ECL_T/EAD_T*100:.2f} %")
    c5.metric("Capital IRB / EAD (99,9 %)", f"{CAP_T/EAD_T*100:.2f} %")

    st.markdown(
        f"""<div class="note">📌 <b>Corrections intégrées par rapport au
        mémoire :</b> capital économique au niveau réglementaire
        <b>99,9 %</b> (z = 3,090) au lieu de 99 % ; cohérence unique
        figure/tableau pour l'ECL ; matrice de transition normalisée à 100 %.
        Le comparateur 99 % / 99,9 % de la page <i>Capital</i> visualise
        l'impact de cette correction.</div>""", unsafe_allow_html=True)

    st.subheader("Architecture de l'application")
    st.markdown("""
| Page | Chapitre du mémoire | Contenu |
|---|---|---|
| 🗂️ Données | Chap. 3 | Dictionnaire, contrôle qualité, export CSV |
| 🔍 Exploration | Chap. 4 | Les 10 lectures graphiques, filtrables |
| 📐 Tests & logit | Chap. 5 | χ², proportions, Mann-Whitney, régression logistique, ROC |
| ⭐ Notation | Chap. 7.2-7.4 | Score, rating A-H, PD, LGD, EAD + **simulateur client** |
| 💰 Pertes & capital | Chap. 7.5-7.7 | EL Bâle III, ECL IFRS 9 par stage, capital IRB 99,9 % |
| ⚡ Stress-test | Chap. 7.10 | Choc de PD paramétrable en continu (0 → +200 %) |
| 🎛️ Tableau de bord | Chap. 8 | 8 KPI à seuils + classement des agences |
""")

# ============================================================================
# PAGE — DONNÉES
# ============================================================================
elif page.startswith("🗂️"):
    st.title("Données du portefeuille")
    st.markdown("Jeu de travail conforme au **dictionnaire des données** "
                "(tableau 3.2 du mémoire).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Contrats", f"{len(df):,}".replace(",", " "))
    c2.metric("Encours total",
              f"{df['capital_restant_du'].sum()/1e6:,.1f} M DH".replace(",", " "))
    c3.metric("Période d'octroi",
              f"{df['date_octroi'].min():%Y-%m} → {df['date_octroi'].max():%Y-%m}")

    st.dataframe(df.head(200), use_container_width=True, height=380)

    st.subheader("Contrôle qualité")
    qc = pd.DataFrame({
        "Contrôle": ["Valeurs manquantes", "Capital restant > montant accordé",
                     "Doublons id_credit", "PD hors [0 ; 1]"],
        "Anomalies": [int(df.isna().sum().sum()),
                      int((df["capital_restant_du"] > df["montant_accorde"]).sum()),
                      int(df["id_credit"].duplicated().sum()),
                      int(((df["pd_1an"] < 0) | (df["pd_1an"] > 1)).sum())]})
    st.table(qc)

    st.download_button("⬇️ Exporter le portefeuille enrichi (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "portefeuille_credits_abb.csv", "text/csv")

# ============================================================================
# PAGE — EXPLORATION (chapitre 4)
# ============================================================================
elif page.startswith("🔍"):
    st.title("Analyse exploratoire du portefeuille")

    colf1, colf2 = st.columns(2)
    f_reg = colf1.multiselect("Filtrer par région", REGIONS, default=REGIONS)
    f_typ = colf2.multiselect("Filtrer par type de crédit", TYPE_ORDER,
                              default=TYPE_ORDER)
    dfe = df[df["region"].isin(f_reg) & df["type_credit"].isin(f_typ)]
    st.caption(f"{len(dfe):,} contrats sélectionnés".replace(",", " "))

    t1, t2, t3, t4 = st.tabs(["Composition", "Qualité du portefeuille",
                              "Facteurs de risque", "Concentration"])

    with t1:
        c1, c2 = st.columns(2)
        rep = dfe.groupby("type_credit")["capital_restant_du"].sum().reindex(TYPE_ORDER)
        fig = px.pie(values=rep.values, names=rep.index, hole=0.35,
                     title="Fig. 4.1 — Encours par type de crédit")
        c1.plotly_chart(fig_style(fig), use_container_width=True)

        reg = dfe.groupby("region")["capital_restant_du"].sum().sort_values() / 1e6
        fig = px.bar(x=reg.values, y=reg.index, orientation="h",
                     labels={"x": "Encours (M DH)", "y": ""},
                     title="Fig. 4.2 — Encours par région")
        c2.plotly_chart(fig_style(fig), use_container_width=True)

        prod = dfe.set_index("date_octroi").resample("ME")["montant_accorde"] \
                  .sum() / 1e6
        fig = px.line(x=prod.index, y=prod.values, markers=True,
                      labels={"x": "", "y": "Production (M DH)"},
                      title="Fig. 4.6 — Production mensuelle de crédits")
        st.plotly_chart(fig_style(fig, 360), use_container_width=True)

    with t2:
        moy = dfe["cds"].mean() * 100
        c1, c2 = st.columns(2)
        q = dfe.groupby("type_credit")["cds"].mean().mul(100) \
               .sort_values(ascending=False)
        fig = px.bar(x=q.index, y=q.values, text=q.round(2),
                     labels={"x": "", "y": "Taux de CDS (%)"},
                     title="Fig. 4.3 — Taux de CDS par type de crédit")
        fig.add_hline(y=moy, line_dash="dash", line_color=VERT,
                      annotation_text=f"Moyenne {moy:.2f} %")
        c1.plotly_chart(fig_style(fig), use_container_width=True)

        qs = dfe.groupby("segment_client")["cds"].mean().mul(100).sort_values()
        fig = px.bar(x=qs.values, y=qs.index, orientation="h", text=qs.round(2),
                     labels={"x": "Taux de CDS (%)", "y": ""},
                     title="Fig. 4.4 — Taux de CDS par segment")
        c2.plotly_chart(fig_style(fig), use_container_width=True)

        fig = px.box(dfe, x="type_credit", y="montant_accorde",
                     category_orders={"type_credit": TYPE_ORDER},
                     labels={"montant_accorde": "Montant accordé (DH)",
                             "type_credit": ""},
                     title="Fig. 4.5 — Distribution des montants accordés")
        fig.update_yaxes(range=[0, dfe["montant_accorde"].quantile(0.97)])
        st.plotly_chart(fig_style(fig), use_container_width=True)

    with t3:
        c1, c2 = st.columns(2)
        g = dfe.groupby("garantie")["cds"].mean().mul(100) \
               .sort_values(ascending=False)
        fig = px.bar(x=g.index, y=g.values, text=g.round(2),
                     labels={"x": "", "y": "Taux de CDS (%)"},
                     title="Fig. 4.7 — Effet de la garantie")
        c1.plotly_chart(fig_style(fig), use_container_width=True)

        dfe2 = dfe.copy()
        dfe2["tranche_anc"] = pd.cut(
            dfe2["anciennete_client_mois"], [-1, 12, 24, 48, 96, 10**4],
            labels=["<1 an", "1-2 ans", "2-4 ans", "4-8 ans", ">8 ans"])
        a = dfe2.groupby("tranche_anc", observed=True)["cds"].mean().mul(100)
        fig = px.bar(x=a.index.astype(str), y=a.values, text=a.round(2),
                     labels={"x": "Ancienneté à l'octroi", "y": "Taux de CDS (%)"},
                     title="Fig. 4.8 — Effet de l'ancienneté bancaire")
        c2.plotly_chart(fig_style(fig), use_container_width=True)

        piv = dfe.pivot_table(index="type_credit", columns="region",
                              values="cds", aggfunc="mean").mul(100)
        fig = px.imshow(piv.round(1), text_auto=True, aspect="auto",
                        color_continuous_scale="OrRd",
                        labels=dict(color="CDS (%)"),
                        title="Fig. 4.9 — Cartographie type de crédit × région")
        st.plotly_chart(fig_style(fig, 460), use_container_width=True)

    with t4:
        enc = np.sort(dfe["capital_restant_du"].values)
        cum = np.insert(np.cumsum(enc) / enc.sum(), 0, 0)
        x = np.linspace(0, 1, len(cum))
        gini = 1 - 2 * _trapz(cum, x)
        fig = go.Figure()
        fig.add_scatter(x=x, y=cum, name="Courbe de Lorenz",
                        line=dict(color=ORANGE, width=3), fill="tonexty")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Égalité parfaite",
                        line=dict(dash="dash", color=GRIS))
        fig.update_layout(title=f"Fig. 4.10 — Courbe de Lorenz "
                                f"(indice de Gini = {gini:.3f})",
                          xaxis_title="Proportion cumulée des crédits",
                          yaxis_title="Proportion cumulée de l'encours")
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(fig_style(fig, 470), use_container_width=True)

        def hhi(s):
            p = s / s.sum()
            return int(10000 * (p ** 2).sum())
        c2.markdown("#### Indices HHI (tab. 6.2)")
        c2.metric("HHI par produit",
                  hhi(dfe.groupby('type_credit')['capital_restant_du'].sum()))
        c2.metric("HHI par région",
                  hhi(dfe.groupby('region')['capital_restant_du'].sum()))
        c2.metric("HHI par agence",
                  hhi(dfe.groupby('agence')['capital_restant_du'].sum()))

# ============================================================================
# PAGE — TESTS STATISTIQUES & RÉGRESSION LOGISTIQUE (chapitre 5)
# ============================================================================
elif page.startswith("📐"):
    st.title("Analyses statistiques et économétriques")

    t1, t2, t3, t4 = st.tabs(["Chi-deux (5.1)", "Proportions (5.2)",
                              "Mann-Whitney (5.3)", "Régression logistique (5.4)"])

    with t1:
        st.markdown("**Le type de crédit est-il indépendant de la qualité "
                    "de remboursement ?**")
        tc = pd.crosstab(df["type_credit"], df["cds"])
        tc.columns = ["Sain", "Créance en souffrance"]
        chi2, p, ddl, _ = st_stats.chi2_contingency(tc)
        v_cramer = np.sqrt(chi2 / (len(df) * (min(tc.shape) - 1)))
        c1, c2 = st.columns([1.2, 1])
        c1.dataframe(tc, use_container_width=True)
        c2.metric("χ²", f"{chi2:.2f}")
        c2.metric("p-value", f"{p:.2e}")
        c2.metric("V de Cramér", f"{v_cramer:.3f}")
        verdict = "rejetée" if p < 0.05 else "non rejetée"
        st.success(f"Hypothèse d'indépendance **{verdict}** (ddl = {ddl}) : "
                   "le type de crédit et la qualité de remboursement sont liés.")

    with t2:
        st.markdown("**Fonctionnaires vs auto-entrepreneurs : le risque "
                    "diffère-t-il significativement ?**")
        g1 = df[df["segment_client"] == "Particulier - Fonctionnaire"]["cds"]
        g2 = df[df["segment_client"] == "Auto-entrepreneur"]["cds"]
        p1, p2, n1, n2 = g1.mean(), g2.mean(), len(g1), len(g2)
        p_pool = (g1.sum() + g2.sum()) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        pv = 2 * st_stats.norm.sf(abs(z))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CDS Fonctionnaires", f"{p1*100:.2f} %", f"n = {n1}")
        c2.metric("CDS Auto-entrepreneurs", f"{p2*100:.2f} %", f"n = {n2}")
        c3.metric("Statistique z", f"{z:.2f}")
        c4.metric("p-value", f"{pv:.2e}")
        st.success(f"Écart hautement significatif — risque relatif ≈ "
                   f"**{p2/max(p1,1e-9):.1f}×** supérieur pour les "
                   "auto-entrepreneurs.")

    with t3:
        st.markdown("**L'ancienneté bancaire diffère-t-elle selon le statut ?**")
        a_sain = df[df["cds"] == 0]["anciennete_client_mois"]
        a_inc = df[df["cds"] == 1]["anciennete_client_mois"]
        u, pv = st_stats.mannwhitneyu(a_sain, a_inc, alternative="greater")
        c1, c2, c3 = st.columns(3)
        c1.metric("Médiane clients sains", f"{a_sain.median():.0f} mois")
        c2.metric("Médiane clients en incident", f"{a_inc.median():.0f} mois")
        c3.metric("p-value (unilatéral)", f"{pv:.4f}")
        fig = px.box(df, x="cds", y="anciennete_client_mois",
                     labels={"cds": "0 = Sain · 1 = Incident",
                             "anciennete_client_mois": "Ancienneté (mois)"},
                     title="Ancienneté de la relation selon le statut")
        st.plotly_chart(fig_style(fig, 380), use_container_width=True)

    with t4:
        st.markdown("**Quels facteurs expliquent l'incident, toutes choses "
                    "égales par ailleurs ?**")
        res, fpr, tpr, auc, prsq, llr_p = estimer_logit(df)
        st.dataframe(res, use_container_width=True)

        # Forest plot des odds ratios
        orp = res.drop("Intercept")
        fig = go.Figure()
        fig.add_scatter(x=orp["Odds ratio"], y=orp.index, mode="markers",
                        marker=dict(size=11, color=np.where(
                            orp["Odds ratio"] > 1, ORANGE, VERT)),
                        error_x=dict(type="data", symmetric=False,
                                     array=orp["IC 95 % haut"] - orp["Odds ratio"],
                                     arrayminus=orp["Odds ratio"] - orp["IC 95 % bas"]))
        fig.add_vline(x=1, line_dash="dash", line_color=GRIS)
        fig.update_layout(title="Fig. 5.1 — Odds ratios (IC 95 %)",
                          xaxis_title="Odds ratio")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_style(fig, 420), use_container_width=True)

        # Courbe ROC (issue du calcul mis en cache)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, name=f"Modèle logit (AUC = {auc:.3f})",
                        line=dict(color=BLEU, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Aléatoire (AUC = 0,5)",
                        line=dict(dash="dash", color=GRIS))
        fig.update_layout(title="Fig. 5.2 — Courbe ROC",
                          xaxis_title="Taux de faux positifs",
                          yaxis_title="Taux de vrais positifs")
        c2.plotly_chart(fig_style(fig, 420), use_container_width=True)

        st.info(f"Pseudo-R² de McFadden = {prsq:.3f} · "
                f"AUC = {auc:.3f} · LR-test p = {llr_p:.2e}. "
                "Modèle volontairement parcimonieux et interprétable "
                "(arbitrage justifié en section 5.5 du mémoire).")

# ============================================================================
# PAGE — NOTATION INTERNE & SIMULATEUR (chapitre 7.2-7.4)
# ============================================================================
elif page.startswith("⭐"):
    st.title("Dispositif de notation interne simulé")

    t1, t2 = st.tabs(["📊 Portefeuille noté", "🧮 Simulateur client"])

    with t1:
        c1, c2 = st.columns(2)
        fig = px.histogram(df, x="score_credit", nbins=60,
                           title="Fig. 7.1 — Distribution du score (300-900)")
        for x0, x1, coul in [(300, 610, "rgba(176,58,46,0.14)"),
                             (610, 650, "rgba(217,123,41,0.14)"),
                             (650, 690, "rgba(201,162,39,0.14)"),
                             (690, 900, "rgba(30,127,79,0.12)")]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=coul, line_width=0)
        fig.add_vline(x=df["score_credit"].mean(), line_dash="dash",
                      annotation_text=f"Moyenne {df['score_credit'].mean():.0f}")
        c1.plotly_chart(fig_style(fig), use_container_width=True)

        agg = df.groupby("rating_interne").agg(
            n=("id_credit", "count"), pd_moy=("pd_1an", "mean")) \
            .reindex(RATING_ORDER)
        fig = go.Figure()
        fig.add_bar(x=agg.index, y=agg["pd_moy"] * 100,
                    text=[f"{v*100:.1f}%<br>(n={int(n)})"
                          for v, n in zip(agg["pd_moy"], agg["n"])],
                    marker_color=[VERT]*3 + [OR]*2 + [ROUGE]*3)
        fig.update_layout(title="Fig. 7.2 — PD moyenne à 1 an par rating",
                          yaxis_title="PD (%)", xaxis_title="Notation interne")
        c2.plotly_chart(fig_style(fig), use_container_width=True)

        c1, c2 = st.columns(2)
        fig = px.box(df, x="garantie", y=df["lgd"] * 100,
                     labels={"y": "LGD (%)", "garantie": ""},
                     title="Fig. 7.3 — LGD simulée par garantie (loi Bêta)")
        c1.plotly_chart(fig_style(fig), use_container_width=True)

        heat = df.pivot_table(index="type_credit", columns="segment_client",
                              values="score_credit", aggfunc="mean")
        fig = px.imshow(heat.round(0), text_auto=True, aspect="auto",
                        color_continuous_scale="RdYlGn",
                        title="Fig. 7.7 — Score moyen produit × segment")
        c2.plotly_chart(fig_style(fig), use_container_width=True)

        st.subheader("Matrice de transition du rating (fig. 7.8 — normalisée)")
        st.caption("Matrice pédagogique : diagonale dominante, asymétrie "
                   "dégradation > amélioration, **chaque ligne somme à 100 %** "
                   "(correction de la ligne C du mémoire).")
        M = np.array([
            [87.0, 9.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.2],
            [5.0, 84.0, 7.0, 2.0, 0.8, 0.4, 0.2, 0.6],
            [1.0, 6.0, 81.0, 7.0, 2.5, 1.0, 0.5, 1.0],   # C corrigée : 82→81
            [0.5, 2.0, 7.0, 79.0, 7.0, 2.5, 1.0, 1.0],
            [0.2, 0.8, 2.0, 8.0, 77.0, 8.0, 2.5, 1.5],
            [0.1, 0.3, 0.8, 2.0, 7.0, 76.0, 10.0, 3.8],
            [0.1, 0.2, 0.5, 1.0, 3.0, 8.0, 74.0, 13.2],
            [0.1, 0.1, 0.2, 0.5, 1.5, 3.0, 8.0, 86.6]])
        fig = px.imshow(M, x=RATING_ORDER, y=RATING_ORDER, text_auto=True,
                        color_continuous_scale="YlOrRd",
                        labels=dict(x="Rating en t+1", y="Rating en t",
                                    color="%"))
        st.plotly_chart(fig_style(fig, 520), use_container_width=True)
        somme = M.sum(axis=1)
        st.caption("Contrôle : sommes des lignes = "
                   + " · ".join(f"{r}: {s:.1f} %" for r, s in
                                zip(RATING_ORDER, somme)))

    with t2:
        st.markdown("### Notez un dossier en direct")
        st.caption("Le simulateur applique la construction exacte du score "
                   "(section 7.2), sans bruit idiosyncratique — il restitue "
                   "l'espérance du score.")
        c1, c2, c3 = st.columns(3)
        seg = c1.selectbox("Segment de clientèle",
                           SEGMENTS_PART + SEGMENTS_PRO)
        typ = c1.selectbox("Type de crédit", TYPE_ORDER)
        gar = c2.selectbox("Garantie", ["Hypotheque", "Garantie Tamwilcom",
                                        "Nantissement", "Aucune"])
        anc = c2.slider("Ancienneté bancaire (mois)", 0, 300, 48)
        mnt = c3.number_input("Montant demandé (DH)", 5_000, 3_000_000,
                              150_000, step=5_000)

        base_seg = {"Particulier - Fonctionnaire": 690,
                    "Particulier - Salarie prive": 635,
                    "Particulier - Retraite": 655,
                    "Professionnel - TPE": 575, "Auto-entrepreneur": 555}
        s = base_seg[seg] + 0.50 * min(anc, 180)
        s += {"Hypotheque": 40, "Garantie Tamwilcom": 25,
              "Nantissement": 12, "Aucune": 0}[gar]
        s += {"Immobilier": 18, "Consommation": 0, "Avance sur salaire": 12,
              "CMLT": -20, "Facilite de caisse": -30, "Intelak": -8}[typ]
        mu, sig = np.log(df["montant_accorde"]).mean(), \
                  np.log(df["montant_accorde"]).std()
        s += -7.0 * max((np.log(mnt) - mu) / sig, 0)
        s = float(np.clip(s, 300, 900))

        bins = [300, 500, 560, 610, 650, 690, 730, 780, 901]
        labels = ["H", "G", "F", "E", "D", "C", "B", "A"]
        rating = labels[np.digitize(s, bins) - 1]
        pd1 = PD_CIBLE[rating]
        lgd_m = {"Hypotheque": 0.26, "Garantie Tamwilcom": 0.37,
                 "Nantissement": 0.48, "Aucune": 0.61}[gar]
        el = pd1 * lgd_m * mnt

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=s,
            title={"text": f"Score — Rating <b>{rating}</b>"},
            gauge={"axis": {"range": [300, 900]},
                   "bar": {"color": BLEU},
                   "steps": [{"range": [300, 610], "color": "#F5B7B1"},
                             {"range": [610, 650], "color": "#FAD7A0"},
                             {"range": [650, 690], "color": "#F9E79F"},
                             {"range": [690, 900], "color": "#ABEBC6"}]}))
        fig.update_layout(height=320, margin=dict(t=60, b=10))
        c1b, c2b = st.columns([1.2, 1])
        c1b.plotly_chart(fig, use_container_width=True)
        c2b.metric("Rating interne", rating)
        c2b.metric("PD à 1 an (cible)", f"{pd1*100:.2f} %")
        c2b.metric("LGD moyenne (garantie)", f"{lgd_m*100:.1f} %")
        c2b.metric("EL à 1 an (approx.)", f"{el:,.0f} DH".replace(",", " "))
        if rating in ("A", "B", "C"):
            st.success("Classe de **faible risque** — octroi standard envisageable.")
        elif rating in ("D", "E"):
            st.warning("Risque **modéré à élevé** — garanties et instruction "
                       "renforcées recommandées (R1, R2 du mémoire).")
        else:
            st.error("Risque **très élevé** — comité de crédit et garantie "
                     "exigée ; plafonnement du montant conseillé.")

# ============================================================================
# PAGE — EL / ECL / CAPITAL (chapitre 7.5-7.7)
# ============================================================================
elif page.startswith("💰"):
    st.title("Pertes attendues et capital économique")

    conf_lbl = st.radio("Niveau de confiance IRB",
                        ["99,9 % — réglementaire Bâle III (z = 3,090)",
                         "99 % — valeur erronée du mémoire (z = 2,326)",
                         "Comparer les deux"], horizontal=True)

    res999 = calculer_pertes(df, 0.0, 0.999)
    res99 = calculer_pertes(df, 0.0, 0.99)
    res = res99 if conf_lbl.startswith("99 %") else res999

    ead, el, ecl, cap = (res["ead"].sum(), res["el_1an"].sum(),
                         res["ecl"].sum(), res["capital_eco"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EAD totale", f"{ead/1e6:,.1f} M DH".replace(",", " "))
    c2.metric("EL 1 an (Bâle III)", f"{el/1e6:,.2f} M DH".replace(",", " "),
              f"{el/ead*100:.2f} % de l'EAD")
    c3.metric("ECL IFRS 9", f"{ecl/1e6:,.2f} M DH".replace(",", " "),
              f"{ecl/ead*100:.2f} % de l'EAD")
    c4.metric("Capital économique IRB", f"{cap/1e6:,.2f} M DH".replace(",", " "),
              f"{cap/ead*100:.2f} % de l'EAD")

    if conf_lbl == "Comparer les deux":
        d99, d999 = res99["capital_eco"].sum(), res999["capital_eco"].sum()
        fig = go.Figure()
        fig.add_bar(x=["99 % (mémoire — erroné)", "99,9 % (réglementaire)"],
                    y=[d99/ead*100, d999/ead*100],
                    marker_color=[ORANGE, BLEU],
                    text=[f"{d99/ead*100:.2f} %", f"{d999/ead*100:.2f} %"])
        fig.update_layout(title="Impact du niveau de confiance sur le "
                                "capital économique (en % de l'EAD)",
                          yaxis_title="Capital / EAD (%)")
        st.plotly_chart(fig_style(fig, 380), use_container_width=True)
        st.markdown(
            f"""<div class="note">🎓 <b>Argument de soutenance :</b> passer de
            99 % à 99,9 % augmente le capital de
            <b>{(d999/d99 - 1)*100:.0f} %</b>
            ({d99/1e6:.1f} → {d999/1e6:.1f} M DH). La formule IRB retail de
            Bâle III impose Φ⁻¹(0,999) = 3,090 — c'est la correction majeure
            apportée au chapitre 7 du mémoire.</div>""",
            unsafe_allow_html=True)

    st.subheader("EL vs capital économique par type de crédit (fig. 7.6)")
    agg = res.groupby("type_credit").agg(
        ead=("ead", "sum"), el=("el_1an", "sum"),
        cap=("capital_eco", "sum")).reindex(TYPE_ORDER)
    fig = go.Figure()
    fig.add_bar(name="EL à 1 an (perte attendue)", x=agg.index,
                y=agg["el"]/agg["ead"]*100, marker_color=BLEU)
    fig.add_bar(name="Capital IRB (perte inattendue)", x=agg.index,
                y=agg["cap"]/agg["ead"]*100, marker_color=ORANGE)
    fig.update_layout(barmode="group", yaxis_title="% de l'EAD")
    st.plotly_chart(fig_style(fig), use_container_width=True)

    tab = pd.DataFrame({
        "EAD (M DH)": (agg["ead"]/1e6).round(1),
        "EL/EAD (%)": (agg["el"]/agg["ead"]*100).round(2),
        "Capital/EAD (%)": (agg["cap"]/agg["ead"]*100).round(2),
        "Ratio CE/EL": (agg["cap"]/agg["el"]).round(1)})
    tab.loc["TOTAL"] = [ead/1e6, el/ead*100, cap/ead*100, cap/el]
    st.dataframe(tab.round(2), use_container_width=True)

    st.subheader("ECL IFRS 9 par stage (fig. 7.5 — source de calcul unique)")
    stg = res.groupby("stage_ifrs9").agg(
        n=("id_credit", "count"), ecl=("ecl", "sum"), ead=("ead", "sum"))
    stg["ecl_ead"] = stg["ecl"] / stg["ead"] * 100
    noms = {1: "Stage 1<br>(sain, 12 mois)", 2: "Stage 2<br>(dégradé, lifetime)",
            3: "Stage 3<br>(défaut avéré)"}
    fig = go.Figure()
    fig.add_bar(x=[noms[i] for i in stg.index], y=stg["ecl"]/1e6,
                marker_color=[VERT, ORANGE, ROUGE],
                text=[f"{v/1e6:.2f} M DH<br>({r:.2f} % EAD)<br>[n={int(n)}]"
                      for v, r, n in zip(stg["ecl"], stg["ecl_ead"], stg["n"])])
    fig.update_layout(yaxis_title="ECL (M DH)")
    st.plotly_chart(fig_style(fig, 420), use_container_width=True)
    st.caption("Ici, la figure et le tableau proviennent du **même calcul** — "
               "correction de l'incohérence figure 7.5 / tableau 7.5 du mémoire. "
               "Rappel conceptuel : l'ECL est couverte par les **provisions**, "
               "le capital couvre la **perte inattendue** (UL = VaR − EL).")

# ============================================================================
# PAGE — STRESS-TEST (chapitre 7.10)
# ============================================================================
elif page.startswith("⚡"):
    st.title("Analyse de stress-test interactive")
    st.markdown("Choc **multiplicatif sur les PD** (méthodologies BAM / "
                "Comité de Bâle) : la PD entre linéairement dans EL, l'effet "
                "sur l'ECL est amplifié par la composante lifetime.")

    choc = st.slider("Choc sur les PD (%)", 0, 200, 80, step=10,
                     help="+80 % ≈ récession sévère · +150 % ≈ crise systémique")
    strs = calculer_pertes(df, choc / 100, 0.999)
    b0 = calculer_pertes(df, 0.0, 0.999)

    ead = strs["ead"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("EL 1 an", f"{strs['el_1an'].sum()/1e6:.1f} M DH",
              f"{(strs['el_1an'].sum()/b0['el_1an'].sum()-1)*100:+.0f} % vs base")
    c2.metric("ECL IFRS 9", f"{strs['ecl'].sum()/1e6:.1f} M DH",
              f"{(strs['ecl'].sum()/b0['ecl'].sum()-1)*100:+.0f} % vs base")
    c3.metric("ECL / EAD", f"{strs['ecl'].sum()/ead*100:.2f} %")

    # Courbe complète 0 → 200 % (mise en cache : le slider est instantané)
    courbe = courbe_stress(df)

    fig = go.Figure()
    fig.add_scatter(x=courbe["choc"], y=courbe["EL/EAD"], name="EL / EAD",
                    line=dict(color=BLEU, width=3))
    fig.add_scatter(x=courbe["choc"], y=courbe["ECL/EAD"], name="ECL / EAD",
                    line=dict(color=ROUGE, width=3))
    fig.add_scatter(x=courbe["choc"], y=courbe["Capital/EAD"],
                    name="Capital IRB / EAD (99,9 %)",
                    line=dict(color=ORANGE, width=2, dash="dot"))
    fig.add_vline(x=choc, line_dash="dash", line_color=GRIS,
                  annotation_text=f"Choc sélectionné : +{choc} %")
    fig.update_layout(title="Sensibilité des indicateurs au choc de PD "
                            "(fig. 7.9 généralisée)",
                      xaxis_title="Choc sur les PD (%)",
                      yaxis_title="% de l'EAD")
    st.plotly_chart(fig_style(fig, 470), use_container_width=True)

    scn = pd.DataFrame({
        "Scénario": ["Base", "Modéré (+30 %)", "Stressé (+80 %)",
                     "Extrême (+150 %)"],
        "EL (M DH)": [calculer_pertes(df, c, 0.999)["el_1an"].sum()/1e6
                      for c in (0, .3, .8, 1.5)],
        "ECL (M DH)": [calculer_pertes(df, c, 0.999)["ecl"].sum()/1e6
                       for c in (0, .3, .8, 1.5)]})
    scn["EL/EAD (%)"] = scn["EL (M DH)"]*1e6/ead*100
    scn["ECL/EAD (%)"] = scn["ECL (M DH)"]*1e6/ead*100
    st.dataframe(scn.round(2), use_container_width=True, hide_index=True)

    st.markdown(
        """<div class="note">🎓 <b>Lecture corrigée (vs conclusion du
        mémoire) :</b> on ne compare pas directement l'ECL au capital — l'ECL
        est absorbée par les <b>provisions</b>, le capital absorbe la perte
        <b>inattendue</b>. Le bon message : sous choc sévère, la hausse
        simultanée des provisions requises <i>et</i> du capital IRB signale
        la nécessité d'un <b>coussin de fonds propres renforcé</b>.</div>""",
        unsafe_allow_html=True)

# ============================================================================
# PAGE — TABLEAU DE BORD (chapitre 8)
# ============================================================================
elif page.startswith("🎛️"):
    st.title("Tableau de bord de pilotage du risque")

    cds_nb = base["cds"].mean() * 100
    cds_enc = base.loc[base["cds"] == 1, "capital_restant_du"].sum() \
        / base["capital_restant_du"].sum() * 100
    el_r, ecl_r, cap_r = EL_T/EAD_T*100, ECL_T/EAD_T*100, CAP_T/EAD_T*100
    parts = base.groupby("type_credit")["capital_restant_du"].sum()
    hhi_p = int(10000 * ((parts/parts.sum())**2).sum())
    score_moy = base["score_credit"].mean()
    part_fgh = base["rating_interne"].isin(["F", "G", "H"]).mean() * 100

    kpis = [
        ("Taux de CDS (nombre)", f"{cds_nb:.2f} %", badge(cds_nb, 5, 8)),
        ("Taux de CDS (encours)", f"{cds_enc:.2f} %", badge(cds_enc, 4, 6)),
        ("EL / EAD à 1 an", f"{el_r:.2f} %", badge(el_r, 3, 5)),
        ("ECL / EAD IFRS 9", f"{ecl_r:.2f} %", badge(ecl_r, 5, 8)),
        ("Capital IRB / EAD (99,9 %)", f"{cap_r:.2f} %", badge(cap_r, 8, 12)),
        ("HHI produit", f"{hhi_p}", badge(hhi_p, 2500, 4000)),
        ("Score moyen", f"{score_moy:.0f}", badge(score_moy, 650, 600, "<")),
        ("% dossiers F/G/H", f"{part_fgh:.1f} %", badge(part_fgh, 18, 25)),
    ]
    st.subheader("KPI de premier niveau (tab. 8.1)")
    for ligne in (kpis[:4], kpis[4:]):
        cols = st.columns(4)
        for col, (nom, val, bd) in zip(cols, ligne):
            col.markdown(
                f"""<div style="background:#F5F8FB;border-left:5px solid {BLEU};
                padding:12px;border-radius:6px;">
                <div style="color:{GRIS};font-size:0.85rem;">{nom}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{BLEU};">
                {val}</div>{bd}</div>""", unsafe_allow_html=True)
    st.markdown("")

    st.subheader("Classement des agences (fig. 8.1)")
    ag = base.groupby("agence").agg(
        n=("id_credit", "count"), cds=("cds", "mean"),
        score=("score_credit", "mean"), el=("el_1an", "sum"))
    ag = ag[ag["n"] >= 15]
    c1, c2 = st.columns(2)

    top_risk = ag.nlargest(8, "cds")
    fig = px.bar(x=top_risk["cds"]*100, y=top_risk.index, orientation="h",
                 text=(top_risk["cds"]*100).round(1),
                 labels={"x": "Taux de CDS (%)", "y": ""},
                 title="À surveiller en priorité (n ≥ 15)")
    fig.update_traces(marker_color=ROUGE)
    fig.update_yaxes(autorange="reversed")
    c1.plotly_chart(fig_style(fig, 420), use_container_width=True)

    top_best = ag.nsmallest(8, "cds")
    fig = px.bar(x=top_best["cds"]*100, y=top_best.index, orientation="h",
                 text=(top_best["cds"]*100).round(1),
                 labels={"x": "Taux de CDS (%)", "y": ""},
                 title="Les plus performantes (n ≥ 15)")
    fig.update_traces(marker_color=VERT)
    fig.update_yaxes(autorange="reversed")
    c2.plotly_chart(fig_style(fig, 420), use_container_width=True)

    st.warning("⚠️ **Mise en garde méthodologique (mémoire, §8.3)** : ce "
               "classement est un signal d'alerte à confirmer sur plusieurs "
               "trimestres — un taux élevé dans une petite agence peut refléter "
               "3-4 dossiers seulement, pas une pratique d'octroi défaillante.")

    st.subheader("Vue prospective : part du Stage 2 (signal d'alerte avancé)")
    stg2 = base.groupby("type_credit").apply(
        lambda g: (g["stage_ifrs9"] == 2).mean() * 100,
        include_groups=False).reindex(TYPE_ORDER)
    fig = px.bar(x=stg2.index, y=stg2.values, text=stg2.round(1),
                 labels={"x": "", "y": "Part des dossiers en Stage 2 (%)"},
                 title="Une hausse du Stage 2 précède l'entrée en souffrance "
                       "réglementaire")
    st.plotly_chart(fig_style(fig, 380), use_container_width=True)

    st.download_button(
        "⬇️ Exporter la synthèse agences (CSV)",
        ag.round(4).to_csv().encode("utf-8"),
        "synthese_agences.csv", "text/csv")
