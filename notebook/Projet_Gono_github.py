import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler  # Remplacement par l'Undersampling
from imblearn.pipeline import Pipeline  # Indispensable pour éviter le data leakage

# ------  TRAITEMENT DES DONNÉES BRUTES -------

# 1. Chargement et prétraitement initial
# Utilisation d'un chemin relatif pour assurer la portabilité du code sur GitHub
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
path = os.path.join(BASE_DIR, 'data/gono.csv')
df = pd.read_csv(path, sep=';', skipinitialspace=True)

# Suppression des colonnes non prédictives ou redondantes
# IDENT (col 1) et CULTURE (col 11) 
cols_inutiles = ['ID', 'CULTURE']
df = df.drop(columns=[c for c in cols_inutiles if c in df.columns], errors='ignore')

# 2. Identification et conversion des valeurs manquantes selon l'énoncé
# Les blancs sont considérés comme manquants 
df = df.replace(r'^\s*$', np.nan, regex=True)

# Codes 9 considérés comme manquants pour les variables spécifiées 
cols_9 = ['ETAT_C', 'ORIENT_SEX', 'MTS_ANT', 'RAISON', 'HISTOIRE', 'DIAGN']
for col in cols_9:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(9, np.nan)

# Codes 99 considérés comme manquants pour l'âge et les partenaires 
cols_99 = ['AGE', 'NB_PART']
for col in cols_99:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(99, np.nan)

# Affichage de l'état initial des données manquantes
print("--- Valeurs manquantes réelles (Blancs + Codes 9/99) ---")
print(df.isnull().sum())
print(f"\nDimension du DataFrame avant suppression : {df.shape}")
print("-" * 50)

# 3. Nettoyage final par suppression des lignes avec NaNs
df = df.dropna()

print("--- Valeurs manquantes après suppression (Confirmation) ---")
print(df.isnull().sum())
print(f"\nDimension du DataFrame final (df) : {df.shape}")
print("-" * 50)


# COMMENTAIRE :
# Notre prétraitement prépare le jeu de données :
# Élagage : Suppression de ID (sans valeur prédictive) et CULTURE (redondant avec la cible). 
# Standardisation : Conversion des blancs et des codes spécifiques (9 et 99) en valeurs manquantes (NaN). 
# Nettoyage : Suppression des lignes incomplètes par dropna() : L'échantillon final reste robuste 
# car la majorité des observations est conservée, permettant une caractérisation fiable des groupes à risque.


# ---- CRÉATION DU DATAFRAME DUMMIES -------

# 1. Création du DataFrame et transformation des variables binaires
df_dummy = pd.DataFrame()

df_dummy['HOMME'] = df['SEXE'].map({1: 1, 2: 0})
df_dummy['AGE_MOINS_30'] = (df['AGE'] < 30).astype(float)
df_dummy['ORIENTATION_HOMOSEXUELLE'] = df['ORIENT_SEX'].map({1: 1, 2: 0})
df_dummy['MTS_ANTERIEURE'] = df['MTS_ANT'].map({2: 1, 1: 0})
df_dummy['CONTACT_CONTAMINE'] = df['HISTOIRE'].map({1: 1, 0: 0})

# Dichotomisation du nombre de partenaires par la médiane
median_part = df['NB_PART'].median()
df_dummy['PARTENAIRES_NOMBRE_ELEVE'] = (df['NB_PART'] > median_part).astype(float)

# 2. Encodage dynamique des variables multicatégorielles
# On définit des dictionnaires pour mapper les codes aux noms 
etat_map = {1: 'CELIBATAIRE', 2: 'MARIE', 3: 'SEPARE', 4: 'VEUF'}
raison_map = {1: 'SYMPTOMES', 2: 'CONTACT', 3: 'DEPISTAGE', 4: 'CONTROLE', 5: 'AUTRE'}

# Utilisation de drop_first=True pour éviter le piège de la multicolinéarité parfaite en régression
etat_dummies = pd.get_dummies(df['ETAT_C'].map(etat_map), prefix='ETAT', drop_first=True).astype(float)
raison_dummies = pd.get_dummies(df['RAISON'].map(raison_map), prefix='VISITE', drop_first=True).astype(float)

# 3. Assemblage final et variable cible
df_dummy = pd.concat([df_dummy, etat_dummies, raison_dummies], axis=1)
df_dummy['DIAGNOSTIC_GONORRHEE'] = df['DIAGN'].astype(float)


# 5. Vérification de l'absence de valeurs manquantes dans le DataFrame final
# Cette étape garantit que les transformations (mapping/dummies) n'ont pas généré de NaNs imprévus
print("--- État des valeurs manquantes dans df_dummy ---")
print(df_dummy.isnull().sum())
print(f"\nDimension finale pour la régression : {df_dummy.shape}")
print("-" * 50)


# Cette étape prépare les données pour la modélisation afin de caractériser précisément les groupes à risque:
    # Création du dataframe : "df_dummy"
    # Dichotomisation : Transformation de l'âge (seuil de 30 ans), des antécédents de MTS et de l'activité sexuelle en 
    # variables binaires selon les recommandations de l'étude.
    # Encodage Catégoriel : Création de dummies pour l'état civil et la raison de la visite afin de les intégrer au modèle.
    # Standardisation : Conversion de toutes les variables en format numérique (float) pour assurer 
    # la compatibilité avec les algorithmes de calcul.
    # Validation : Vérification finale de l'absence de valeurs manquantes.


# ---- AED SUR DONNÉES BINAIRES ----

# 1. Configuration et affichage des répartitions (Comptages)
cols_to_plot = df_dummy.columns
n_cols = 3
n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

plt.figure(figsize=(15, n_rows * 4))
for i, col in enumerate(cols_to_plot):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.countplot(data=df_dummy, x=col, hue=col, palette='viridis', legend=False)
    plt.title(f"Répartition : {col}")
    plt.xlabel("Valeur (0 ou 1)")
    plt.ylabel("Effectif")

plt.tight_layout()
plt.show()

# 2. Analyse des corrélations avec clustering hiérarchique (Dendrogramme)
corr_matrix = df_dummy.corr()

cluster_map = sns.clustermap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    figsize=(12, 10),
    cbar_pos=(0.02, 0.8, 0.03, 0.15)
)
plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.suptitle("Clustering hiérarchique des variables de risque", y=1.02)
plt.show()


# COMMENTAIRE :
# L'AED révèle des points critiques pour la stabilité de la modélisation :
# - La corrélation parfaite entre 'VISITE_CONTACT' et 'CONTACT_CONTAMINE' 
#   impose la suppression d'une des deux variables pour éviter l'instabilité des coefficients.
# - L'orientation homosexuelle émerge comme le principal facteur associé 
#   au diagnostic positif, tandis que l'état 'marié' est le facteur le plus 
#   protecteur, bien que son impact soit modéré (Phi = -0.08).
# - Le fort déséquilibre observé nécessite un rééquilibrage par sous-échantillonnage (Undersampling).
# - Suppression des variables redondantes identifiées.

# Application des conclusions de l'AED avant la modélisation
cols_a_retirer_aed = ['VISITE_CONTACT', 'ETAT_VEUF']
df_dummy = df_dummy.drop(columns=[c for c in cols_a_retirer_aed if c in df_dummy.columns], errors='ignore')


# ----- PRÉPARATION DES MATRICES INITIALES (X, y) ----

# Séparation des features et de la cible sur le dataset initial non-équilibré
X_initial = df_dummy.drop(columns=['DIAGNOSTIC_GONORRHEE'])
y_initial = df_dummy['DIAGNOSTIC_GONORRHEE']


# Pour le choix du modèle, nous avons plusieurs possibilités, qui ont chacune leurs avantages.
# La Régression Logistique est appropriée grâce à son interprétabilité via les Odds Ratios.
# Le Naive Bayes est également robuste et interprétable pour les données binaires.
# À l'inverse, le kNN agit comme une "boîte noire" moins adaptée à l'interprétation.
# Nous allons donc comparer la régression logistique et le naive Bayes par validation
# croisée pour identifier le meilleur compromis.


# ---- CHOIX DU MODÈLE : RÉGRESSION LOGISTIQUE & NAIVE BAYES ----

# 1. Fonction stepwise backward selection
def backward_selection(X, y, threshold_out=0.05):
    features = list(X.columns)
    while len(features) > 0:
        X_const = sm.add_constant(X[features])
        model = sm.Logit(y, X_const).fit(disp=0)
        p_values = model.pvalues.drop('const', errors='ignore')
        max_p_value = p_values.max()
        if max_p_value > threshold_out:
            features.remove(p_values.idxmax())
        else:
            break
    return features

# Sélection effectuée sur les données brutes pour respecter la structure statistique
final_features = backward_selection(X_initial, y_initial)
X_reduced = X_initial[final_features]

# Stratégie de validation croisée à 10 plis
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 2. Configuration des Pipelines avec RandomUnderSampler pour éviter tout data leakage
# Le rééquilibrage n'est dorénavant appliqué que sur les plis de calcul (Train Folds)
pipeline_lr = Pipeline([
    ('sampler', RandomUnderSampler(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline_nb = Pipeline([
    ('sampler', RandomUnderSampler(random_state=42)),
    ('classifier', BernoulliNB())
])

# 3. Exécution des prédictions robustes par validation croisée
y_pred_lr = cross_val_predict(pipeline_lr, X_reduced, y_initial, cv=cv)
y_prob_lr = cross_val_predict(pipeline_lr, X_reduced, y_initial, cv=cv, method='predict_proba')[:, 1]
cm_lr = confusion_matrix(y_initial, y_pred_lr)

y_pred_nb = cross_val_predict(pipeline_nb, X_reduced, y_initial, cv=cv)
y_prob_nb = cross_val_predict(pipeline_nb, X_reduced, y_initial, cv=cv, method='predict_proba')[:, 1]
cm_nb = confusion_matrix(y_initial, y_pred_nb)

# 4. Visualisation comparative
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Comparaison des courbes ROC
fpr_lr, tpr_lr, _ = roc_curve(y_initial, y_prob_lr)
fpr_nb, tpr_nb, _ = roc_curve(y_initial, y_prob_nb)

ax[0].plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Régression Logistique (AUC = {auc(fpr_lr, tpr_lr):.3f})')
ax[0].plot(fpr_nb, tpr_nb, color='rebeccapurple', lw=2, label=f'Naive Bayes (AUC = {auc(fpr_nb, tpr_nb):.3f})')
ax[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
ax[0].set_title('Comparaison des Courbes ROC (Sous-échantillonnage)')
ax[0].set_xlabel('Taux de Faux Positifs (FPR)')
ax[0].set_ylabel('Taux de Vrais Positifs (TPR)')
ax[0].legend(loc="lower right")

# Calcul des scores globaux
metrics = {
    'Modèle': ['Régression Logistique', 'Naive Bayes'],
    'AUC': [auc(fpr_lr, tpr_lr), auc(fpr_nb, tpr_nb)],
    'Exactitude (Accuracy)': [(cm_lr[0,0]+cm_lr[1,1])/len(y_initial), (cm_nb[0,0]+cm_nb[1,1])/len(y_initial)],
    'Sensibilité (Recall)': [cm_lr[1,1]/(cm_lr[1,1]+cm_lr[1,0]), cm_nb[1,1]/(cm_nb[1,1]+cm_nb[1,0])],
    'Spécificité': [cm_lr[0,0]/(cm_lr[0,0]+cm_lr[0,1]), cm_nb[0,0]/(cm_nb[0,0]+cm_nb[0,1])]
}
df_comp = pd.DataFrame(metrics)

# Tracé d'un barplot comparatif des métriques
df_melted = df_comp.melt(id_vars='Modèle', var_name='Métrique', value_name='Score')
sns.barplot(data=df_melted[df_melted['Métrique'] != 'AUC'], x='Métrique', y='Score', hue='Modèle', ax=ax[1], palette=['#3498DB', '#9B59B6'])
ax[1].set_title('Comparaison des Performances Cliniques')
ax[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# 5. Affichage du tableau de synthèse
print("\n" + "="*55)
print("SYNTHÈSE COMPARATIVE DES MODÈLES (UNDERSAMPLING)")
print("="*55)
print(df_comp.to_string(index=False))
print("="*55)


# L'égalité des AUC (~0.65) confirme que les deux modèles possèdent la même puissance intrinsèque.
# Si Naive Bayes est plus sensible (73%), la Régression Logistique peut égaler
# cette performance en abaissant son seuil de décision, rendant l'argument du choix d'algorithme secondaire.
#
# La Régression Logistique reste cependant l'outil de référence grâce à son interprétabilité :
# elle seule fournit des Odds Ratios permettant de quantifier et d'expliquer les leviers de risque.
# Nous utiliserons donc la régression logistique pour expliquer simplement les coefficients du modèle au médecin.


# ---- RÉGRESSION LOGISTIQUE FINALE : COEFFICIENTS ET ODDS RATIOS ----

# Pour analyser globalement la population de manière équilibrée et stable,
# nous construisons le modèle final sur l'intégralité du dataset réduit après sous-échantillonnage.
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_reduced, y_initial)

# Validation de la distribution de la cible après Undersampling
print("--- Validation de l'équilibre de la cible (y) après Undersampling ---")
comptage_y = y_res.value_counts()
print(comptage_y)
print(f"Ratio : {comptage_y[1] / comptage_y[0]:.1f}:1 (Doit être 1:1)")
print("-" * 50)

# Ajout de la constante pour l'estimation de l'intercept beta_0
X_final_const = sm.add_constant(X_res)
final_logit_model = sm.Logit(y_res, X_final_const).fit(disp=0)

# 3. Affichage du rapport statistique détaillé (Summary)
print("\n" + "="*20 + " RAPPORT STATISTIQUE DÉTAILLÉ (UNDERSAMPLING) " + "="*20)
print(final_logit_model.summary())

# 4. Extraction des coefficients, p-values et calcul des Odds Ratios (OR)
summary_df = pd.DataFrame({
    'Beta (Coeff)': final_logit_model.params,
    'P-value': final_logit_model.pvalues,
    'Odds Ratio': np.exp(final_logit_model.params)
})

# 5. Calcul des intervalles de confiance pour les Odds Ratios (IC 95%)
conf = final_logit_model.conf_int()
summary_df['OR_lower'] = np.exp(conf[0])
summary_df['OR_upper'] = np.exp(conf[1])

# 6. Affichage de la synthèse triée par impact (Odds Ratio)
print("\n--- Synthèse des Risques pour le Diagnostic (Undersampling) ---")
print(summary_df.sort_values(by='Odds Ratio', ascending=False))
print("-" * 60)


# ---- INTERPRÉTATION DES RÉSULTATS DE LA RÉGRESSION LOGISTIQUE ----

# 1. Performance globale du modèle
# Le modèle présente une AUC stabilisée autour de 0.65, indiquant une capacité prédictive modérée.
# L'approche par sous-échantillonnage mécanique permet d'ajuster le seuil pour maximiser la sensibilité,
# ce qui est idéal dans un cadre de dépistage clinique (limiter les faux négatifs).

# 2. Analyse des variables influentes (Odds Ratios)
# Les variables significatives retenues par sélection descendante permettent d'identifier les profils à risque.
# Les coefficients calculés suite à l'Undersampling conservent l'orientation et l'importance relative des risques,
# tout en gommant le biais lié au fort déséquilibre initial des classes.

# 3. Recommandations cliniques
# Le médecin doit être en "alerte rouge" face au profil : 
# Homme de moins de 30 ans, homosexuel, avec partenaires multiples.
# Ne pas se laisser rassurer par un patient marié ou asymptomatique (variables non significatives ou exclues).
# Proposer un dépistage systématique et proactif dès qu'au moins deux de ces critères sont réunis.
