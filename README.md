# Modèle de Prédiction et de Caractérisation des Groupes à Risque de la Gonorrhée (Projet GONO)

## 1. INTRODUCTION

### Cadre de réalisation
Projet de modélisation statistique et d'épidémiologie quantitative réalisé dans le cadre de mon master 1 IREF.

### Présentation du sujet
Le dépistage des Infections Transmissibles Sexuellement (ITS / MTS) représente un défi stratégique majeur. Face à la réticence de certaines populations à fréquenter des cliniques spécialisées, l'objectif est d'intégrer le dépistage au sein des consultations privées de médecine générale en tirant parti du réseau de laboratoires des grands centres hospitaliers. La gonorrhée sert ici d'indicateur épidémiologique clé, s'agissant de la seule ITS commune détectable par culture bactériologique de laboratoire standard.

### Intérêt et cas d'usage
Définir et caractériser précisément les groupes cibles à haut risque pour la gonorrhée afin de fournir aux médecins de famille des signaux d'alerte clairs pour identifier rapidement les patients à risque lors des consultations de routine. L'identification des facteurs les plus discriminants permet également d'optimiser l'allocation des ressources de santé publique et de cibler prioritairement les territoires et profils sociaux où déployer les programmes de formation médicale.

---

## 2. SOURCES ET DONNÉES

### Origine des données
Fichier de données épidémiologiques anonymisées `gono.csv` comprenant initialement 3 144 patients examinés dans le cadre du programme de dépistage.

### Périmètre et variables clés
- **Variable cible :** Diagnostic de la gonorrhée (variable binaire : positif / négatif).
- **Variables explicatives :** Données sociodémographiques (âge, état civil), comportementales (orientation sexuelle, nombre de partenaires sexuels au cours du dernier mois), cliniques (antécédents d'ITS, motif de visite) et géographiques.

---

## 3. MÉTHODOLOGIE ET DÉTAILS TECHNIQUES

### Pipeline de traitement et Feature Engineering
- **Nettoyage des données :** Conversion des espaces blancs en valeurs manquantes (`NaN`). Traitement des codes spécifiques d'enquête (code `9` pour l'état civil, l'orientation sexuelle, les antécédents et le diagnostic ; code `99` pour l'âge et le nombre de partenaires). Suppression des identifiants non prédictifs (`ID`) et élimination des observations incomplètes (*Listwise Deletion*), portant l'échantillon final nettoyé à **2 664 observations**.
- **Ingénierie des caractéristiques :**
  - Dichotomisation de l'âge (moins de 30 ans vs 30 ans et plus).
  - Transformation des antécédents d'ITS en variable binaire.
  - Binning du nombre de partenaires sexuels mensuels selon la médiane (profils peu actifs vs très actifs).
  - Encodage catégoriel (*Dummy Encoding*) pour l'état civil et le motif de visite, avec omission de la catégorie de référence pour éviter le piège de la multicolinéarité parfaite.
- **Analyse exploratoire et redondance :** Détection d'un déséquilibre de classe majeur et élimination des variables explicatives redondantes (ex: corrélation parfaite entre la visite pour contact et l'existence d'un contact contaminé avéré).

### Prévention du Data Leakage et rééquilibrage
Mise en place d'un sous-échantillonnage (*RandomUnderSampler*) directement encapsulé au sein d'un `Pipeline` scikit-learn. Cette approche garantit que le rééquilibrage n'est appliqué que sur les plis d'entraînement de la validation croisée, préservant l'étanchéité absolue des plis de test (688 cas positifs et 688 cas négatifs par pli d'ajustement).

### Modélisation et sélection
- **Validation croisée :** Validation croisée stratifiée à 10 blocs (*Stratified 10-Fold Cross-Validation*).
- **Algorithmes évalués :** Régression logistique et Classifieur Naive Bayes de Bernoulli.
- **Sélection de variables :** Élimination pas à pas descendante (*Backward Stepwise Elimination*) basée sur la statistique de Wald au seuil de significativité $\alpha = 5\%$.

### Stack technologique
- **Langage :** Python 3
- **Traitement de données :** `pandas`, `numpy`
- **Modélisation & Apprentissage :** `scikit-learn`, `statsmodels`, `imblearn` (`Pipeline`, `RandomUnderSampler`)
- **Visualisation graphique :** `matplotlib`, `seaborn`

---

## 4. CONCLUSION ET RÉSULTATS CLÉS

### Performances comparatives
Évaluation des modèles en validation croisée à 10 blocs :

| Modèle | AUC | Exactitude (Accuracy) | Sensibilité (Recall) | Spécificité |
| :--- | :---: | :---: | :---: | :---: |
| **Régression Logistique** | **0,6478** | **0,6104** | **0,6090** | **0,6108** |
| **Naive Bayes de Bernoulli** | 0,6490 | 0,6040 | 0,6294 | 0,5951 |

### Modèle retenu et équation
La **Régression Logistique** a été retenue pour sa grande interprétabilité clinique. Le modèle final ne conserve que trois variables comportementales et démographiques, toutes hautement significatives ($p < 0,001$).

L'équation du score linéaire (log-odds $z$) s'établit comme suit :

$$z = -0,9589 + 0,7355 \cdot X_1 + 0,6258 \cdot X_2 + 0,4283 \cdot X_3$$

Où la probabilité $P$ d'un diagnostic positif est donnée par:

$$P = \frac{1}{1 + e^{-z}}$$ 

Avec :
- $X_1$ : Orientation homosexuelle (1 si Oui, 0 sinon)
- $X_2$ : Nombre élevé de partenaires (1 si supérieur à la médiane, 0 sinon)
- $X_3$ : Âge inférieur à 30 ans (1 si $< 30$ ans, 0 sinon)

### Interprétation des Odds Ratios (OR)

Estimation sur l'échantillon rééquilibré (1 376 observations) :

| Variable Explicative | Coefficient ($\beta$) | Erreur Type | Statistique $z$ | $p$-value | Odds Ratio (OR) | IC à 95% (OR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept (const)** | -0,9589 | 0,126 | -7,632 | $< 0,001$ | 0,3833 | [0,300 ; 0,490] |
| **Orientation homosexuelle** | 0,7355 | 0,122 | 6,026 | $< 0,001$ | **2,0865** | [1,643 ; 2,650] |
| **Partenaires (nombre élevé)** | 0,6258 | 0,122 | 5,148 | $< 0,001$ | **1,8698** | [1,473 ; 2,373] |
| **Âge moins de 30 ans** | 0,4283 | 0,117 | 3,654 | $< 0,001$ | **1,5346** | [1,220 ; 1,931] |

### Recommandations cliniques et opérationnelles
- **Orientation sexuelle (OR = 2,09) :** À caractéristiques égales, un patient homosexuel présente **2,09 fois plus de risques** d'être infecté qu'un patient hétérosexuel.
- **Activité sexuelle (OR = 1,87) :** Un nombre de partenaires supérieur à la médiane multiplie le risque par **1,87**.
- **Âge (OR = 1,53) :** Les patients de moins de 30 ans affichent un surrisque de **53 %** par rapport aux tranches plus âgées.
- **Synthèse :** Les variables structurelles (état civil, motifs secondaires) s'effacent statistiquement devant ces trois déterminants comportementaux. Un dépistage ciblé et proactif est recommandé en médecine de ville dès la présence combinée de deux de ces facteurs.

### Limites et perspectives
- Tester l'ajout d'interactions d'ordre supérieur entre l'âge et les comportements sexuels.
- Évaluer l'apport de modèles non linéaires ou d'arbres de décision pour capter des profils de risque plus complexes.

---

## 5. STRUCTURE DU DÉPÔT

```text
.
├── gono.csv                 # Jeu de données épidémiologique brut (3 144 observations)
├── images/
│   ├── AED.png              # Graphiques d'analyse exploratoire des données
│   ├── MAT.png              # Matrice de corrélation avec clustering
│   └── ROC.png              # Courbes ROC comparatives des modèles
├── projet_gono.ipynb      # Notebook Jupyter contenant le pipeline complet
└── README.md                # Documentation du projet
```

---
