# Modèle de Prédiction et de Caractérisation des Groupes à Risque de la Gonorrhée (Projet GONO)

## 📌 Présentation du Projet
Le dépistage des Maladies Transmises Sexuellement (MTS) représente un défi stratégique majeur en santé publique. Face à la réticence de certaines populations à se rendre dans des cliniques spécialisées, ce projet s'inscrit dans le cadre d'un **programme de formation destiné aux médecins de famille**. En leur permettant d'utiliser gratuitement les laboratoires des grands centres hospitaliers, ce programme vise à intégrer le dépistage directement au sein des consultations de pratique privée.

L'objectif principal de ce projet est de **définir et caractériser précisément les groupes cibles à haut risque** pour la gonorrhée. En identifiant les facteurs de risque discriminants, l'étude permet de cibler efficacement les milieux géographiques et sociaux où former les médecins généralistes.

La gonorrhée a été choisie comme indicateur clé car elle constitue la seule MTS commune dépistée efficacement par un simple test de laboratoire sur culture.

---

## 📊 Données et Prétraitement
Le projet s'appuie sur le fichier de données anonymisées `gono.csv`, qui recense initialement 3 144 patients examinés dans le cadre du programme.

### Nettoyage et Gestion des Valeurs Manquantes
Le script intègre un protocole strict de nettoyage pour traiter les données manquantes (codées spécifiquement selon le protocole de l'enquête) :
* **Blancs :** Convertis systématiquement en valeurs manquantes (`NaN`).
* **Code `9` :** Considéré comme manquant pour les variables d'état civil, d'orientation sexuelle, d'antécédents et de diagnostic.
* **Code `99` :** Considéré comme manquant pour l'âge et le nombre de partenaires.
* **Élagage :** Suppression de la colonne `ID` (non prédictive) et élimination des lignes incomplètes par `dropna()`. L'échantillon final nettoyé comprend **2 664 observations**.

### Ingénierie des Variables (Feature Engineering)
Conformément à la littérature épidémiologique et aux exigences cliniques, les transformations suivantes ont été appliquées :
* **Dichotomisation de l'âge :** Séparation en deux groupes (Moins de 30 ans / 30 ans et plus).
* **Antécédents de MTS :** Transformation en variable binaire (Déjà contracté / Jamais contracté).
* **Activité sexuelle :** Dichotomisation du nombre de partenaires au cours du dernier mois selon la médiane pour séparer les profils "peu actifs" et "très actifs".
* **Encodage catégoriel :** Création de variables *dummies* pour l'état civil et la raison de la visite (avec exclusion de la première modalité pour éviter le piège de la multicolinéarité parfaite).

---

## 🧠 Approche Méthodologique & Choix des Modèles

Une Analyse Exploratoire des Données (AED) par **clustering hiérarchique (dendrogramme)** a mis en évidence un fort déséquilibre des classes (les diagnostics positifs étant minoritaires) ainsi que des redondances évidentes (corrélation parfaite entre la raison de la visite pour contact et l'existence d'un contact contaminé). Les variables redondantes ont été écartées avant la modélisation.

Pour répondre au besoin de caractérisation, deux modèles adaptés aux features binaires ont été mis en compétition via une **validation croisée stratifiée à 10 plis (Stratified 10-Fold CV)** :
1. **Régression Logistique**
2. **Classifieur Naive Bayes de Bernoulli**

### ⚠️ Protection contre le Data Leakage
Le fort déséquilibre initial de la variable cible nécessite un rééquilibrage par sous-échantillonnage (**Undersampling**). Afin d'éviter tout phénomène de *Data Leakage* (fuite d'information qui biaiserait les métriques d'évaluation), l'algorithme `RandomUnderSampler` est encapsulé directement au sein d'un `Pipeline` de calcul. Ainsi, **le rééquilibrage n'est appliqué que sur les plis d'entraînement** de la validation croisée, laissant les plis de test totalement inchangés (688 cas positifs vs 688 cas négatifs par pli d'ajustement).

---

## 📈 Résultats et Sélection Finale

### Comparaison des Modèles (Validation Croisée 10-Fold)
Les performances intrinsèques des deux modèles affichent une capacité prédictive similaire ($AUC \approx 0.65$) :

| Modèle | AUC | Exactitude (Accuracy) | Sensibilité (Recall) | Spécificité |
| :--- | :---: | :---: | :---: | :---: |
| **Régression Logistique** | 0.6478 | 0.6104 | 0.6090 | 0.6108 |
| **Naive Bayes** | 0.6490 | 0.6040 | 0.6294 | 0.5951 |

Le choix final s'est porté sur la **Régression Logistique** car elle offre une **interprétabilité clinique majeure**. Contrairement au Naive Bayes ou aux modèles "boîte noire", elle permet d'extraire l'impact mathématique direct de chaque variable sous forme d'Odds Ratios.

### Équation du Modèle Final
Suite à une sélection descendante (*Backward Stepwise Elimination* au seuil $\alpha = 5\%$), le modèle logistique final retient 3 variables hautement significatives. 

La probabilité $P$ d'obtenir un diagnostic positif est définie par la fonction sigmoïde :

$$P = \frac{1}{1 + e^{-z}}$$

Où le score linéaire (ou log-odds) $z$ est calculé à partir des coefficients estimés :

$$z = -0.9589 + 0.7355 \cdot X_1 + 0.6258 \cdot X_2 + 0.4283 \cdot X_3$$

Avec les variables indicatrices suivantes :
* $X_1$ : ORIENTATION_HOMOSEXUELLE (1 si Oui, 0 sinon)
* $X_2$ : PARTENAIRES_NOMBRE_ELEVE (1 si Supérieur à la médiane, 0 sinon)
* $X_3$ : AGE_MOINS_30 (1 si Moins de 30 ans, 0 si 30 ans et plus)

### Coefficients Statistiques et Odds Ratios (OR)

Voici les résultats détaillés de l'estimation sur l'échantillon balancé (1 376 observations) :

| Variable Explicative | Coefficient ($\beta$) | Erreur Type | Statistique $z$ | $P > \|z\|$ | Odds Ratio (OR) | IC à 95% (OR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept (const)** | -0.9589 | 0.126 | -7.632 | < 0.001 | 0.3833 | [0.300 ; 0.490] |
| **ORIENTATION_HOMOSEXUELLE** | 0.7355 | 0.122 | 6.026 | < 0.001 | 2.0865 | [1.643 ; 2.650] |
| **PARTENAIRES_NOMBRE_ELEVE** | 0.6258 | 0.122 | 5.148 | < 0.001 | 1.8698 | [1.473 ; 2.373] |
| **AGE_MOINS_30** | 0.4283 | 0.117 | 3.654 | < 0.001 | 1.5346 | [1.220 ; 1.931] |

---

## 🚀 Recommandations Cliniques (Synthèse)
Toutes les variables retenues sont extrêmement significatives ($p < 0.001$). L'analyse des Odds Ratios permet d'isoler le profil typique sur lequel le médecin de famille doit appliquer une vigilance maximale :
* **Orientation Homosexuelle (OR = 2.09) :** À caractéristiques égales, un patient homosexuel a **2,09 fois plus de chances** de présenter un diagnostic positif qu'un patient hétérosexuel.
* **Nombre de Partenaires Élevé (OR = 1.87) :** Un patient ayant une activité sexuelle supérieure à la médiane a **1,87 fois plus de chances** d'être contaminé.
* **Âge Moins de 30 Ans (OR = 1.53) :** Les moins de 30 ans présentent un risque accru de **53%** par rapport aux patients plus âgés.

Le modèle démontre mathématiquement que les facteurs de structure (état civil, motifs de visite secondaires) s'effacent statistiquement devant ces trois critères comportementaux majeurs. Un dépistage proactif est fortement recommandé dès la présence de deux de ces facteurs.

---

## 🛠️ Installation et Utilisation

### Prérequis
Assurez-vous d'avoir Python 3.8+ installé. Les dépendances nécessaires sont listées dans le fichier `requirements.txt`.

```bash
pip install -r requirements.txt

Exécution du projet
Placez votre fichier de données sous le nom gono.csv dans le même répertoire que le script.
Lancez le script principal pour générer les analyses graphiques, les rapports de validation croisée et la table finale des Odds Ratios :

python gono_analysis.py

Dépendances principales
pandas & numpy : Nettoyage et manipulation des données
matplotlib & seaborn : Visualisation (AED, Courbes ROC)
scikit-learn : Validation croisée et métriques d'évaluation
imbalanced-learn : Gestion du déséquilibre des classes sans fuite de données
statsmodels : Estimation fine de la régression logistique et calcul des p-values



