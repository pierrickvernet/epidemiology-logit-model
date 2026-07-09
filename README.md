# Modèle de Prédiction et de Caractérisation des Groupes à Risque de la Gonorrhée (Projet GONO)

## 📌 Présentation du Projet
Le dépistage des Maladies Transmises Sexuellement (MTS) représente un défi stratégique majeur en santé publique. Face à la réticence de certaines populations à se rendre dans des cliniques spécialisées, ce projet s'inscrit dans le cadre d'un **programme de formation destiné aux médecins de famille**. En leur permettant d'utiliser gratuitement les laboratoires des grands centres hospitaliers, ce programme vise à intégrer le dépistage directement au sein des consultations de pratique privée.

L'objectif principal de ce projet est de **définir et caractériser précisément les groupes cibles à haut risque** pour la gonorrhée. En identifiant les facteurs de risque discriminants, l'étude permet de cibler efficacement les milieux géographiques et sociaux où former les médecins généralistes.

La gonorrhée a été choisie comme indicateur clé car elle constitue la seule MTS commune dépistée efficacement par un simple test de laboratoire sur culture.

---

## 📊 Données et Prétraitement
Le projet s'appuie sur le fichier de données anonymisées `gono.csv`, qui recense les patients examinés dans le cadre du programme. 

### Nettoyage et Gestion des Valeurs Manquantes
Le script intègre un protocole strict de nettoyage pour traiter les données manquantes (codées spécifiquement selon le protocole de l'enquête) :
* **Blancs :** Convertis systématiquement en valeurs manquantes (`NaN`).
* **Code `9` :** Considéré comme manquant pour les variables d'état civil, d'orientation sexuelle, d'antécédents et de diagnostic.
* **Code `99` :** Considéré comme manquant pour l'âge et le nombre de partenaires.
* **Élagage :** Suppression de la colonne `ID` (non prédictive) et élimination des lignes incomplètes par `dropna()`.

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
Le fort déséquilibre initial de la variable cible nécessite un rééquilibrage par sous-échantillonnage (**Undersampling**). Afin d'éviter tout phénomène de *Data Leakage* (fuite d'information qui biaiserait les métriques d'évaluation), l'algorithme `RandomUnderSampler` est encapsulé directement au sein d'un `Pipeline` de calcul. Ainsi, **le rééquilibrage n'est appliqué que sur les plis d'entraînement** de la validation croisée, laissant les plis de test totalement inchangés.

---

## 📈 Résultats et Sélection Finale

Les performances intrinsèques des deux modèles sont évaluées et comparées à l'aide de courbes ROC, de l'Aire Sous la Courbe (AUC) ainsi que de la sensibilité (Recall) — métrique cruciale en épidémiologie où l'on cherche à minimiser à tout prix les faux négatifs.

**Le choix final s'est porté sur la Régression Logistique** pour deux raisons :
* **Performances équivalentes :** Les deux modèles affichent une capacité prédictive similaire ($AUC \approx 0.65$). La régression logistique permet d'ajuster finement le seuil de décision pour maximiser la sensibilité clinique si nécessaire.
* **Interprétabilité clinique majeure :** Contrairement au Naive Bayes ou aux modèles "boîte noire", la régression logistique permet d'extraire les **Odds Ratios (OR)** et leurs intervalles de confiance à 95 %. Cela permet de quantifier précisément l'impact de chaque facteur sur l'augmentation du risque.

---

## 🚀 Recommandations Cliniques (Synthèse)
L'analyse finale des coefficients du modèle permet d'isoler le profil typique sur lequel le médecin de famille doit appliquer une vigilance maximale ("Alerte Rouge") :
* **Homme**
* **Moins de 30 ans**
* **Orientation homosexuelle**
* **Nombre de partenaires élevé (supérieur à la médiane)**

Le modèle démontre que le personnel soignant ne doit pas se laisser rassurer par des facteurs comme un statut marié ou une visite dite "asymptomatique" si les critères majeurs ci-dessus sont réunis. Un dépistage proactif est fortement recommandé dès la présence de deux de ces facteurs.

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



