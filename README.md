# 1. Contexte et objectif

Ce repo est dédié à la CI/CD du code d'entraînement de notre modèle **Auditeur de cohérence médicale**.

Le pipeline implémenté (GitHub Actions) exécute les tâches suivantes :
1. **Contrôler la qualité du code** dans `src/` et `tests/` avec l'outil `ruff`.
2. **Exécuter les scripts de test** dans `tests/`, pour tester notre `train.py` dans `src/`. Les tests sont exécutés dans un conteneur Docker (image construite au préalable dans le workflow).
3. **Déployer l'image Docker** : Si les tests sont passés avec succès, le code `train.py` est déployé dans une image Docker enregistrée sur Docker Hub.
4. **Déclencher Airflow** : Trigger un DAG sur [Airflow](https://github.com/Sam-MECHH/lead_final_project_airflow_local_server) qui va orchestrer l'entraînement de notre modèle sur le Cloud (EC2). Pour cela, le workflow exécutera `trigger_airflow.py` dans `scripts/`.

# 2. Code d'entraînement

Le code d'entraînement du modèle est constitué de trois parties :

1. **Preprocessing** : Extraction des *embeddings* en sortie de BioVil-T à partir des images et des rapports.
2. **Entraînement** : Entraînement des couches *cross-attention* + MLP pour pouvoir classifier les paires associées (*match* ou *mismatch*).
   - Early stopping implémenté.
   - Métriques et artifacts sauvegardés via MLflow.
   - Le meilleur modèle et ses performances sont également sauvegardés.
3. **Évaluation** :
   - Si la performance du modèle dépasse un certain seuil $\rightarrow$ Le modèle passe en **registered** dans MLflow.
   - Si le modèle est meilleur que celui actuellement en production $\rightarrow$ Passer le modèle en **production** + déclencher un *reload* dans l'API associée pour charger le nouveau modèle.
