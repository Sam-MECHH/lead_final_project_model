1. Contexte et objectif
Ce repo est dédié à la CI/CD du code d'entrainement de notre modèle Auditeur de cohérence médicale.

Le pipeline implémenté (GitHub Actions) exécute les taches suivantes:
   1- Contrôler la qualité du code dans src/ et tests/ avec l'outil ruff.
   2- Exécuter les scripts de test dans test/, pour tester notre train.py dans src/. Les tests sont exécutés dans un conteneur Docker. Image construite au préalable dans le workflow.
   3- Si les tests sont passés avec succès, le code train.py est déployé dans une image Docker enregistrée sur Docker Hub.
   4- Trigger un dag sur Airflow (https://github.com/Sam-MECHH/lead_final_project_airflow_local_server) qui va orchestré l'entrainement de notre modèle sur Cloud (EC2). Pour cela, le wrokflow exécutera trigger_airflow.py dans scripts/.

2. Code d'entrainement
Le code d'entrainement du modèle est constitué de trois parties:
   1- Preprocessing: extraction à partir des images et rapprots les embeddings en sortie du BioVil-t
   2- Entrainement: entrainement des couches cross-attention+MLP pour pouvoir classifier les paires associées (match ou mismatch).
        - Early stopping implémenté.
        - Metrics et artefacts sont sauvegardées via MLFlow.
        - Le meilleur modèle et ses performances sont aussi sauvegardées.
   3- Evaluation:
        -  Si la performance du modèle dépasse un certain seuil --> Le modèle passe en "registered" dans MLflow
         - Si le modèle est meilleur que celui actuellement en production: passer le modèle en production + déclencher un reload dans l'API associée pour charger le nouveau modèle. 
