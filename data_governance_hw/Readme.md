
# Data governance HW
The model in this task is a simple classifier model using TD-IDF embeddings as an input.

DVC pipeline specified in dvc.yaml. At the end several files are generated:

    permutate_feature_importance_top_10.png
    metrics.json

Reproduce pipeline with this command:

    dvc repo

Link to the Google Drive:

    dvc remote add --default myremote gdrive://1Ri4X0YP4yG1WRMp0FD642VXVj3WOZuBb?usp=sharing 
    dvc pull 
Unit tests and DVC pipeline are specified in  **data_governance.yml**

Code quality Github Actions are specified in  **code_quality.yml**

GitHub actions are triggered on two operations:

    push
    pull_request

For quality code checking two libraries were used:

    black
    pylint (threshold = 5)

