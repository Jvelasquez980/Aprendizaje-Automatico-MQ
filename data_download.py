import kaggle

kaggle.api.authenticate() #need to have kaggle.json file in ~/.kaggle/ directory

kaggle.api.dataset_download_files('mohammadamireshraghi/blood-cell-cancer-all-4class', path='data/', unzip=True)