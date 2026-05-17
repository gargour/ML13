.PHONY: setup train serve test pipeline

setup:
	pip install -r requirements.txt
	mlflow ui --host 0.0.0.0 --port 5000 &

train:
	python trainV2.py

serve:
	mlflow models serve -m "runs:/01d56b335a394a80b1d5cab3eea6df22/model" --port 1234 --no-conda &

test:
	python test_api.py

pipeline: train serve test
	@echo 'Pipeline complet exécuté avec succès et modèle déployé !'