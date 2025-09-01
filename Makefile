# Makefile (activated conda env)
.PHONY: train ui clean

train:
	python -m src.train --config configs/config.yaml

ui:
	streamlit run ui/app.py

clean:
	rm -rf __pycache__ .pytest_cache models reports
