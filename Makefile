.PHONY: install seed-data run test lint all

install:
	pip install -r requirements.txt

seed-data:
	python -m src.generate_raw_data

run:
	python -m src.pipeline

test:
	pytest -v

lint:
	flake8 src tests --max-line-length=110

# the single documented command referenced in the README / report
all: install seed-data run test
