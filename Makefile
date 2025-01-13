
excerpt-data-install:
	@echo "Installing mathwriting-2024-excerpt"
	sh utils/data-install.excerpt.sh

data-install:
	@echo "Installing mathwriting-2024"
	sh utils/data-install.sh

data-clean:
	@echo "Cleaning mathwriting-2024"
	rm -r data/
