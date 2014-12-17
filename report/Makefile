SECTIONS = report.md
TEMPLATE = template.tex
OUTPUT = ./build/report.pdf

all: paper

.PHONY: all clean

pre:
	mkdir -p build

post:
	@echo POST

clean:
	rm -rf build

paper: $(SECTIONS) pre
	pandoc --toc -N -o $(OUTPUT) --template=$(TEMPLATE) $(SECTIONS)

default: paper
