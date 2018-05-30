pdf: report.tex
	latexmk -pdf -pv -pdflatex="pdflatex -interaction=nonstopmode -halt-on-error -quiet --shell-escape %O %S" report.tex

watch:
	latexmk -pdf -pvc -pdflatex="pdflatex -interaction=nonstopmode -quiet --shell-escape %O %S" report.tex

clean:
	latexmk -CA
