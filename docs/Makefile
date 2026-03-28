.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest coverage gettext

SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make single-file HTML documentation"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML help files"
	@echo "  qthelp     to make Qt help files"
	@echo "  devhelp    to make devhelp files"
	@echo "  epub       to make EPUB files"
	@echo "  latex      to make LaTeX files"
	@echo "  latexpdf   to make PDF files from LaTeX"
	@echo "  latexps    to make PS files from LaTeX"
	@echo "  text       to make text files"
	@echo "  man        to make manual page files"
	@echo "  texinfo    to make Texinfo files"
	@echo "  gettext    to make PO message catalog files"
	@echo "  changes    to make an overview of all changed/updated files"
	@echo "  linkcheck  to make a link check of all cross-references"
	@echo "  doctest    to make doctest files"
	@echo "  coverage   to make a coverage report"

clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML documentation is in $(BUILDDIR)/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML documentation is in $(BUILDDIR)/dirhtml."

singlehtml:
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML documentation is in $(BUILDDIR)/singlehtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "The pickle files are in $(BUILDDIR)/pickle."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "The JSON files is in $(BUILDDIR)/json."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished. The HTML documentation is in $(BUILDDIR)/html."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished. The HTML documentation is in $(BUILDDIR)/qthelp."

devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished. The HTML documentation is in $(BUILDDIR)/devhelp."

epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The HTML documentation is in $(BUILDDIR)/singlehtml."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished. The LaTeX files are in $(BUILDDIR)/latex."

latexpdf:
	$(SPHINXBUILD) -b latexpdf $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished. The PDF files are in $(BUILDDIR)/latex."

latexps:
	$(SPHINXYMLBUILD) -b latexps $(ALLSPHINXOPTS) $(BUILDDIR)/latexps
	@echo
	@echo "Build finished. the PS files are in $(BUILDDIR)/latexps."

text:
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "The text files are in $(BUILDDIR)/text."

man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "The man files are in $(BUILDDIR)/man."

texinfo:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in $(BUILDDIR)/texinfo."

gettext:
	$(SPHINXBUILD) -b gettext $(ALLSPHINXOPTS) $(BUILDDIR)/gettext
	@echo
	@echo "The gettext files are in $(BUILDDIR)/gettext."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The changes API changes is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; check results in $(BUILDDIR)/linkcheck/broken.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo
	@echo "Build finished. The doctest files are in $(BUILDDIR)/doctest."

coverage:
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) $(BUILDDIR)/coverage
	@echo
	@echo "Build finished. The doctest files are in $(BUILDDIR)/coverage."
