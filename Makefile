# Description: Makefile for the retrieval service
.PHONY: init

init:
	mkdir -p chroma.db
	curl -L https://github.com/etendosoftware/com.etendoerp.copilot.retrieval/releases/download/v1.0.0/chroma.sqlite3 -o chroma.db/chroma.sqlite3
