Setup
=====

With visual studio code (vscode)
================================

- Install the `Visual Studio Code Remote - Containers` extension
- Open the base directory "in Container" --- this will build a dev environment docker container, install all the needed vscode extensions and python requirements, and connect to it
- Start jupyter from an integrated shell with `scripts/start_jupyter.sh`; the token is "the_token"

Without vscode
===============

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then start a jupyter notebook:
```
jupyter notebook
```
(though note that the notebook currently assumes that you are running inside a vscode dev container)