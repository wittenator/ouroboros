# ouroboros

A framework for federated learning research.

## Installation

The project is built using the dependency management system [Poetry](https://python-poetry.org/docs/) and requires Python >= 3.8 to be installed. At the time of writing this documentation, Poetry can be installed using the following command:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
```

This will install Poetry to `$HOME/.local/bin`. In order to comfortably use Poetry from the command line, it has to be added to your path permanently. The easiest way to achieve this, is to add the following line to your `.bashrc` file (for Bash, for other shells like ZSH, please change the file accordingly).

```bash
export PATH="/home/<username>/.local/bin:$PATH"
```

After adding this line to your `.bashrc`, please restart your shell for the changes to take effect. When you are running an operating system where the default Python version is below 3.8 (e.g., Ubuntu 20.04), you have to explicitly tell Poetry to use the correct Python version. Under Ubuntu 20.04, Python 3.8 is already installed, but `python` and `python3` still link to Python 3.7. To use Python 3.8, `python3.8` has to be used. Iff Python 3.8 or newer is not installed on your system, please make sure to install it first, in the following command it is assumed that your Python command is `python3.8`, please adapt if necessary.

```bash
poetry env use python3.8
```

Finally, in order to be able to use Poetry, Python`s distribution utilities need to be installed. Under Debian-based systems (e.g. Ubuntu), this can be achieved using the following command:

```bash
sudo apt install python3-distutils
```

The project already comes pre-configured, so to install all necessary dependencies, execute `poetry install` in the root directory of the repository. It may happen, that you have to install further dependencies, because some dependencies are built from source. For example, under Ubuntu 20.04 it was necessary to install the headers for the `zlib`, `jpeg`, and `python3.8` packages to build Pillow from source:

```bash
sudo apt install zlib1g-dev libjpeg9-dev python3.8-dev
```

Then you can start any python script in the virtual env by prepending all commands with `poetry run` e.g. `poetry run python test.py`.
