# ouroboros

A Python framework for federated learning research.

## Installation

The project is built using the dependency management system [Poetry](https://python-poetry.org/docs/) and requires Python >= 3.8 to be installed. Before Poetry can be installed, some dependencies are required that need to be installed first. This includes the Python distribution tools, as well as virtual environments. Under Debian-based systems (e.g., Ubuntu), this can be achieved using the following command:

```bash
sudo apt update
sudo apt install python3-distutils python3-virtualenv
```

At the time of writing this documentation, Poetry can be installed using the following command. Please make sure that you specify the exact Python path to make sure that Poetry uses the correct Python interpreter. On their website they tell you to use `python`, but depending on your system, this may not be Python 3.8 or be a version of Python that was installed using Anaconda.

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | /usr/bin/python3.8 -
```

This will install Poetry to `$HOME/.local/bin`. In order to comfortably use Poetry from the command line, it has to be added to your path permanently. The easiest way to achieve this, is to add the following line to your `.bashrc` file (for Bash, for other shells like ZSH, please change the file accordingly).

```bash
export PATH="/home/<username>/.local/bin:$PATH"
```

After adding this line to your `.bashrc`, please restart your shell for the changes to take effect. When you are running an operating system where the default Python version is below 3.8, you have to explicitly tell Poetry to use the correct Python version. On some operating systems Python 3.8 is not installed while on others it may be installed, but `python` and `python3` may still link to an older version. To use Python 3.8, please install if it necessary and run the following command to tell Poetry to use a different Python interpreter (it is assumed that your Python command is `python3.8`, please adapt if necessary).

```bash
poetry env use python3.8
```

The project already comes pre-configured, so to install all necessary dependencies, execute `poetry install` in the root directory of the repository. It may happen, that you have to install further dependencies, because some dependencies are built from source. For example, under Ubuntu 20.04 it was necessary to install the headers for the `zlib`, `jpeg`, and `python3.8` packages to build Pillow from source:

```bash
sudo apt install zlib1g-dev libjpeg9-dev python3.8-dev
```

Now you can start any Python script in the virtual environment provided by Poetry, by pre-pending all commands with `poetry run`, e.g., `poetry run python test.py`.
