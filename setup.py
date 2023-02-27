from setuptools import setup
from pathlib import Path


about = {}
here = Path(__file__).parent.resolve()
with open(here / "src" / "__version__.py", "r") as f:
    exec(f.read(), about)

with open("README.md", "r") as arq:
    readme = arq.read()

with open(here / "src" / 'requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(name='projeto-mestrado',
    version=about["__version__"],
    license='MIT License',
    author='Vinícius Castro',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='vinicius.castro@estudante.ufscar.br',
    keywords='optuna, grid search',
    description=u'Algoritmo desenvolvido em decorrência do mestrado',
    packages=['src'],
    install_requires=install_requires,)