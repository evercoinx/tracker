from setuptools import setup

from tracker import __package__ as pkg_name
from tracker import __version__

setup(
    name=pkg_name,
    version=__version__,
    description="Tracker Module",
    license="UNLICENSED",
    author="evercoinx",
    author_email="xyz@evercloud.io",
    url=f"https://github.com/evercoinx/{pkg_name}",
    packages=[pkg_name],
    install_requires=["mss", "numpy", "Pillow", "tesserocr", "typing_extensions"],
    python_requires=">=3.7",
    entry_points={"console_scripts": [f"{pkg_name} = {pkg_name}.__main__:main"]},
)
