from setuptools import setup

from tracker import __package__ as pkg_name

setup(
    name=pkg_name,
    version="0.1.0",
    description="Tracker Module",
    license="MIT",
    author="evercoinx",
    author_email="xyz@evercloud.io",
    url=f"https://github.com/evercoinx/{pkg_name}",
    packages=[pkg_name],
    install_requires=["numpy", "mss"],
    python_requires=">=3.7",
    entry_points={"console_scripts": [f"{pkg_name} = {pkg_name}.__main__:main"]},
)
