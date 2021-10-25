from setuptools import setup

pkg_name = "tracker"

setup(
    name=pkg_name,
    version="0.1.0",
    description="Tracker Module",
    license="MIT",
    author="Serge Grigorenko",
    author_email="serge@evercloud.io",
    url=f"https://github.com/piwatch/{pkg_name}",
    packages=[pkg_name],
    install_requires=["numpy", "mss"],
    python_requires=">=3.7",
    entry_points={"console_scripts": [f"{pkg_name} = {pkg_name}.__main__:main"]},
)
