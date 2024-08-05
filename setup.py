from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name="tiny-llm",
    version="0.1",
    packages=[*find_packages(where="src"), *find_packages(where="configurations")],
    package_dir={"": "src"},
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["train_model=training:main"],
    },
    author="Quentin Wendegass",
    python_requires=">=3.10",
)
