from setuptools import setup, find_packages
import os

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Define requirements
install_requires = [
    'numpy>=1.21.0',
    'tqdm>=4.64.0',
]

test_requires = [
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
    'pytest-mock>=3.10.0',
]

dev_requires = test_requires + [
    'black>=22.3.0',
    'flake8>=4.0.0',
    'isort>=5.10.0',
    'mypy>=0.950',
    'pylint>=2.13.0',
]

# Setup configuration
setup(
    name="trigram-lm",
    version="0.2.0",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
        'dev': dev_requires,
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A trigram language model for text generation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/trigram-lm",
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/trigram-lm/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'trigram=src.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['data/*.txt'],
    },
)
