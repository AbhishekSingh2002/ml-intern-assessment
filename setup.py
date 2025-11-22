from setuptools import setup, find_packages

setup(
    name="ml-assignment",
    version="0.1.0",
    packages=find_packages(where='ml-assignment'),
    package_dir={'': 'ml-assignment'},
    install_requires=[
        'numpy>=1.21.0',
        'requests>=2.28.0',
        'beautifulsoup4>=4.11.0',
        'tqdm>=4.64.0',
        'pytest>=7.0.0',
    ],
    python_requires='>=3.8',
    author="Your Name",
    description="A trigram language model for text generation",
    url="https://github.com/yourusername/trigram-lm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'trigram=ml_assignment.src.generate:main',
        ],
    },
    include_package_data=True,
    package_data={
        'ml_assignment': ['data/*.txt'],
    },
)
