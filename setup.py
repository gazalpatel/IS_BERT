from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
    name="language_bert",
    version="0.2021.6.14",
    author="Gazal Patel",
    author_email="qgazal.patel@gmail.com",
    description="Sentence Embeddings using BERT / RoBERTa / XLNet",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/gazalpatel/Language_BERT",
    download_url="https://github.com/gazalpatel/Language_BERT/archive/refs/heads/main.zip",
    packages=find_packages(),
    install_requires=[
        'transformers>=3.1.0,<3.4.0',
        'tqdm',
        'torch>=1.2.0',
        'numpy',
        'scikit-learn',
        'scipy',
        'nltk'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformer Networks BERT XLNet sentence embedding PyTorch NLP deep learning"
)
