import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PassivePy",
    version="0.0.1",
    author="Mitra Mirshafiee",
    author_email="mitra.mirshafiee@gmail.com",
    description="A package for processing large corpora and detecting passive voice.",
    long_description="Our aim with this work is to create a reliable (e.g., passive voice judgments are consistent), valid (e.g., passive voice judgments are accurate), flexible (e.g., texts can be assessed at different units of analysis), replicable (e.g., the approach can be performed by a range of research teams with varying levels of computational expertise), and scalable way (e.g., small and large collections of texts can be analyzed) to capture passive voice from different corpora for social and psychological evaluations of text. To achieve these aims, we introduce PassivePy, a fully transparent and documented Python library.",
    long_description_content_type="markdown",
    url="https://github.com/mitramir55/PassivePy",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        'Programming Language :: Python :: 3', 

        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)