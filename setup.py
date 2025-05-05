from setuptools import setup, find_packages

setup(
    name="multimodal_emotion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'transformers>=4.11.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'opencv-python>=4.5.0',
        'scikit-learn>=0.24.0',
        'jupyter>=1.0.0',
        'notebook>=6.0.0',
        'Pillow>=8.0.0',
        'tqdm>=4.62.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Multimodal Emotion Recognition using Text and Images",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-emotion-recognition",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
