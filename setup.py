from setuptools import setup, find_packages

setup(
    name="car_price_prediction",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask",
        "flask-cors",
        "pandas",
        "numpy",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)
