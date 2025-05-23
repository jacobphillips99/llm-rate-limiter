from setuptools import find_packages, setup

setup(
    name="llm-rate-limiter",
    version="0.1.0",
    description="A rate limiter for LLM API calls with built-in monitoring",
    author="Jacob Phillips",
    author_email="jacob.phillips8905@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
