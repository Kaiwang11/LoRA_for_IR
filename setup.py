from setuptools import setup, find_packages

setup(
        name='sentence_transformers',
        version='0.1',
        packages=find_packages(),
        install_requires=[
                    # List your package dependencies here
                    # 'some_package>=1.0'i,
                    'peft',
                'transformers',
            'torch',
            'scikit-learn',
            'pillow',
            'nltk'
                ],
        entry_points={
                    'console_scripts': [
                                    # If you have any scripts to run, add them here
                                    # 'script_name = package.module:function',
                                ],
                },
)

