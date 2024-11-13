from setuptools import setup, find_packages

setup(
    name='stochCalc',                    
    version='0.1.0',                        
    author='Diljit Singh',                  
    author_email='diljitsingh22 @Googles email service',
    description='Stochastic calculus related functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Singh-Diljit/stochCalc',
    packages=find_packages(),                
    classifiers=[                           
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',                
    install_requires=[
        'sympy>=1.13.0'
        ],
    )
