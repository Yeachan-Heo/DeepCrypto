from setuptools import setup

setup(
    name='DeepCrypto',
    version='',
    packages=['deepcrypto', 'deepcrypto.backtest', 'deepcrypto.data_utils', 'deepcrypto.data_utils.crawlers',
              'deepcrypto.portfolio_analysis', 'deepcrypto.ui.cli', 'deepcrypto.ui.gui'],
    url='',
    license='',
    author='ych',
    author_email='',
    description='',
    entry_points={
        'console_scripts' : ['backtest=deepcrypto.ui.cli:main']
    }
)
