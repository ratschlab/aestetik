pip uninstall -y aestetik
rm -rf dist/ aestetik.egg-info/ build/
python setup.py install --user
