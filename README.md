# Jlyph-BackEnd
This Back end
天才又美丽的冯捷琳女士的Jlyph工作

## Project Init
- Python 3.10.11
- pip 23.2.1 (python 3.10)

Build the environment for this project
```bash
cd Jlyph-BackEnd/ # enter project folder
python3.10 -m pip install virtualenv # install virtual environment
virtualenv env # create virtual environment
source env/bin/activate # enter virtual environment
deactivate # exit virtual environment
```

Install packages in the environment, and all package files are stored under env
```bash
source env/bin/activate
python3.10 -m pip install -U -r requirements.txt
```

update the requirements before exit
```bash
python3.10 -m pip check # check for package conflicts
python3.10 -m pip freeze > requirements.txt
```

Start the Flask app
```bash
python3.10 app.py
# or
python3.10 -m flask --app app run
```