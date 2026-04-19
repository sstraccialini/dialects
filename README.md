# Modeling Historical and Linguistic Relations Between Italian Dialects and Contemporary Languages Through Embedding Spaces

This repository contains the code and data used for the project of 20879 - Language Technology, Spring 2026 course at Bocconi University. The project focuses on modeling historical and linguistic relations between Italian dialects and contemporary languages through embedding spaces.

## How to run


### Environment setup
For this project, we used Python 3.10. To set up the environment, you can follow these steps:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
python -m spacy download it_core_news_sm

python .\data\create.py

```

### Data extraction
To extract the data, you can insert the links to the dumps in the `dumps` variable in `data/create.py` and run the script (`python .\data\create.py`). The script will download the dumps, extract the text, and save it in the specified output directories.:


### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.