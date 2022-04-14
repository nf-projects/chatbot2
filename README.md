# chatbot2
A chatbot in Python using a PyTorch Neural Network

# Setup
1. Install miniconda
2. Select miniconda python environment in vscode
3. Open miniconda command terminal and cd into the project folder
4. create new miniconda env `conda create --name chatbot2`
5. `conda activate chatbot2`
6. `conda install python`
7. `pip install streamlit`
8. `pip3 install torch torchvision`
9. `pip install nltk`
10. Install the conda dependencies (numpy, nltk, streamlit-chat)
IF PIP IS INSTALLING BUT IT'S NOT RECOGNIZING THE MODULE:
- Instead of `pip install [...]` do `py -3 -m pip install [...]`

- to train, run `py train.py`
- to run the streamlit webapp, run `streamlit run streamlit.py`