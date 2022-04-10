from chat import getreply


# (unnecessary) layer for chat <=> streamlit to avoid imports
def returnreply(input):
    return getreply(input)