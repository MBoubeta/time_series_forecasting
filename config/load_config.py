import json


def load_config(json_file: json):
    """
    Load the information of the .json file
    :param json_file: json configuration file
    :return: data containing the information of .json file
    """

    with open('config.json') as config_file:
        data = json.load(config_file)

    return data
