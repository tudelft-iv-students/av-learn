"""This module implements methods used for the nuScenes map representation"""


def load_table(data: list) -> dict:
    """Load a nuScenes table.

    This method takes a list of nested dictionaries and converts it to a
    dictionary using the inner tokens as keys.

    :param data: A list of dictionaries.
    """
    table = {}
    for record in data:
        table[record["token"]] = record

    return table
