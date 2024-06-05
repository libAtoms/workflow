from ase.atoms import Atoms


def save_config_type(at, action, config_type):
    """save a config type in one or more Atoms objects

    parameters:
    -----------
    at: Atoms:
        objects to store config type in
    action: "append" | "overwrite" | False
        action to perform on additional config type string
    config_type: str
        string to overwrite/append atoms.info["config_type"]
    """
    if not action:
        return

    if action not in ["append", "overwrite"]:
        raise ValueError(f"action {action} not 'append' or 'overwrite'")

    if action == 'append' and 'config_type' in at.info:
        at.info['config_type'] += ':' + config_type
    else:
        at.info['config_type'] = config_type
