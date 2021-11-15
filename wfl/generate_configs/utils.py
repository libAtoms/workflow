def config_type_append(at, config_type):
    if 'config_type' in at.info:
        at.info['config_type'] += '_' + config_type
    else:
        at.info['config_type'] = config_type
