def modify(configs, default_factor=1.0, property_factors={}, config_type_exclude=[]):
    property_keys = ['energy', 'force', 'virial', 'hessian']

    for at in configs:
        if at.info.get('config_type', None) in config_type_exclude:
            continue
        for p in property_keys:
            psig = p + '_sigma'
            if psig in at.info:
                # save or restore original value
                if '_orig_' + psig in at.info:
                    # restore
                    at.info[psig] = at.info['_orig_' + psig]
                else:
                    # save
                    at.info['_orig_' + psig] = at.info[psig]

                # apply scale (to original value)
                at.info[psig] *= property_factors.get(p, default_factor)
