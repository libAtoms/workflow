class Params:
    implemented_calculators = ("CASTEP", "VASP", "ORCA")

    def __init__(self, params_dict, cur_iter=None):
        self.d = params_dict
        self.__cur_iter = cur_iter

        self._dft_code = None
        self._dft_params = dict()

    @property
    def cur_iter(self):
        return self.__cur_iter

    @cur_iter.setter
    def cur_iter(self, value):
        self.__cur_iter = value

    @property
    def dft_code(self):
        if self._dft_code is None:
            self._read_dft_params()
        return self._dft_code

    @property
    def dft_params(self):
        if self._dft_code is None:
            self._read_dft_params()
        return self._dft_params

    def _read_dft_params(self):
        calc = self.get("DFT_evaluate/calculator")
        if calc is None:
            return

        # decide which DFT calculator
        if calc in self.implemented_calculators:
            self._dft_code = calc
        else:
            raise NotImplementedError("DFT calculator not implemented yet:", calc)

        # read settings
        for key, val in self.get("DFT_evaluate").items():
            if key == "DFT_calculator":
                continue
            self._dft_params[key] = val

    def get(self, item_path, default=None):
        if item_path.startswith('/'):
            item_path = item_path.replace('/', '', 1)
        path_comps = item_path.split('/', 1)
        leading_comp = path_comps[0]

        if leading_comp in self.d:
            result = self.d[leading_comp]

        if self.cur_iter is not None and 'iter_specific' in self.d and leading_comp in self.d['iter_specific']:
            for range_spec, val in self.d['iter_specific'][leading_comp].items():
                range_components = range_spec.split(':')
                range_components = [None if len(c) == 0 else int(c) for c in range_components]
                if len(range_components) == 1:
                    # just an integer
                    if range_components[0] == self.cur_iter:
                        result = val
                elif len(range_components) == 2 or len(range_components) == 3:
                    # actual range, optional step
                    if ((range_components[0] is None or self.cur_iter >= range_components[0]) and
                            (range_components[1] is None or self.cur_iter < range_components[1])):
                        rel_iter = self.cur_iter - (range_components[0] if range_components[0] is not None else 0)
                        if len(range_components) == 2 or rel_iter % range_components[2] == 0:
                            result = val
                else:
                    raise RuntimeError('Uninterpretable number of : in range spec \'{}\''.format(range_spec))

        if len(path_comps) > 1:
            # not final item, recurse
            try:
                result
            except NameError as exc:
                raise ValueError('Failed to find path component \'{}\''.format(leading_comp)) from exc
            result = Params(result, cur_iter=self.cur_iter).get(path_comps[1], default=default)
        else:
            # final item return (or default)
            try:
                result
            except NameError:
                result = default

        return result
