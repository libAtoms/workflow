import numpy as np
from ase.atoms import Atoms
from pytest import raises

from wfl.configset import ConfigSet, OutputSpec
from wfl.select.by_descriptor import greedy_fps_conf_global, prep_descs_and_exclude, write_selected_and_clean

ref_array = np.array(
    [[8.76710127e-02, 1.16867105e-40, 1.06347842e-09, 2.85176889e-40, 9.83879615e-09, 1.57131721e-01, 1.05046539e-37,
      1.67231833e-06, 2.71987172e-38, 1.54739218e-05, 1.40812664e-01, 7.60618189e-35, 1.31485924e-03, 6.35356267e-34,
      1.21682702e-02, 3.67661530e-01, -4.32952228e-37, 4.19117235e-07, -5.27110393e-38, 1.10996897e-05, 4.65951847e-01,
      -1.06073661e-34, 4.66521816e-04, 6.63239787e-35, 1.23448787e-02, 7.70921863e-01, 2.46489482e-33, 1.12373323e-03,
      5.69261234e-34, 8.20647160e-03, 8.46663111e-03, 3.00914565e-38, -7.99370621e-08, 2.81128031e-39, -3.63753868e-07,
      1.07300930e-02, 2.83859092e-35, -8.88698075e-05, -5.06025978e-36, -4.04506031e-04, 2.51065945e-02,
      -4.13299771e-35, 1.98524236e-05, -8.40511250e-37, -2.17995841e-04, 4.08822941e-04, 5.36047649e-36, 1.88356614e-05,
      4.98765449e-37, 1.96176843e-05, 0.00000000e+00],
     [1.17538362e-01, 7.77080382e-41, 3.35667444e-09, 3.17773911e-40, 3.51580517e-09, 1.35287674e-01, 2.36872849e-37,
      5.27767085e-06, -6.73442604e-38, 5.52750845e-06, 7.78586431e-02, 4.78408271e-34, 4.14901867e-03, 5.10844748e-35,
      4.34514356e-03, 4.34740265e-01, 1.05319344e-37, -7.43954445e-07, -7.69648690e-38, -1.85663656e-06, 3.53829032e-01,
      7.34864498e-34, -8.26913798e-04, -1.31353376e-34, -2.05998643e-03, 8.03988992e-01, 1.19177404e-33, 4.99623906e-04,
      7.36080841e-34, 8.99804198e-03, 1.14000969e-02, -1.02946597e-38, -1.33742639e-07, 1.47816643e-38, -1.85717994e-07,
      9.27837971e-03, -1.48524768e-35, -1.48700620e-04, 1.12881808e-36, -2.06480381e-04, 2.98156154e-02, 1.19133323e-35,
      -4.29198831e-06, -2.94518899e-35, 1.96030715e-05, 5.52850182e-04, 9.21374788e-37, 2.46129649e-05, 1.18935578e-36,
      2.59922839e-05, 0.00000000e+00],
     [8.52428361e-02, 2.71640037e-41, 1.32940902e-08, 1.78144891e-40, 8.31246247e-09, 1.27491351e-01, 5.70918531e-38,
      2.09023841e-05, -1.14576085e-37, 1.30662924e-05, 9.53396512e-02, 1.43173811e-34, 1.64324769e-02, 8.79859202e-35,
      1.02694003e-02, 3.73551096e-01, 1.39248422e-38, -2.35584839e-06, 3.56433609e-37, -1.17214263e-05, 3.95055260e-01,
      -2.22440999e-34, -2.61911821e-03, -8.19706140e-35, -1.30250160e-02, 8.18487673e-01, 4.19438771e-34,
      3.91918436e-04, 1.26666516e-33, 1.51940009e-02, 9.57720793e-03, -5.50466419e-39, 5.27508123e-07, 1.03747334e-38,
      -2.10800520e-07, 1.01285377e-02, -3.56475961e-37, 5.86481705e-04, 1.03557181e-36, -2.34270962e-04, 2.96767298e-02,
      -1.28331346e-35, -5.35944871e-05, 4.58723673e-35, 3.07883090e-04, 5.38009503e-04, 1.16088471e-36, 2.20994366e-05,
      1.05229652e-36, 1.63656676e-05, 0.00000000e+00],
     [8.35552100e-02, 2.66597411e-40, 1.00484384e-09, 5.39587442e-40, 7.60639382e-09, 1.65885712e-01, 3.80712148e-37,
      1.58025272e-06, 1.62133649e-37, 1.19624864e-05, 1.64669979e-01, 2.74209064e-34, 1.24258059e-03, 1.27141565e-34,
      9.40663143e-03, 3.54313072e-01, 3.43177148e-37, 8.04016458e-07, 2.14720768e-37, 7.28804791e-06, 4.97402110e-01,
      2.74541718e-34, 8.94576296e-04, -4.72633060e-35, 8.10642117e-03, 7.51226363e-01, 1.33561654e-33, 1.35864383e-03,
      4.70782056e-34, 7.03519931e-03, 8.61757386e-03, 8.12699788e-39, 1.29298506e-07, 6.99182231e-39, -3.61246132e-07,
      1.20977738e-02, 9.34085299e-36, 1.43800488e-04, 1.13241586e-36, -4.01691073e-04, 2.58394733e-02, -1.88747431e-35,
      1.26210994e-04, 3.81925705e-36, -1.38358963e-04, 4.44392272e-04, 4.77861049e-37, 2.35489795e-05, 3.48514969e-37,
      1.76788141e-05, 0.00000000e+00],
     [8.44944493e-02, 9.40043862e-41, 5.36820785e-09, 2.58421251e-40, 5.80969884e-09, 1.44169259e-01, -1.13444077e-37,
      8.44197807e-06, -3.57291882e-37, 9.13754141e-06, 1.22994914e-01, 1.56583733e-34, 6.63787590e-03, 5.18527219e-34,
      7.18579981e-03, 3.65803746e-01, -4.08398678e-37, 3.53930504e-06, -1.35884636e-37, 7.63670542e-06, 4.41344416e-01,
      2.66457448e-34, 3.93615400e-03, -4.21834183e-35, 8.49406972e-03, 7.91841249e-01, 1.34353977e-33, 2.19940127e-03,
      5.21026905e-34, 7.03599803e-03, 9.64304151e-03, 1.70166116e-38, -1.68585912e-07, 3.36520448e-39, -4.80426657e-07,
      1.16343875e-02, -8.19066237e-36, -1.87451996e-04, 7.49349808e-36, -5.34314056e-04, 2.95201804e-02,
      -6.74992259e-35, -3.86583557e-05, -7.01076530e-36, -4.78598933e-04, 5.50262474e-04, 1.91367863e-36,
      8.39801590e-06, 4.96261112e-37, 2.96972630e-05, 0.00000000e+00],
     [8.86860807e-02, 2.42473156e-40, 9.86207458e-09, 7.15345634e-40, 1.55112186e-08, 1.38599053e-01, -9.43677361e-39,
      1.55066905e-05, -6.35371095e-37, 2.43894828e-05, 1.08301649e-01, 2.44942005e-34, 1.21910177e-02, 1.26328376e-33,
      1.91747319e-02, 3.77212522e-01, 5.97360132e-37, -3.00223787e-07, -2.14092434e-37, 5.02466167e-07, 4.16846254e-01,
      5.92792239e-34, -3.33572630e-04, -5.30396741e-35, 5.64477709e-04, 8.02207547e-01, 1.97171004e-33, 4.73948476e-04,
      3.91299174e-34, 1.22389679e-02, 9.14189252e-03, 2.31283371e-39, -1.71387515e-07, 1.84525233e-38, -3.93980343e-07,
      1.01024315e-02, -8.50809618e-36, -1.90570156e-04, -3.25972997e-36, -4.38068066e-04, 2.74948751e-02,
      -8.61886495e-36, -4.83292674e-05, 5.74596469e-36, -8.51092175e-05, 4.71179909e-04, 1.61837414e-37, 9.78671935e-06,
      6.60483393e-37, 7.64331537e-06, 0.00000000e+00],
     [8.04590892e-02, 8.53805919e-40, 5.35709039e-09, 2.78996495e-40, 9.59751851e-09, 1.50623158e-01, 1.20827325e-37,
      8.42352489e-06, -3.23485292e-37, 1.50926885e-05, 1.40986779e-01, 3.82689135e-35, 6.62260367e-03, 3.77980911e-34,
      1.18670913e-02, 3.53782286e-01, 4.61087288e-37, 6.47427106e-07, 3.67893771e-38, 5.57886293e-06, 4.68314627e-01,
      3.03475491e-35, 7.20406423e-04, -1.87915785e-34, 6.20567486e-03, 7.77798425e-01, 1.11044839e-33, 1.21349075e-03,
      1.76577613e-34, 6.14785269e-03, 1.00204866e-02, -2.07983910e-38, -8.05705911e-08, -7.02045806e-39,
      -4.96213248e-07, 1.32644868e-02, 6.01204952e-37, -8.95612147e-05, 1.28175061e-36, -5.51824775e-04, 3.11555043e-02,
      2.57081622e-35, 5.84276815e-05, 7.43972634e-36, -3.55784133e-04, 6.23982653e-04, 9.86137675e-37, 3.16924035e-05,
      1.02455209e-36, 3.97319223e-05, 0.00000000e+00],
     [8.92390566e-02, 2.99084854e-40, 1.11806280e-09, 1.19101568e-40, 9.35392188e-09, 8.90439939e-02, 2.35549827e-37,
      1.75785501e-06, 5.98993815e-39, 1.46999105e-05, 4.44246788e-02, 2.02507690e-34, 1.38187868e-03, 4.34294767e-34,
      1.15506300e-02, 3.92906460e-01, 1.32704434e-37, -4.43076102e-07, 1.06083085e-37, -2.34300399e-05, 2.77219536e-01,
      1.24432440e-33, -4.92500779e-04, -7.24213287e-34, -2.60334556e-02, 8.64954719e-01, 8.98200329e-33, 2.62502788e-04,
      7.83370673e-34, 3.52894585e-02, 7.92196516e-03, 2.13055708e-39, 5.24894396e-08, -1.03507839e-39, -2.76106105e-08,
      5.58943089e-03, 5.63330975e-36, 5.83529062e-05, -1.12184001e-36, -3.07723084e-05, 2.46633536e-02, 2.37906494e-35,
      -1.92228050e-05, 1.99600105e-36, -2.20005887e-04, 3.51625927e-04, 1.17656402e-37, 4.28063617e-06, 7.59778895e-38,
      1.13615763e-05, 0.00000000e+00],
     [8.33832361e-02, 6.63308513e-40, 3.51665961e-09, 8.91114659e-41, 6.21139272e-09, 1.22042048e-01, -4.22827636e-38,
      5.52961811e-06, -1.34456351e-37, 9.76715271e-06, 8.93120853e-02, 1.49340715e-34, 4.34740360e-03, 2.71448016e-34,
      7.67921953e-03, 3.71266307e-01, -1.90209202e-37, 4.21903590e-07, -2.42639621e-37, 1.69357809e-06, 3.84238855e-01,
      1.98227666e-34, 4.69556061e-04, 1.82221252e-34, 1.89139042e-03, 8.26537066e-01, 1.71913522e-34, 9.89740513e-04,
      1.09122742e-33, 1.77016461e-02, 9.74400415e-03, 1.49654846e-38, 7.26796136e-08, -1.28847103e-39, -3.40324723e-07,
      1.00844729e-02, 1.24108132e-35, 8.07739063e-05, -3.56784011e-37, -3.78586017e-04, 3.06781562e-02, 8.97895353e-36,
      -9.96760478e-05, -4.71590747e-36, -6.02291141e-04, 5.69332766e-04, 7.47848239e-37, 1.24224770e-05, 1.98131455e-37,
      2.40335149e-05, 0.00000000e+00],
     [8.22445077e-02, 1.20560296e-39, 6.23413520e-09, 4.17176269e-40, 8.82470396e-09, 1.41509453e-01, -3.82435144e-37,
      9.80341788e-06, 2.93903888e-37, 1.38781471e-05, 1.21740197e-01, 2.60859718e-34, 7.70812627e-03, 3.16258037e-34,
      1.09127158e-02, 3.61694110e-01, -1.26369874e-36, 3.19741811e-06, 5.25443910e-37, 7.38066599e-06, 4.40052998e-01,
      7.37822702e-34, 3.55557994e-03, 5.43860073e-34, 8.20929161e-03, 7.95327452e-01, 1.18791502e-33, 1.24547555e-03,
      9.55609127e-34, 6.81399875e-03, 9.37520000e-03, 3.54202840e-38, 2.21623529e-08, -1.02616905e-38, -5.64424147e-07,
      1.14062816e-02, -1.46323012e-35, 2.46794935e-05, 3.33816761e-36, -6.27610867e-04, 2.91541293e-02, -3.63590514e-35,
      1.15119621e-04, -7.13714750e-36, -1.98535310e-04, 5.34347991e-04, 6.51561419e-37, 1.77226434e-05, 6.40862927e-37,
      2.71619185e-05, 0.00000000e+00]])


def calc_desc_fake(configs_in, configs_out, descs, key, per_atom):
    # fake descriptor calculator for envs with no quippy installed
    if descs != "soap n_max=4 l_max=4 cutoff=4.0 atom_sigma=0.25 average" or per_atom:
        raise ValueError("pre-calculated descriptors are not for the given desc, fix your tests")

    for i, at in enumerate(configs_in):
        at.info[key] = ref_array[i]
        configs_out.store(at)

    configs_out.close()

    return configs_out.to_ConfigSet()


def get_indices(selected_configset):
    return [at0.info['config_i'] for at0 in selected_configset]


def test_greedy_fps_fake_descriptor(tmp_path):
    
    greedy_fps(calc_desc_fake, tmp_path)


def test_greedy_fps_quippy_descriptor(tmp_path, quippy):

    from wfl.descriptors.quippy import calculate
    greedy_fps(calculate, tmp_path)
    

def greedy_fps(calc_desc, tmp_path):

    np.random.seed(5)

    ats = [Atoms('Si4', cell=(2, 2, 2), pbc=[True] * 3, scaled_positions=np.random.uniform(size=(4, 3))) for _ in
           range(10)]
    for at_i, at in enumerate(ats):
        at.info['config_i'] = at_i

    ats_desc = calc_desc(ConfigSet(ats),
                         OutputSpec('test.select_greedy_FPS.desc.xyz', file_root=tmp_path),
                         descs='soap n_max=4 l_max=4 cutoff=4.0 atom_sigma=0.25 average', key='desc',
                         per_atom=False)

    print('try with desc in Atoms.info no exclude')
    np.random.seed(42)
    selected_indices = get_indices(greedy_fps_conf_global(
        ats_desc, OutputSpec('test.select_greedy_FPS.selected_no_exclude.xyz', file_root=tmp_path),
        num=5, at_descs_info_key='desc'))
    print("real descs no exclude selected_indices", selected_indices)
    assert selected_indices == [1, 2, 3, 4, 7]

    exclude_list = [list(ats_desc)[1]]

    print('try with desc in Atoms.info')
    np.random.seed(42)
    selected_indices1 = get_indices(greedy_fps_conf_global(
        ats_desc, OutputSpec('test.select_greedy_FPS.selected.xyz', file_root=tmp_path),
        num=5, at_descs_info_key='desc', exclude_list=exclude_list))
    print("real descs selected_indices", selected_indices1)
    assert selected_indices1 == [0, 3, 4, 7, 8]

    print('try with desc in separate array')
    separate_descs = [at.info['desc'] for at in ats_desc]
    ats = list(ats_desc)
    for at in ats:
        del at.info['desc']
    np.random.seed(42)
    selected_indices2 = get_indices(greedy_fps_conf_global(
        ats, OutputSpec('test.select_greedy_FPS.selected_separate_array.xyz', file_root=tmp_path),
        num=5, at_descs=separate_descs, exclude_list=exclude_list))
    print("separate selected_indices", selected_indices2)
    assert selected_indices2 == [0, 3, 4, 7, 8]

    # test output being done
    selected_indices2_from_output = get_indices(greedy_fps_conf_global(
        ats[::-1],  # this changes the indices, so if the calculation is performed then this should fail
        outputs=OutputSpec('test.select_greedy_FPS.selected_separate_array.xyz', file_root=tmp_path),
        num=5, at_descs=separate_descs, exclude_list=exclude_list))
    assert selected_indices2_from_output == selected_indices2

    # test if errors are raised
    with raises(RuntimeError, match="Asked for 20 configs but only 10 are available"):
        _ = greedy_fps_conf_global(inputs=ats_desc, outputs=OutputSpec("dummy.xyz", file_root=tmp_path),
                                   num=20, at_descs_info_key='desc')


def test_write_selected_and_clean():
    cfs_in = ConfigSet([Atoms('Si', cell=[i, i, i]) for i in range(3)])
    cfs_out = OutputSpec()

    # expected errors
    with raises(RuntimeError, match=r".*Got False.*"):
        write_selected_and_clean(cfs_in, cfs_out, [1, 2], keep_descriptor_info=False)
    with raises(AssertionError):
        write_selected_and_clean(cfs_in, cfs_out, [1, 1])

    # deleting the keys
    for at in cfs_in:
        at.info["dummy_desc"] = 'dummy'
    write_selected_and_clean(cfs_in, cfs_out, [0, 1], "dummy_desc", False)

    for at in cfs_out.to_ConfigSet():
        assert "dummy_desc" not in at.info.keys()


def test_prep_descs_and_exclude():
    cfs_in = ConfigSet([Atoms('Si') for _ in range(3)])

    with raises(AssertionError):
        _ = prep_descs_and_exclude(cfs_in, "None", "None", "None")


def test_speed(tmp_path):
    import time

    np.random.seed(5)
    ats_desc = [Atoms('Si4', cell=(2, 2, 2), pbc=[True] * 3, scaled_positions=np.random.uniform(size=(4, 3))) for _ in
                range(10000)]
    for at_i, at in enumerate(ats_desc):
        at.info['config_i'] = at_i
        at.info['desc'] = np.random.uniform(size=50)
        at.info['desc'] /= np.linalg.norm(at.info['desc'])

    np.random.seed(42)
    t0 = -time.perf_counter()
    selected = greedy_fps_conf_global(ats_desc,
                                      OutputSpec('test.select_greedy_FPS.no_real_desc.selected_O_N_sq.xyz', file_root=tmp_path),
                                      num=50, at_descs_info_key='desc', O_N_sq=True)
    t0 += time.perf_counter()
    print("O_N_sq 50/10000 runtime", t0)

    selected_inds_N_sq = [at.info['config_i'] for at in selected]
    print("fake descs selected_inds", selected_inds_N_sq)

    np.random.seed(42)
    t0 = -time.perf_counter()
    selected = greedy_fps_conf_global(ats_desc,
                                      OutputSpec('test.select_greedy_FPS.no_real_desc.selected_O_N.xyz', file_root=tmp_path),
                                      num=50, at_descs_info_key='desc', O_N_sq=False)
    t0 += time.perf_counter()
    print("O_N 50/10000 runtime", t0)

    selected_inds_N = [at.info['config_i'] for at in selected]
    print("fake descs selected_inds", selected_inds_N)

    assert selected_inds_N_sq == selected_inds_N


def test_prev_excl(tmp_path):
    np.random.seed(5)
    ats_desc = [Atoms('Si4', cell=(2, 2, 2), pbc=[True] * 3, scaled_positions=np.random.uniform(size=(4, 3))) for _ in
                range(6)]
    for at_i, at in enumerate(ats_desc):
        at.info['config_i'] = at_i
        at.info['desc'] = np.random.uniform(size=50)
        at.info['desc'] /= np.linalg.norm(at.info['desc'])

    descs_prev = []
    for _ in enumerate(ats_desc):
        v = np.random.uniform(size=50)
        v /= np.linalg.norm(v)
        descs_prev.append(v)
    descs_prev = np.asarray(descs_prev)

    np.random.seed(42)
    no_excl_selected_O_N_2 = get_indices(greedy_fps_conf_global(
        inputs=ats_desc, num=3, at_descs_info_key='desc', O_N_sq=True,
        outputs=OutputSpec('test.select_greedy_FPS.no_real_desc.no_prev_excl_O_Nsq.xyz', file_root=tmp_path)))

    np.random.seed(42)
    selected_O_N_2 = get_indices(greedy_fps_conf_global(
        inputs=ats_desc, outputs=OutputSpec('test.select_greedy_FPS.no_real_desc.prev_excl_O_Nsq.xyz', file_root=tmp_path),
        num=3, at_descs_info_key='desc', prev_selected_descs=descs_prev, O_N_sq=True))
    # with previous descriptors specified, the result is not the same
    assert no_excl_selected_O_N_2 != selected_O_N_2

    np.random.seed(42)
    selected_O_N = get_indices(greedy_fps_conf_global(
        inputs=ats_desc, outputs=OutputSpec('test.select_greedy_FPS.no_real_desc.prev_excl_O_N.xyz', file_root=tmp_path),
        num=3, at_descs_info_key='desc', prev_selected_descs=descs_prev, O_N_sq=False))

    assert selected_O_N_2 == selected_O_N

    # make sure that with verbose on, the code runs and the same results is observed
    indices_on2_verbose = get_indices(greedy_fps_conf_global(
        inputs=ats_desc, outputs=OutputSpec('test.select_greedy_FPS.no_real_desc.prev_excl_O_N1.xyz', file_root=tmp_path),
        num=3, at_descs_info_key='desc', prev_selected_descs=descs_prev, O_N_sq=True, verbose=True))
    indices_on_verbose = get_indices(greedy_fps_conf_global(
        inputs=ats_desc, outputs=OutputSpec('test.select_greedy_FPS.no_real_desc.prev_excl_O_N2.xyz', file_root=tmp_path),
        num=3, at_descs_info_key='desc', prev_selected_descs=descs_prev, O_N_sq=False, verbose=True))

    assert selected_O_N_2 == indices_on2_verbose
    assert indices_on_verbose == indices_on2_verbose

    indices_desc_conversion = get_indices(greedy_fps_conf_global(
        inputs=ats_desc, outputs=OutputSpec('test.select_greedy_FPS.no_real_desc.prev_excl_O_N3.xyz', file_root=tmp_path),
        num=3, at_descs_info_key='desc', prev_selected_descs=descs_prev.tolist(), O_N_sq=False, verbose=True))

    assert indices_desc_conversion == indices_on2_verbose
