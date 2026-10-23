"""Train S-DON on lug3d. See README.md for data and usage."""

from run_config import parse_run_args, prepare_run, load_train_index


def main():
    args = parse_run_args("sdon", "lug3d")
    prepare_run(args, "sdon", "lug3d")
    import os
    os.environ['DDE_BACKEND'] = 'tensorflow'
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PowerTransformer
    from deepxde.backend import tf
    tf.config.optimizer.set_jit(True)
    import os
    import deepxde as dde
    from deepxde.data.data import Data
    from deepxde.data.sampler import BatchSampler
    import keras.backend as K
    import time as TT
    dde.config.disable_xla_jit()
    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
    data_loc = str(args.data_dir)
    n_cases = args.num_samples
    print('n_cases = ', n_cases)
    n_nodes = 29852
    HIDDEN = 256
    N_input_fn = 1
    N_component = 2
    N_output_frame = 100
    m = 101
    batch_size = args.batch_size
    seed = args.seed
    try:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)
    except:
        pass
    dde.config.set_default_float('float64')

    class DeepONetCartesianProd(dde.maps.NN):
        """Deep operator network for dataset in the format of Cartesian product.

            Args:
                    layer_sizes_branch: A list of integers as the width of a fully connected network,
                            or `(dim, f)` where `dim` is the input dimension and `f` is a network
                            function. The width of the last layer in the branch and trunk net should be
                            equal.
                    layer_sizes_trunk (list): A list of integers as the width of a fully connected
                            network.
                    activation: If `activation` is a ``string``, then the same activation is used in
                            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
                            net uses the activation `activation["trunk"]`, and the branch net uses
                            `activation["branch"]`.
            """

        def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, regularization=None):
            super().__init__()
            if isinstance(activation, dict):
                activation_branch = activation['branch']
                self.activation_trunk = dde.maps.activations.get(activation['trunk'])
            else:
                activation_branch = self.activation_trunk = dde.maps.activations.get(activation)
            self.branch = layer_sizes_branch[1]
            self.trunk = layer_sizes_trunk[0]
            self.b = tf.Variable(tf.zeros(1, dtype=dde.config.real(tf)))

        def call(self, inputs, training=False):
            x_func = inputs[0]
            x_loc = inputs[1]
            x_func = self.branch(x_func)
            if self._input_transform is not None:
                x_loc = self._input_transform(x_loc)
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = tf.einsum('bht,nhc->btnc', x_func, x_loc)
            x += self.b
            return x

    class TripleCartesianProd(Data):
        """Dataset with each data point as a triple. The ordered pair of the first two
            elements are created from a Cartesian product of the first two lists. If we compute
            the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

            This dataset can be used with the network ``DeepONetCartesianProd`` for operator
            learning.

            Args:
                    X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
                            `dim1`), and the second element has the shape (`N2`, `dim2`).
                    y_train: A NumPy array of shape (`N1`, `N2`).
            """

        def __init__(self, X_train, y_train, X_test, y_test):
            (self.train_x, self.train_y) = (X_train, y_train)
            (self.test_x, self.test_y) = (X_test, y_test)
            self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
            self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

        def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
            return loss_fn(targets, outputs)

        def train_next_batch(self, batch_size=None):
            if batch_size is None:
                return (self.train_x, self.train_y)
            if not isinstance(batch_size, (tuple, list)):
                indices = self.branch_sampler.get_next(batch_size)
                return ((self.train_x[0][indices], self.train_x[1]), self.train_y[indices])
            indices_branch = self.branch_sampler.get_next(batch_size[0])
            indices_trunk = self.trunk_sampler.get_next(batch_size[1])
            return ((self.train_x[0][indices_branch], self.train_x[1][indices_trunk]), self.train_y[indices_branch, indices_trunk])

        def test(self):
            return (self.test_x, self.test_y)
    trunk_scaling_factor = 1000.0
    xy_train_testing = np.load(os.path.join(data_loc, 'xyz_coords.npy')).astype(np.float64)
    xy_train_testing = xy_train_testing * trunk_scaling_factor
    n_nodes = xy_train_testing.shape[0]
    disp_filtered = np.load(os.path.join(data_loc, 'data_amp_5000.npy'))
    print('disp_filtered shape')
    print(disp_filtered.shape)
    temp_data = np.load(os.path.join(data_loc, 'peeq_all.npy'))
    print('temp_data shape')
    print(temp_data.shape)
    stress_scaling_factor = 1000000.0
    stress_data = np.load(os.path.join(data_loc, 'ystress_all.npy'))
    stress_data = stress_data / stress_scaling_factor
    print('stress_data shape')
    print(stress_data.shape)

    def Survey(data):
        print('Mean ', np.mean(data), ' , max ', np.max(data), ' , min ', np.min(data))
    print('----------')
    Heat_Amp = np.expand_dims(disp_filtered[:n_cases, :], axis=-1)
    data_t = temp_data[:n_cases, 1:, :]
    data_s = stress_data[:n_cases, 1:, :]
    print('data_t.shape = ', data_t.shape)
    print('data_s.shape = ', data_s.shape)
    Tshape = data_t.shape
    Sshape = data_s.shape
    fraction_train = 0.8
    print('fraction_train = ' + str(fraction_train))
    N_valid_case = len(Heat_Amp)
    N_train = int(N_valid_case * fraction_train)
    train_case = np.random.choice(N_valid_case, N_train, replace=False)
    test_case = np.setdiff1d(np.arange(N_valid_case), train_case)
    np.save('train_indices_trial_1.npy', train_case)
    np.save('test_indices_trial_1.npy', test_case)
    # Fit output scaling on the fixed training split only.
    scalerT = MinMaxScaler()
    scalerT.fit(data_t[train_case].reshape([-1, Tshape[2]]))
    scaled_temp = scalerT.transform(data_t.reshape([-1, Tshape[2]])).reshape(Tshape)
    print('Temp survey: ')
    Survey(scaled_temp)
    print('----------')
    scalerS = MinMaxScaler()
    scalerS.fit(data_s[train_case].reshape([-1, Sshape[2]]))
    scaled_stress = scalerS.transform(data_s.reshape([-1, Sshape[2]])).reshape(Sshape)
    print('Stress survey: ')
    Survey(scaled_stress)
    print('----------')
    Temp = np.zeros((n_cases, N_output_frame, n_nodes, N_component))
    Temp[:, :, :, 0] = scaled_temp
    Temp[:, :, :, 1] = scaled_stress
    print('Target shape: ', Temp.shape)
    u0_train = Heat_Amp[train_case, :]
    u0_testing = Heat_Amp[test_case, :]
    s_train = Temp[train_case, :]
    s_testing = Temp[test_case, :]
    print('u0_train.shape = ', u0_train.shape)
    print('type of u0_train = ', type(u0_train))
    print('u0_testing.shape = ', u0_testing.shape)
    print('s_train.shape = ', s_train.shape)
    print('s_testing.shape = ', s_testing.shape)
    print('xy_train_testing.shape', xy_train_testing.shape)
    x_train = (u0_train, xy_train_testing)
    y_train = s_train
    x_test = (u0_testing, xy_train_testing)
    y_test = s_testing
    data = TripleCartesianProd(x_train, y_train, x_test, y_test)
    my_act1 = 'tanh'
    branch = tf.keras.models.Sequential([tf.keras.layers.GRU(units=256, batch_input_shape=(batch_size, m, N_input_fn), activation=my_act1, return_sequences=True, dropout=0.0, recurrent_dropout=0.0), tf.keras.layers.GRU(units=128, activation=my_act1, return_sequences=False, dropout=0.0, recurrent_dropout=0.0), tf.keras.layers.RepeatVector(HIDDEN), tf.keras.layers.GRU(units=128, activation=my_act1, return_sequences=True, dropout=0.0, recurrent_dropout=0.0), tf.keras.layers.GRU(units=256, activation=my_act1, return_sequences=True, dropout=0.0, recurrent_dropout=0.0), tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(N_output_frame))])
    branch.summary()
    my_act2 = 'relu'
    trunk = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(3,)), tf.keras.layers.Dense(256, activation=my_act2, kernel_initializer='GlorotNormal'), tf.keras.layers.Dense(256, activation=my_act2, kernel_initializer='GlorotNormal'), tf.keras.layers.Dense(256, activation=my_act2, kernel_initializer='GlorotNormal'), tf.keras.layers.Dense(256, activation=my_act2, kernel_initializer='GlorotNormal'), tf.keras.layers.Dense(256, activation=my_act2, kernel_initializer='GlorotNormal'), tf.keras.layers.Dense(HIDDEN * N_component, activation=my_act2, kernel_initializer='GlorotNormal'), tf.keras.layers.Reshape([HIDDEN, N_component])])
    trunk.summary()
    net = DeepONetCartesianProd([m, branch], [trunk], my_act2, 'Glorot normal')
    model = dde.Model(data, net)
    print('y_train shape:', y_train.shape)

    def MSE(y_true, y_pred):
        tmp = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
        print('-----Inside MSE function-----')
        print('y_true type: ', type(y_true))
        print('y_true: ', y_true)
        print('y_pred type: ', type(y_pred))
        print('y_pred: ', y_pred)
        data_loss = tf.math.reduce_mean(tmp)
        return data_loss

    def MAE(y_true, y_pred):
        tmp = tf.math.abs(K.flatten(y_true) - K.flatten(y_pred), name='abs')
        data_loss = tf.math.reduce_mean(tmp)
        return data_loss

    def COP(y_true, y_pred):
        sqr_err = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
        var_true = y_true.shape[0] * tf.math.reduce_variance(K.flatten(y_true))
        data_loss = tf.math.divide(tf.math.reduce_sum(sqr_err), var_true)
        return data_loss

    def err(y_train, y_pred):
        ax = -1
        return np.linalg.norm(y_train - y_pred, axis=ax) / (np.linalg.norm(y_train, axis=ax) + 0.0001)

    def err2(y_train, y_pred):
        ax = 1
        abs_diff = np.abs(y_train - y_pred)
        test_minus_mean = np.abs(y_train - y_train.mean(axis=ax, keepdims=True))
        rel_abs_err = np.sum(abs_diff, axis=ax) / np.sum(test_minus_mean, axis=ax)
        return rel_abs_err

    def err3(y_train, y_pred):
        ax = 1
        sum_sqr_error = np.sum(np.power(y_train - y_pred, 2), axis=1)
        test_variance = np.sum(np.power(y_train - np.mean(y_train, axis=1, keepdims=True), 2), axis=1)
        Cop_sam = 1 - sum_sqr_error / test_variance
        return Cop_sam

    def metric1(y_train, y_pred):
        my_shape = y_train.shape[:-1]
        y_train_original = scalerS.inverse_transform(y_train[:, :, :, 1].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        y_pred_original = scalerS.inverse_transform(y_pred[:, :, :, 1].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        return np.mean(err(y_train_original, y_pred_original).flatten())

    def metric1_temp(y_train, y_pred):
        my_shape = y_train.shape[:-1]
        y_train_original = scalerT.inverse_transform(y_train[:, :, :, 0].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        y_pred_original = scalerT.inverse_transform(y_pred[:, :, :, 0].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        return np.mean(err(y_train_original, y_pred_original).flatten())

    def metric2(y_train, y_pred):
        y_train_original = scalerS.inverse_transform(y_train[:, :, :, 1])
        y_pred_original = scalerS.inverse_transform(y_pred[:, :, :, 1])
        return np.mean(err2(y_train_original, y_pred_original).flatten())

    def metric3(y_train, y_pred):
        y_train_original = scalerS.inverse_transform(y_train[:, :, :, 1])
        y_pred_original = scalerS.inverse_transform(y_pred[:, :, :, 1])
        return np.mean(err3(y_train_original, y_pred_original).flatten())

    def metric_mae_stress(y_train, y_pred):
        my_shape = y_train.shape[:-1]
        y_train_original = scalerS.inverse_transform(y_train[:, :, :, 1].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        y_pred_original = scalerS.inverse_transform(y_pred[:, :, :, 1].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        tmp = tf.math.abs(K.flatten(y_train_original) - K.flatten(y_pred_original))
        data_loss = tf.math.reduce_mean(tmp)
        return data_loss

    def metric_mae_peeq(y_train, y_pred):
        my_shape = y_train.shape[:-1]
        y_train_original = scalerT.inverse_transform(y_train[:, :, :, 0].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        y_pred_original = scalerT.inverse_transform(y_pred[:, :, :, 0].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
        tmp = tf.math.abs(K.flatten(y_train_original) - K.flatten(y_pred_original))
        data_loss = tf.math.reduce_mean(tmp)
        return data_loss
    model.compile('adam', lr=0.001, decay=('inverse time', 1, 0.0001), loss=MAE, metrics=[metric_mae_peeq, metric1, metric_mae_stress])
    (losshistory, train_state) = model.train(iterations=args.iterations, batch_size=batch_size, model_save_path='./Models/model_1br_1')
    np.save('losshistory_1br_1.npy', losshistory)
    st = TT.time()
    y_pred = model.predict(data.test_x)
    duration = TT.time() - st
    print('y_pred.shape =', y_pred.shape)
    print('Prediction took ', duration, ' s')
    print('Prediction speed = ', duration / float(len(y_pred)), ' s/case')
    my_shape = y_test.shape[:-1]
    Org_temp_test = scalerT.inverse_transform(y_test[:, :, :, 0].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
    Org_stress_test = scalerS.inverse_transform(y_test[:, :, :, 1].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
    Org_temp_pred = scalerT.inverse_transform(y_pred[:, :, :, 0].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
    Org_stress_pred = scalerS.inverse_transform(y_pred[:, :, :, 1].reshape([my_shape[0] * my_shape[1], my_shape[2]])).reshape(my_shape)
    print('Successful to convert to original forms')
    print()
    np.savez_compressed('model_1br_1.npz', a=Org_temp_test, b=Org_stress_test, c=Org_temp_pred, d=Org_stress_pred, e=test_case)
    print('Saving all successful')
    error_t = err(y_test[:, :, :, 0], y_pred[:, :, :, 0])
    error_s = err(y_test[:, :, :, 1], y_pred[:, :, :, 1])
    org_error_t = err(Org_temp_test, Org_temp_pred)
    org_error_s = err(Org_stress_test, Org_stress_pred)
    rae_org_error_t = err2(Org_temp_test, Org_temp_pred)
    rae_org_error_s = err2(Org_stress_test, Org_stress_pred)
    print('Single training run results')
    print()
    print('error_t = ', error_t)
    print()
    print('----------------------------------------')
    print()
    print('error_s = ', error_s)
    print('----------------------------------------')
    print()
    print()
    print('Stress_scaling_factor: ', stress_scaling_factor)
    print('Trunk_scaling_factor: ', trunk_scaling_factor)
    print()
    print('Scaled L2 error')
    print('mean of temperature relative L2 error of s: {:.2e}'.format(error_t.mean()))
    print('std of temperature relative L2 error of s: {:.2e}'.format(error_t.std()))
    print('--------------------------------------------------------------')
    print('mean of stress relative L2 error of s: {:.2e}'.format(error_s.mean()))
    print('std of stress relative L2 error of s: {:.2e}'.format(error_s.std()))
    print('--------------------------------------------------------------')
    print('--------------------------------------------------------------')
    print()
    print('Origianl L2 error')
    print('mean of Original temperature relative L2 error of t: {:.2e}'.format(org_error_t.mean()))
    print('std of Original temperature relative L2 error of t: {:.2e}'.format(org_error_t.std()))
    print('--------------------------------------------------------------')
    print('mean of Orginal stress relative L2 error of s: {:.2e}'.format(org_error_s.mean()))
    print('std of Orginal stress relative L2 error of s: {:.2e}'.format(org_error_s.std()))
    print('--------------------------------------------------------------')
    print('--------------------------------------------------------------')
    print()
    print('Origianl RAE')
    print('mean of Original temperature relative RAE error of t: {:.2e}'.format(rae_org_error_t.mean()))
    print('std of Orginal temperature relative RAE error of t: {:.2e}'.format(rae_org_error_t.std()))
    print('--------------------------------------------------------------')
    print('mean of Original stress relative RAE error of s: {:.2e}'.format(rae_org_error_s.mean()))
    print('std of Orginal stress relative RAE error of s: {:.2e}'.format(rae_org_error_s.std()))
    print('--------------------------------------------------------------')
    print('--------------------------------------------------------------')
    print()
    print()
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print()
    print()
    print()
    plt.hist(error_s.flatten(), bins=25)
    plt.savefig('Stress_Err_hist_1br0.jpg', dpi=300)
    plt.hist(error_t.flatten(), bins=25)
    plt.savefig('Temp_Err_hist_1br0.jpg', dpi=300)


if __name__ == "__main__":
    main()
