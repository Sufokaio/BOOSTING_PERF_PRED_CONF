# Sklearn-compatible wrapper around HINNPerf with internal grid search for hyperparameter selection.
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class HINNPerfWrapper:

    def __init__(self, seed=42, test_mode=False):
        self.seed       = seed
        self.test_mode  = test_mode
        self._X_train   = None
        self._y_train   = None
        self._best_config = None

    def fit(self, X_train, y_train):
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
        except ImportError:
            pass

        from HINNPERF.HINNPerf_data_preproc import DataPreproc
        from HINNPERF.HINNPerf_models import MLPHierarchicalModel
        from HINNPERF.HINNPerf_model_runner import ModelRunner
        from HINNPERF.HINNPerf_args import list_of_param_dicts

        self._X_train = np.array(X_train, dtype=float)
        self._y_train = np.array(y_train, dtype=float).ravel()

        n_train = len(self._X_train)
        whole_data = np.column_stack([self._X_train, self._y_train])
        training_index = np.arange(n_train)
        testing_index  = np.arange(n_train)

        data_gen = DataPreproc(whole_data, training_index, testing_index)
        runner   = ModelRunner(data_gen, MLPHierarchicalModel)

        if self.test_mode:
            config_dict = dict(
                input_dim=[data_gen.config_num],
                num_neuron=[128],
                num_block=[2],
                num_layer_pb=[2],
                lamda=[0.001],
                linear=[False],
                gnorm=[True],
                lr=[0.001],
                decay=[None],
                verbose=[False],
            )
        else:
            config_dict = dict(
                input_dim=[data_gen.config_num],
                num_neuron=[128],
                num_block=[2, 3, 4],
                num_layer_pb=[2, 3, 4],
                lamda=[0.001, 0.1, 10.0, 100.0],
                linear=[False],
                gnorm=[True, False],
                lr=[0.0001, 0.001, 0.01],
                decay=[None],
                verbose=[False],
            )

        config_list = list_of_param_dicts(config_dict)
        best_val    = float("inf")
        best_config = config_list[0]

        for con in config_list:
            _, abs_error_val = runner.train(con)
            if abs_error_val < best_val:
                best_val    = abs_error_val
                best_config = con

        self._best_config = best_config
        return self

    def predict(self, X_test):
        from HINNPERF.HINNPerf_data_preproc import DataPreproc
        from HINNPERF.HINNPerf_models import MLPHierarchicalModel
        from HINNPERF.HINNPerf_model_runner import ModelRunner

        X_test = np.array(X_test, dtype=float)
        n_train = len(self._X_train)
        n_test  = len(X_test)

        whole_data = np.vstack([
            np.column_stack([self._X_train, self._y_train]),
            np.column_stack([X_test, np.zeros(n_test)]),
        ])
        training_index = np.arange(n_train)
        testing_index  = np.arange(n_train, n_train + n_test)

        data_gen = DataPreproc(whole_data, training_index, testing_index)
        runner   = ModelRunner(data_gen, MLPHierarchicalModel)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y_pred_test, _, _, _, _ = runner.test(self._best_config)

        return np.asarray(Y_pred_test).ravel()
