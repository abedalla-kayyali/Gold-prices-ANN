#! /usr//bin/python3

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from data_importer import synthetic_data, get_gold_prices

class HiddenLayer():
    def __init__(self, features, depth, pkeep):
        """A single hidden layer
        
        Arguments:
            features {int} -- Number of input features to the layer
            depth {int} -- Depth of the layer, neurons in the layer
            pkeep {float} -- Layer's probability of keeping for Dropout regularization
        """

        # Features of layer
        self._feats = features
        # Depth of layer
        self._depth = depth
        # Probability of keeping for dropout regularization
        self.pkeep = pkeep
        # Layer weights
        self._w = tf.Variable(tf.random.normal((self._feats, self._depth)))

    def layer_output(self, input_to_layer):
        """Output of the layer
        
        Arguments:
            input_to_layer {tensorflow variable} -- Input to the layer
        
        Returns:
            tensorlfow variable -- Output of the layer
        """

        return tf.nn.relu(tf.matmul(input_to_layer, self._w))

class FinalLayer():
    def __init__(self, features, pkeep):
        """The output layer of regressor
        
        Arguments:
            features {int} -- Number of input features to the layer
            pkeep {float} -- Layer's probability of keeping for Dropout regularization
        """

        # Features of layer
        self._feats = features
        # Probability of keeping for dropout regularization
        self.pkeep = pkeep
        # Layer weights
        self._w = tf.Variable(tf.random.normal((self._feats, 1)))

    def layer_output(self, input_to_layer):
        """The final output of the regressor
        
        Arguments:
            input_to_layer {tensorflow variable} -- Input to the layer
        
        Returns:
            tensorlfow variable -- Output of the layer
        """

        return tf.matmul(input_to_layer, self._w)

class ANNRegressor():
    def __init__(self, learning_rate, hidden_depths, layer_pkeeps, cache_decay, momentum):
        """ANN regressor.
        
        Arguments:
            learning_rate {float} -- Learning rate for Gradient descent
            hidden_depths {list} -- Depths of each layer, excluding final layer
            layer_pkeeps {list} -- Probability of keeping nodes/neurons for each layer, including input layer
            cache_decay {float} -- Cache decay rate for RMSProp
            momentum {float} -- Nestrov's momentum value
        """

        # Number of hidden layers
        self._no_hl = len(hidden_depths)
        # Depth of hidden layers
        self._hl_depths = hidden_depths
        # Learning rate
        self._lr = learning_rate
        # Momentum
        self._mu = momentum
        # RMSprop cache decay
        self._decay = cache_decay
        # Layers probability of keeping for dropout regularization
        self._pkeeps = layer_pkeeps
        # Inputs placeholder
        self._x_ph = tf.placeholder(tf.float32)
        # outputs placeholder
        self._y_hat_ph = tf.placeholder(tf.float32)
        # Reference to all hidden layer objects
        self._h_layers = list()
        # Placeholder for final layer
        self._f_layer = None
        # Placeholder for error function
        self._error_fn = None
        # Placeholder for output
        self._y = None
        # Placeholder for training function
        self._train_fn = None

    def _init_train_model(self, inputs):
        """[INTERNAL] Initialize the model for training.
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs for training
        """

        # Reinitialize hidden layers list
        self._h_layers = list()
        # Define hidden layers
        for l_no in range(self._no_hl):
            if l_no == 0:
                layer = HiddenLayer(inputs.shape[1], self._hl_depths[l_no], self._pkeeps[l_no])
            else:
                layer = HiddenLayer(self._hl_depths[l_no - 1], self._hl_depths[l_no], self._pkeeps[l_no])
            self._h_layers.append(layer)
        # Define final layer
        self._f_layer = FinalLayer(self._hl_depths[-1], self._pkeeps[-1])
        # Define layer's outputs
        layer_z = self._x_ph
        for layer in self._h_layers:
            layer_z = tf.nn.dropout(layer_z, layer.pkeep)
            layer_z = layer.layer_output(layer_z)
        # Define final output
        layer_z = tf.nn.dropout(layer_z, self._f_layer.pkeep)
        self._y = self._f_layer.layer_output(layer_z)
        # Define error function
        self._error_fn = tf.reduce_mean(tf.square(self._y - self._y_hat_ph))
        # Define training function
        self._train_fn = tf.train.RMSPropOptimizer(self._lr, momentum=self._mu, decay=self._decay).minimize(self._error_fn)

    def _calc_r2ed(self, targets, predicted):
        """[INTERNAL] Calculate the R squared error of targets and predictions.
        
        Arguments:
            targets {numpy.ndarray} -- Target values
            predicted {numpy.ndarray} -- Predicted values
        
        Returns:
            float -- the R squared value
        """

        # Sum of squared residual
        ss_res = np.sum(np.square(targets - predicted))
        # Sum of squared total
        ss_tot = np.sum(np.square(targets - targets.mean()))
        # R squared
        r_sq = 1 - (ss_res/ss_tot)
        return r_sq

    def _separate_test_train(self, inputs, targets, parts_train):
        """[INTERNAL] Separate data into testing and training sets by random selection
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to the model
            targets {numpy.ndarray} -- Target values
            parts_train {int} -- Parts used for training vs 1 part for testing
        
        Returns:
            Tuple -- Two tuples of training inputs/targets and testing inputs/targets
        """

        # Fix shape of targets if necessary
        if len(targets.shape) == 1:
            targets = np.expand_dims(targets, axis=1)
        # Find length of training data
        train_length = int(inputs.shape[0] - (inputs.shape[0] / parts_train))
        # Select random indices for train data
        indices = np.random.choice(
            np.arange(inputs.shape[0]), size=train_length, replace=False
        )
        # Separate training data
        inputs_train = inputs[indices, :]
        targets_train = targets[indices, :]
        # Separate test data
        inputs_test = np.delete(inputs, indices, axis=0)
        targets_test = np.delete(targets, indices, axis=0)
        # Make tuples for train and test data
        train_data = (inputs_train, targets_train)
        test_data = (inputs_test, targets_test)
        return (train_data, test_data)

    def _set_prediction_model(self):
        """[INTERNAL] Set the model to predict for prediction
        """

        # Redefine layer's outputs
        layer_z = self._x_ph
        for layer in self._h_layers:
            layer_z = layer_z * (1 / layer.pkeep)
            layer_z = layer.layer_output(layer_z)
        # Redefine final output
        layer_z = layer_z * (1 / self._f_layer.pkeep)
        self._y = self._f_layer.layer_output(layer_z)

    def _plot_training(self, errors, train_set, test_set):
        """[INTERNAL] Plot the training R squared errors, Final train data predictions vs target values
        and validation predicted vs target values
        
        Arguments:
            errors {list} -- R squared errors during training
            train_set {tuple} -- Training's final prediction vs target values
            test_set {tuple} -- Validation's prediction vs target values
        """

        # Plot training costs
        plt.plot(errors)
        plt.title('Training costs')
        plt.xlabel('Iterations')
        plt.ylabel('Costs')
        plt.show()
        # Plot training data and predictions
        plt.plot(train_set[0], label='Predictions')
        plt.plot(train_set[1], label='True data')
        plt.legend()
        plt.title('Training data\'s final prediction')
        plt.show()
        # Plot testing data and predictions
        plt.plot(test_set[0], label='Predictions')
        plt.plot(test_set[1], label='True data')
        plt.legend()
        plt.title('Testing data\'s final prediction')
        plt.show()

    def add_input_bias_col(self, inputs):
        """Add bias column to inputs to absorb weights
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to the model
        
        Returns:
            numpy.ndarray -- Inputs with bias column
        """

        # Fix shape of inputs if necessary
        if len(inputs.shape) == 1:
            inputs = np.expand_dims(inputs, axis=1)
        # Column of ones to accommodate bias into weights
        bias_col = np.ones([inputs.shape[0], 1])
        return np.hstack([bias_col, inputs])

    def fit(self, inputs, targets, no_iters, parts_train, batch_size, thresh=0.005, thresh_limit=100, model_dir=None, plot_results=False):
        """Fit the model to the given data
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to train to
            targets {numpy.ndarray} -- Target values
            no_iters {int} -- Number of iterations to train for/epochs
            parts_train {int} -- Parts of inputs/targets for training vs 1 part for validation
            batch_size {int} -- Batch size for training
        
        Keyword Arguments:
            thresh {float} -- Error percentage change which will break the iterations if met (b/w 0 and 1)(default: {0.005})
            thresh_limit {int} -- Number of continuos iterations for which the error threshold has to be met (default: {100})
            model_dir {str} -- Folder to save the model to (default: {None})
            plot_results {bool} -- Plot the training progress (default: {False})
        
        Returns:
            [type] -- [description]
        """

        # Threshold meeting counter
        thresh_count = 0
        # last iteration's R squared error for threshold
        r2ed_prev = None
        # Separate training and testing data
        train_data, test_data = self._separate_test_train(inputs, targets, parts_train)
        inputs_train = train_data[0]
        targets_train = train_data[1]
        inputs_test = test_data[0]
        targets_test = test_data[1]
        # Initialize the model
        self._init_train_model(inputs_train)
        # Errors placeholder
        errors = list()
        # Calculate number of batches
        no_batches = int(inputs_train.shape[0] / batch_size)
        # Current batch number
        batch_no = 0
        # Initialize tensorflow variables
        init = tf.global_variables_initializer()
        # Session saver
        saver = tf.train.Saver()
        # Start tensorflow session
        with tf.Session() as session:
            session.run(init)
            # For the number of iterations
            for i in range(no_iters):
                # Train the model
                session.run(
                    self._train_fn,
                    feed_dict={
                        self._x_ph: inputs_train[batch_no:batch_no+batch_size],
                        self._y_hat_ph: targets_train[batch_no:batch_no+batch_size]
                    }
                )
                # Calculate and store the error
                error = session.run(
                    self._error_fn,
                    feed_dict={self._x_ph: inputs_train, self._y_hat_ph: targets_train}
                )
                errors.append(float(error))
                if np.isnan(errors[-1]):
                    print('WARNING!!! NaN cost after {} iterations, breaking...'.format(i))
                    break
                # Increment batch number
                batch_no += 1
                # Check batch number for overflow
                if batch_no >= no_batches:
                    batch_no = 0
                # Check threshold criteria
                model_out = session.run(
                    self._y,
                    feed_dict={self._x_ph: inputs_train}
                )
                if r2ed_prev:
                    r2ed_curr = self._calc_r2ed(targets_train, model_out)
                    if np.sqrt(np.square(r2ed_prev - r2ed_curr)) <= thresh:
                        thresh_count += 1
                        if thresh_count >= thresh_limit:
                            print('Test error threshold met after {} iterations, breaking iterations...'.format(i))
                            break
                    else:
                        thresh_count = 0
                    r2ed_prev = r2ed_curr
                else:
                    r2ed_prev = self._calc_r2ed(targets_train, model_out)
                # Print progress of training
                if (not i == 0) and ((i % 100) == 0):
                    temp_pred = session.run(
                        self._y,
                        feed_dict={self._x_ph: inputs_test}
                    )
                    temp_r2ed = self._calc_r2ed(targets_test, temp_pred)*100
                    print(i, ' iterations |', round(r2ed_curr*100, 3), '% train R squared', end=' | ')
                    print(round(temp_r2ed, 3), '% test R squared ...')
            # Calculate final output
            final_output_train = session.run(
                self._y,
                feed_dict={self._x_ph: inputs_train}
            )
            # Calculate output for testing data
            final_output_test = session.run(
                self._y,
                feed_dict={self._x_ph: inputs_test}
            )
            if model_dir:
                saver.save(session, model_dir+'tf_model')
        print('R squared for training: ', self._calc_r2ed(targets_train, final_output_train)*100, '%')
        print('R squared for testing: ',self._calc_r2ed(targets_test, final_output_test)*100, '%')
        if plot_results:
            self._plot_training(errors, (final_output_train, targets_train), (final_output_test, targets_test))
        return errors, (final_output_train, targets_train), (final_output_test, targets_test)

    def predict(self, inputs, model_dir=None):
        """Predict values from given inputs.
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to predict for
        
        Keyword Arguments:
            model_dir {str} -- Model folder to load from (default: {None})
        
        Returns:
            numpy.ndarray -- Predicted values
        """

        saver = tf.train.Saver()
        with tf.Session() as sess:
            if model_dir:
                saver.restore(sess, model_dir+'tf_model')
            outputs = sess.run(
                self._y,
                feed_dict={self._x_ph: inputs}
            )
        return outputs

    def k_fold_x_validation(self, inputs, targets, no_iters, parts_train, batch_size, prams, thresh=0.005, thresh_limit=100):
        """K-Fold Cross validations for set of model parameters
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to train to
            targets {numpy.ndarray} -- Target values
            no_iters {int} -- Number of iterations to train for/epochs
            parts_train {int} -- Parts of inputs/targets for training vs 1 part for validation
            batch_size {int} -- Batch size for training
            params {list of tuples} -- Parameters to test for (same order as for __init__)
        
        Keyword Arguments:
            thresh {float} -- Error percentage change which will break the iterations if met (b/w 0 and 1)(default: {0.005})
            thresh_limit {int} -- Number of continuos iterations for which the error threshold has to be met (default: {100})
        """

        # Placeholder for test R squared
        test_r2ed = list()
        # For each parameter set
        for pram in prams:
            print('=======================================')
            print('For parameters: ', pram)
            # Set model parameters
            self._lr = pram[0]
            self._no_hl = pram[1]
            self._hl_depths = pram[2]
            self._pkeeps = pram[3]
            self._decay = pram[4]
            self._mu = pram[5]
            # Train the model
            _, _, (test_pred, test_true) = self.fit(inputs, targets, no_iters, parts_train, batch_size)
            # Store the results
            test_r2ed.append(self._calc_r2ed(test_true, test_pred))
        # Print results
        print('=======================================')
        print('Highest testing R squared is for parameter set number: ', test_r2ed.index(max(test_r2ed)) + 1)

def main():
    # Synthetic data for testing
    '''print('Getting data...')
    inputs, outputs = synthetic_data()
    print('Initializing model...')
    model = ANNRegressor(0.005, [5, 5], [1, 1, 1], 0.9, 0.9)
    inputs = model.add_input_bias_col(inputs)
    print('Training model...')
    model.fit(inputs, outputs, 1000, 10, 30, plot_results=True)'''

    # Gold prices dataset
    print('Getting data...')
    inputs, outputs = get_gold_prices()
    print('Initializing model...')
    model = ANNRegressor(0.005, [15, 15], [1, 1, 1], 0.9, 0.9)
    inputs = model.add_input_bias_col(inputs)
    print('Training model...')
    model.fit(inputs, outputs, 1000, 10, 30, plot_results=True)

if __name__ == '__main__':
    main()
