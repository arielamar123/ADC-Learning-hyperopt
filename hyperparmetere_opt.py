import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.plot.contour import plot_contour
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements

###########################
# Parmeters Intialization #
###########################


num_rx_antenas = 6  # Number of Rx antennas
dense_sampling_L = 20  # Observed discretized time frame
num_transmitted_symbols = 4  # Number of transmitted symbols
valid_percent = 0.2  # Percentage of training data used for validation
train_size = int(1e4)  # Training size 4
validation_size = int(train_size * valid_percent)
test_size = int(1e5)  # Test data size 5
num_of_channels = 1  # 1 - Gaussian channel
BPSK_symbols = [-1, 1]  # BPSK symbols
error_variance = 0.1  # error variance for csi uncertainty
frame_size = 200
f_0 = 1e3
w = 2 * np.pi * f_0
snr_vals = 10 ** (-0.1 * np.arange(13))
epochs = 20  # 25
batch_size = 50
time_vec = np.arange(1, dense_sampling_L + 1) / dense_sampling_L
channel_matrix_cos = 1 + 0.5 * np.cos(time_vec)
gaussian_sampling_std = 0.4
noise_vector = 1 + 0.3 * np.cos(1.5 * (np.arange(1, dense_sampling_L + 1)) + 0.2)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

BITS_ERR_MSG = '**************too much bits*****************'
###########################
# channel matrix creation #
###########################


channel_matrix_exp = np.exp(-np.abs(np.ones((num_rx_antenas, 1)) * np.arange(num_transmitted_symbols) - (
        np.ones((num_transmitted_symbols, 1)) * [np.arange(num_rx_antenas)]).T))


def create_received_signal(data, ro=1):
    '''
    Creating x(t) = G(t)*s+w(t) the signal received by the antenna
    n - number of antennas
    L - number of dense samples
    N - data size
    :param data: sequence of symbols to pass through the channel (4, N)
    :param ro: SNR
    :return: x(t) = G(t)*s+w(t) (n*L, N)
    '''
    channel_matrix = np.zeros((num_rx_antenas * dense_sampling_L, data.shape[1]))
    for t in range(dense_sampling_L):
        signal = np.dot(channel_matrix_cos[t] * channel_matrix_exp, data)
        # power_of_signal = signal.var()
        # noise_power = power_of_signal / ro
        channel_matrix[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.sqrt(
            ro) * signal + np.random.randn(
            num_rx_antenas,
            data.shape[1])
    return channel_matrix


def create_received_signal_uncertanity_error(data, ro=1):
    '''
    Creating x(t) = G(t)*s+w(t) the signal received by the antenna
    n - number of antennas
    L - number of dense samples
    N - data size
    :param data: sequence of symbols to pass through the channel (4, N)
    :param ro: SNR
    :return: x(t) = G(t)*s+w(t) (n*L, N)
    '''

    noisy_channel_matrix = np.zeros((num_rx_antenas * dense_sampling_L, data.shape[1]))
    for t in range(dense_sampling_L):
        noisy_channel_matrix[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.random.randn(
            num_rx_antenas,
            data.shape[1])

    noisy_recieved_signal = np.zeros((num_rx_antenas * dense_sampling_L, data.shape[1]))
    for t in range(dense_sampling_L):
        signal = np.zeros((num_rx_antenas, data.shape[1]))
        noisy_channel_matrix_exp = channel_matrix_exp * (
                1 + np.sqrt(error_variance) * np.random.randn(channel_matrix_exp.shape[0],
                                                              channel_matrix_exp.shape[1]))
        for f in range(data.shape[1] // frame_size):
            signal[:, f * frame_size:(f + 1) * frame_size] = np.dot(channel_matrix_cos[t] * noisy_channel_matrix_exp,
                                                                    data[:, f * frame_size:(f + 1) * frame_size])
        noisy_recieved_signal[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.sqrt(ro) * signal
    return noisy_recieved_signal + noisy_channel_matrix


###############################
# Creating the Neural Network #
###############################

class AnalogNetwork(nn.Module):
    def __init__(self, num_of_adc_p):
        super(AnalogNetwork, self).__init__()
        self.analog_filter = nn.Linear(num_rx_antenas * dense_sampling_L, num_of_adc_p * dense_sampling_L,
                                       bias=False).double()  # without bias to get only the matrix

    def forward(self, x):
        x = self.analog_filter(x)
        return x


class SamplingLayer(nn.Module):
    def __init__(self, num_of_adc_p, num_samples_L_tilde):
        super(SamplingLayer, self).__init__()
        start_samples = np.linspace(0, dense_sampling_L, num_samples_L_tilde + 2, dtype=np.float64)
        self.weight = torch.nn.Parameter(data=torch.from_numpy(start_samples[1:-1]), requires_grad=True)
        self.num_of_adc_p = num_of_adc_p
        self.num_samples_L_tilde = num_samples_L_tilde

    def forward(self, x):
        out = torch.zeros((len(x), self.num_samples_L_tilde * self.num_of_adc_p)).double().to(device)
        t = torch.from_numpy(np.arange(1, dense_sampling_L + 1, dtype=np.double)).to(device)
        for v, j in enumerate(range(0, dense_sampling_L * self.num_of_adc_p, dense_sampling_L)):
            for k in range(self.num_samples_L_tilde):
                out[:, v * self.num_samples_L_tilde + k] = torch.sum(
                    x[:, j:j + dense_sampling_L] * torch.exp(
                        -(t - self.weight[k]) ** 2 / gaussian_sampling_std ** 2), dim=1)
        return out


class HardSamplingLayer(nn.Module):
    def __init__(self, weight, num_of_adc_p, num_samples_L_tilde):
        super(HardSamplingLayer, self).__init__()
        weight = weight.cpu().detach().numpy()
        weight = np.round(weight)
        rounded_weight = torch.from_numpy(weight)
        rounded_weight[rounded_weight < 1] = 1
        rounded_weight[rounded_weight > dense_sampling_L] = dense_sampling_L - 1
        self.weight = rounded_weight.long()
        self.num_of_adc_p = num_of_adc_p
        self.num_samples_L_tilde = num_samples_L_tilde

    def forward(self, x):
        out = torch.zeros((len(x), self.num_samples_L_tilde * self.num_of_adc_p)).double().to(device)
        for i in range(self.num_of_adc_p):
            for j in range(self.num_samples_L_tilde):
                # try:
                out[:, i * self.num_samples_L_tilde + j] = x[:, i * dense_sampling_L + self.weight[j]]
                # except:
                #     print('*******ERROR******** ', self.weight)
                #     exit(1)
        return out


class QuantizationLayer(nn.Module):

    def __init__(self, num_code_words, max_labels, max_samples):
        super(QuantizationLayer, self).__init__()
        self.a = torch.nn.Parameter(
            data=torch.from_numpy(
                np.ones(num_code_words - 1) * max_labels / num_code_words
            ), requires_grad=False)
        self.b = torch.nn.Parameter(
            data=torch.from_numpy(
                np.linspace(-1, 1, num_code_words - 1) * max_samples),
            requires_grad=True)
        if len(self.b) > 1:
            self.c = torch.nn.Parameter(
                data=torch.from_numpy(
                    (15 / np.mean(np.diff(self.b.data.numpy()))) * np.ones(num_code_words - 1)
                ), requires_grad=False)
        else:
            self.c = torch.nn.Parameter(data=torch.from_numpy(15 / int(self.b.data) * np.ones(num_code_words - 1)),
                                        requires_grad=False)
        self.num_code_words = num_code_words

    def forward(self, x):
        z = torch.zeros(self.num_code_words - 1, x.shape[0], x.shape[1]).double().to(device)
        for i in range(self.num_code_words - 1):
            z[i, :, :] = self.a[i] * torch.tanh(self.c[i] * (x - self.b[i]))
        return torch.sum(z, dim=0)

    def plot(self, ro):
        bdiff = torch.max(torch.from_numpy(np.diff(self.b.data.numpy())))
        x_vals = np.linspace(np.min(self.b.data.numpy()) - bdiff, np.max(self.b.data.numpy()) + bdiff, 1000)
        quant = []
        for x_val in x_vals:
            quant.append(torch.sum(self.a * torch.tanh(self.c * (x_val - self.b))))
        plt.figure(ro)
        plt.title(str(ro))
        plt.plot(x_vals, quant)


class HardQuantizationLayer(nn.Module):

    def __init__(self, a, b, c, num_code_words):
        super(HardQuantizationLayer, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.num_code_words = num_code_words

    def forward(self, x):
        z = torch.zeros(self.num_code_words - 1, x.shape[0], x.shape[1]).double().to(device)
        for i in range(self.num_code_words - 1):
            z[i, :, :] = self.a[i] * torch.sign(self.c[i] * (x - self.b[i]))
        return torch.sum(z, dim=0)


class DigitalNetwork(nn.Module):
    def __init__(self, num_of_adc_p, num_samples_L_tilde):
        super(DigitalNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(num_of_adc_p * num_samples_L_tilde, 32, bias=False).double()
        self.fc2 = torch.nn.Linear(32, 16, bias=False).double()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ADCNet(nn.Module):
    def __init__(self, num_of_adc_p, num_samples_L_tilde, num_code_words, max_labels, max_samples):
        super(ADCNet, self).__init__()
        self.analog_network = AnalogNetwork(num_of_adc_p).to(device)
        self.sampling_layer = SamplingLayer(num_of_adc_p, num_samples_L_tilde).to(device)
        self.quantization_layer = QuantizationLayer(num_code_words, max_labels, max_samples).to(device)
        self.digital_network = DigitalNetwork(num_of_adc_p, num_samples_L_tilde).to(device)

    def forward(self, x):
        x = self.analog_network(x)
        x = self.sampling_layer(x)
        x = self.quantization_layer(x)
        x = self.digital_network(x)
        return x


def train(detector, parameters, train_opt_data):
    nets = []
    for r, ro in tqdm(enumerate(np.flip(snr_vals))):

        training_set, labels_train = train_opt_data[r]

        validation_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, validation_size), p=[0.5, 0.5])

        labels_validation = np.sum(0.5 * (validation_data + 1).T * 2 ** np.flip(np.arange(4)), axis=1).astype(np.long)

        validation_samples = detector(validation_data, ro).T
        validation_set = []

        for i in range(len(validation_samples)):
            validation_set.append([validation_samples[i], labels_validation[i]])

        # num_of_adc_p = 2.0 ** parameters['num_of_adc_p']
        num_of_adc_p = 1
        num_samples_L_tilde = 2.0 ** parameters['num_samples_L_tilde']
        num_code_words = 2.0 ** (2.0 ** parameters['num_code_words'])

        print('p: ', num_of_adc_p, ', actual: ', int(round(num_of_adc_p)))
        print('L: ', num_samples_L_tilde, ', actual: ', int(round(num_samples_L_tilde)))
        print('code_words: ', num_code_words, 'bits: ', int(round(np.log2(num_code_words))))
        overall_bits = int(round(num_of_adc_p)) * int(round(num_samples_L_tilde)) * int(round(np.log2(num_code_words)))
        print('overall bits: ', overall_bits)
        net = ADCNet(int(round(num_of_adc_p)), int(round(num_samples_L_tilde)), int(round(num_code_words)),
                     np.max(labels_train), np.max(train_samples)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-2)

        valid_loss_min = np.Inf
        val_net = deepcopy(net)
        for epoch in tqdm(range(epochs)):
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            ###################
            # train the model #
            ###################
            net.train()
            for i, data in tqdm(enumerate(DataLoader(training_set, batch_size=batch_size))):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = net(inputs)
                # calculate the batch loss
                loss = criterion(outputs, labels.long())
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * batch_size
            ######################
            # validate the model #
            ######################
            net_val = net
            net_val.eval()
            with torch.no_grad():
                net_val.sampling_layer = HardSamplingLayer(net.sampling_layer.weight, int(round(num_of_adc_p)),
                                                           int(round(num_samples_L_tilde)))
                net_val.quantization_layer = HardQuantizationLayer(net.quantization_layer.a, net.quantization_layer.b,
                                                                   net.quantization_layer.c, int(round(num_code_words)))
                for i, data in tqdm(enumerate(DataLoader(validation_set, batch_size=batch_size))):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = net_val(inputs)
                    # calculate the batch loss
                    loss = criterion(outputs, labels.long())
                    # update validation loss
                    valid_loss += loss.item() * batch_size
                train_loss = train_loss / train_size
                valid_loss = valid_loss / validation_size
                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    epoch, train_loss, valid_loss))

                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
                    val_net = net
                    valid_loss_min = valid_loss
        nets.append(val_net)
    return nets


def evaluate(parameters, nets, alpha):
    BER = []

    # num_of_adc_p = 2.0 ** parameters['num_of_adc_p']
    num_of_adc_p = 1
    num_samples_L_tilde = 2.0 ** parameters['num_samples_L_tilde']
    num_code_words = 2.0 ** (2.0 ** parameters['num_code_words'])
    overall_bits = int(round(num_of_adc_p)) * int(round(num_samples_L_tilde)) * int(round(np.log2(num_code_words)))

    for r, ro in enumerate(np.flip(snr_vals)):
        test_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, test_size), p=[0.5, 0.5])
        labels_test = np.sum(0.5 * (test_data + 1).T * 2 ** np.flip(np.arange(4)), axis=1).astype(np.long)

        test_samples = create_received_signal(test_data, ro).T
        test_set = []

        for i in range(len(test_samples)):
            test_set.append([test_samples[i], labels_test[i]])

        best_net = nets[r]
        net_test = best_net

        print('p: ', num_of_adc_p, ', actual: ', int(round(num_of_adc_p)))
        print('L: ', num_samples_L_tilde, ', actual: ', int(round(num_samples_L_tilde)))
        print('code_words: ', num_code_words, 'bits: ', int(round(np.log2(num_code_words))))
        print('overall bits: ', overall_bits)

        net_test.eval()
        net_test.sampling_layer = HardSamplingLayer(best_net.sampling_layer.weight, int(round(num_of_adc_p)),
                                                    int(round(num_samples_L_tilde)))
        net_test.quantization_layer = HardQuantizationLayer(best_net.quantization_layer.a,
                                                            best_net.quantization_layer.b,
                                                            best_net.quantization_layer.c, int(round(num_code_words)))
        with torch.no_grad():
            cur_error = []
            for i, data in tqdm(enumerate(DataLoader(test_set, batch_size=batch_size))):
                inputs, labels = data
                inputs = inputs.to(device)
                net_outputs = net_test(inputs)
                outputs = net_outputs.argmax(dim=1)
                power = 2 ** np.flip(np.arange(4))
                est_ber = np.floor((outputs.cpu().numpy()[:, None] % (2 * power)) / power).T
                est_ber_scale = 2 * est_ber - np.ones((num_transmitted_symbols, batch_size))
                cur_error.append(
                    np.mean(np.mean(est_ber_scale != test_data[:, i * batch_size:(i + 1) * batch_size], axis=0)))
            BER.append(sum(cur_error) / len(cur_error))
    return sum(BER) / len(BER) + alpha * overall_bits


def evaluate_plot(parameters, nets):
    BER = []
    for r, ro in enumerate(np.flip(snr_vals)):

        test_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, test_size), p=[0.5, 0.5])
        labels_test = np.sum(0.5 * (test_data + 1).T * 2 ** np.flip(np.arange(4)), axis=1).astype(np.long)
        test_samples = create_received_signal(test_data, ro).T
        test_set = []

        for i in range(len(test_samples)):
            test_set.append([test_samples[i], labels_test[i]])

        best_net = nets[r]
        net_test = deepcopy(best_net)

        # num_of_adc_p = 2.0 ** parameters['num_of_adc_p']
        num_of_adc_p = 1
        num_samples_L_tilde = 2.0 ** parameters['num_samples_L_tilde']
        num_code_words = 2.0 ** (2.0 ** parameters['num_code_words'])

        print('p: ', num_of_adc_p, ', actual: ', int(round(num_of_adc_p)))
        print('L: ', num_samples_L_tilde, ', actual: ', int(round(num_samples_L_tilde)))
        print('code_words: ', num_code_words, 'bits: ', int(round(np.log2(num_code_words))))
        overall_bits = int(round(num_of_adc_p)) * int(round(num_samples_L_tilde)) * int(round(np.log2(num_code_words)))
        print('overall bits: ', overall_bits)

        net_test.eval()
        net_test.sampling_layer = HardSamplingLayer(best_net.sampling_layer.weight, int(round(num_of_adc_p)),
                                                    int(round(num_samples_L_tilde)))
        net_test.quantization_layer = HardQuantizationLayer(best_net.quantization_layer.a,
                                                            best_net.quantization_layer.b,
                                                            best_net.quantization_layer.c, int(round(num_code_words)))
        with torch.no_grad():
            cur_error = []
            for i, data in tqdm(enumerate(DataLoader(test_set, batch_size=batch_size))):
                inputs, labels = data
                inputs = inputs.to(device)
                net_outputs = net_test(inputs)
                outputs = net_outputs.argmax(dim=1)
                power = 2 ** np.flip(np.arange(4))
                est_ber = np.floor((outputs.cpu().numpy()[:, None] % (2 * power)) / power).T
                est_ber_scale = 2 * est_ber - np.ones((num_transmitted_symbols, batch_size))
                cur_error.append(
                    np.mean(np.mean(est_ber_scale != test_data[:, i * batch_size:(i + 1) * batch_size], axis=0)))
            BER.append(sum(cur_error) / len(cur_error))
    print('The BER for this trial:', BER)
    return BER


def train_evaluate(parameterization, alpha):
    nets = train(create_received_signal, parameterization, train_opt_data)
    return evaluate(parameterization, nets, alpha)


def hyperparameter_optimization(train_opt_data, alpha):
    best_parameters, values, experiment, model = optimize(
        parameters=[
            # {"name": "num_of_adc_p", "type": "range", "bounds": [0.0, np.log2(25)]},
            {"name": "num_samples_L_tilde", "type": "range", "bounds": [float(np.log2(3)), np.log2(6)]},
            {"name": "num_code_words", "type": "range", "bounds": [float(np.log2(3)), np.log2(6)]},
        ],
        evaluation_function=lambda parameterization: train_evaluate(parameterization, alpha),
        objective_name='Bit Error Rate', minimize=True,
        parameter_constraints=[' num_samples_L_tilde + num_code_words <= 5.129283016944']

        , total_trials=20)  # 'num_of_adc_p + num_samples_L_tilde + num_code_words <= 4.90689059561'
    best_parameters['num_of_adc_p'] = 1
    nets = train(create_received_signal, best_parameters, train_opt_data)
    BER = evaluate_plot(best_parameters, nets)

    for r, ro in enumerate(BER):
        print('ro = ', r, ' BER: ', ro)

    print('Best parameters are: ', best_parameters)
    return BER, best_parameters, values, experiment, model


def plot_hyperparameters_contours(alpha, model):
    for i in range(1, 21):
        plot_config_1 = plot_contour(model=model, param_x='num_of_adc_p', param_y='num_samples_L_tilde',
                                     metric_name='Bit Error Rate',
                                     slice_values={'num_code_words': i}, lower_is_better=True)

        with open(str(alpha) + '_p_vs_L_tilde__code_words_' + str(i) + '.html', 'w') as outfile:
            outfile.write(render_report_elements(
                "example_report",
                html_elements=[plot_config_to_html(plot_config_1)],
                header=False,
            ))

    for i in range(1, 21):
        plot_config_2 = plot_contour(model=model, param_x='num_of_adc_p', param_y='num_code_words',
                                     metric_name='Bit Error Rate', slice_values={'num_samples_L_tilde': np.log2(i)},
                                     lower_is_better=True)

        with open(str(alpha) + '_p_vs_code_words__L_tilde_' + str(i) + '.html', 'w') as outfile:
            outfile.write(render_report_elements(
                "example_report",
                html_elements=[plot_config_to_html(plot_config_2)],
                header=False,
            ))

    for i in range(1, 21):
        plot_config_3 = plot_contour(model=model, param_x='num_samples_L_tilde', param_y='num_code_words',
                                     metric_name='Bit Error Rate', slice_values={'num_of_adc_p': np.log2(i)},
                                     lower_is_better=True)

        with open(str(alpha) + '_L_tilde_vs_code_words__p_' + str(i) + '.html', 'w') as outfile:
            outfile.write(render_report_elements(
                "example_report",
                html_elements=[plot_config_to_html(plot_config_3)],
                header=False,
            ))


def plot_BER_vs_iterations(alpha, experiment):
    best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(best_objectives, axis=1),
        title="Overall BER vs. # of iterations",
        ylabel="Bit Error Rate",
    )

    with open(str(alpha) + '_BER_VS_ITERATIONS.html', 'w') as outfile:
        outfile.write(render_report_elements(
            "example_report",
            html_elements=[plot_config_to_html(best_objective_plot)],
            header=False,
        ))


def plot_original_vs_optimized(BER, BER_ORIGINAL, best_parameters, original_parameters, alphas):
    for ber, best_params, alpha in zip(BER, best_parameters, alphas):
        # best_p = best_params['num_of_adc_p']
        best_p = 0
        best_L_tilde = best_params['num_samples_L_tilde']
        best_code_words = best_params['num_code_words']
        plt.plot(np.arange(13), ber, marker='o',
                 label='Perfect CSI optimized - alpha=' + str(alpha) + ' p = ' + str(
                     int(round(2.0 ** best_p))) + ',L_tilde = ' + str(
                     int(round(2.0 ** best_L_tilde))) + ',Code Words = ' + str(
                     int(round(2.0 ** best_code_words))) + ',Bits used: ' + str(
                     int(round(2.0 ** best_p)) * int(round(2.0 ** best_L_tilde)) * int(
                         round(best_code_words))))
    plt.title('BER vs SNR')
    plt.plot(np.arange(13), BER_ORIGINAL, marker='o', color='orange',
             label='Perfect CSI not optimized' + ' p = ' + str(original_parameters['num_of_adc_p']) + ' L_tilde = ' +
                   str(int(round(2.0 ** original_parameters['num_samples_L_tilde']))) + ' Code Words = ' + str(
                 int(round(2.0 ** (2.0 ** original_parameters['num_code_words'])))) + ' Bits used: ' + str(
                 original_parameters['num_of_adc_p'] * int(
                     round(2.0 ** original_parameters['num_samples_L_tilde'])) * int(
                     round(2.0 ** original_parameters[
                         'num_code_words']))))
    plt.ylabel('BER')
    plt.xlabel('SNR[db]')
    plt.yscale('log')
    plt.ylim((10 ** (-6), 10 ** (-1)))
    plt.legend()
    plt.savefig('result3.png')
    plt.show()


if __name__ == '__main__':

    #################
    # Data Creation #
    #################

    train_opt_data = []
    for r, ro in tqdm(enumerate(np.flip(snr_vals))):
        train_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, train_size), p=[0.5, 0.5])
        labels_train = np.sum(0.5 * (train_data + 1).T * 2 ** np.flip(np.arange(4)), axis=1).astype(np.long)
        train_samples = create_received_signal(train_data, ro).T
        training_set = []

        for i in range(len(train_samples)):
            training_set.append([train_samples[i], labels_train[i]])

        train_opt_data.append((training_set, labels_train))

    ####################################################
    # Train the model with hyperparameter optimization #
    ####################################################
    alphas = [0.02, 0.02, 0.02]
    # alphas = [0.02, 0.04, 0.06]
    # BER_0, best_parameters_0, values_0, experiment_0, model_0 = hyperparameter_optimization(train_opt_data, alphas[0])
    # plot_hyperparameters_contours(alphas[0], model_0)
    # plot_BER_vs_iterations(alphas[0], experiment_0)

    # BER_1, best_parameters_1, values_1, experiment_1, model_1 = hyperparameter_optimization(train_opt_data, alphas[1])
    # plot_hyperparameters_contours(alphas[1], model_1)
    # plot_BER_vs_iterations(alphas[1], experiment_1)

    BER_2, best_parameters_2, values_2, experiment_2, model_2 = hyperparameter_optimization(train_opt_data, alphas[2])
    # # plot_hyperparameters_contours(alphas[2], model_2)
    plot_BER_vs_iterations(alphas[2], experiment_2)

    ####################################################
    #   Train the model with original hyperparmeters   #
    ####################################################

    original_parameters = {}
    # original_parameters['num_of_adc_p'] = np.log2(1)
    # original_parameters['num_samples_L_tilde'] = np.log2(6)
    # original_parameters['num_code_words'] = np.log2(256)
    original_parameters['num_of_adc_p'] = 1
    original_parameters['num_samples_L_tilde'] = np.log2(6)
    original_parameters['num_code_words'] = np.log2(8)

    nets_original = train(create_received_signal, original_parameters, train_opt_data)
    BER_ORIGINAL = evaluate_plot(original_parameters, nets_original)
    # BER = [BER_0, BER_1, BER_2]
    BER = [BER_2]
    # # best_parameters = [best_parameters_0, best_parameters_1, best_parameters_2]
    best_parameters = [best_parameters_2]
    plot_original_vs_optimized(BER, BER_ORIGINAL, best_parameters, original_parameters, alphas)
    # plot_original_vs_optimized([], BER_ORIGINAL, [], original_parameters, alphas)
