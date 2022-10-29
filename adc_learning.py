import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from copy import deepcopy
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.plot.contour import plot_contour
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from scipy.io import savemat
import scipy.io as sio


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
    channel_matrix = np.zeros((num_rx_antenas * params.dense_sampling_L, data.shape[1]))
    for t in range(params.dense_sampling_L):
        signal = np.dot(channel_matrix_cos[t] * channel_matrix_exp, data)
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

    noisy_channel_matrix = np.zeros((num_rx_antenas * params.dense_sampling_L, data.shape[1]))
    for t in range(params.dense_sampling_L):
        noisy_channel_matrix[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.random.randn(
            num_rx_antenas,
            data.shape[1])

    noisy_recieved_signal = np.zeros((num_rx_antenas * params.dense_sampling_L, data.shape[1]))
    for t in range(params.dense_sampling_L):
        signal = np.zeros((num_rx_antenas, data.shape[1]))
        noisy_channel_matrix_exp = channel_matrix_exp * (
                1 + np.sqrt(params.error_variance) * np.random.randn(channel_matrix_exp.shape[0],
                                                                     channel_matrix_exp.shape[1]))
        for f in range(data.shape[1] // params.frame_size):
            signal[:, f * params.frame_size:(f + 1) * params.frame_size] = np.dot(
                channel_matrix_cos[t] * noisy_channel_matrix_exp,
                data[:, f * params.frame_size:(f + 1) * params.frame_size])
        noisy_recieved_signal[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.sqrt(ro) * signal
    return noisy_recieved_signal + noisy_channel_matrix


###############################
# Creating the Neural Network #
###############################

class AnalogNetwork(nn.Module):
    def __init__(self, num_of_adc_p):
        super(AnalogNetwork, self).__init__()
        self.analog_filter = nn.Linear(num_rx_antenas * params.dense_sampling_L, num_of_adc_p * params.dense_sampling_L,
                                       bias=False).double()  # without bias to get only the matrix

    def forward(self, x):
        x = self.analog_filter(x)
        return x


class SamplingLayer(nn.Module):
    def __init__(self, num_of_adc_p, num_samples_L_tilde):
        super(SamplingLayer, self).__init__()
        start_samples = np.linspace(0, params.dense_sampling_L, num_samples_L_tilde + 2, dtype=np.float64)
        self.weight = torch.nn.Parameter(data=torch.from_numpy(start_samples[1:-1]), requires_grad=True)
        self.num_of_adc_p = num_of_adc_p
        self.num_samples_L_tilde = num_samples_L_tilde

    def forward(self, x):
        out = torch.zeros((len(x), self.num_samples_L_tilde * self.num_of_adc_p)).double().to(device)
        t = torch.from_numpy(np.arange(1, params.dense_sampling_L + 1, dtype=np.double)).to(device)
        for v, j in enumerate(range(0, params.dense_sampling_L * self.num_of_adc_p, params.dense_sampling_L)):
            for k in range(self.num_samples_L_tilde):
                out[:, v * self.num_samples_L_tilde + k] = torch.sum(
                    x[:, j:j + params.dense_sampling_L] * torch.exp(
                        -(t - self.weight[k]) ** 2 / params.gaussian_sampling_std ** 2), dim=1)
        return out


class HardSamplingLayer(nn.Module):
    def __init__(self, weight, num_of_adc_p, num_samples_L_tilde):
        super(HardSamplingLayer, self).__init__()
        weight = weight.cpu().detach().numpy()
        weight = np.round(weight)
        rounded_weight = torch.from_numpy(weight)
        rounded_weight[rounded_weight < 1] = 1
        rounded_weight[rounded_weight > params.dense_sampling_L] = params.dense_sampling_L - 1
        self.weight = rounded_weight.long()
        self.num_of_adc_p = num_of_adc_p
        self.num_samples_L_tilde = num_samples_L_tilde

    def forward(self, x):
        out = torch.zeros((len(x), self.num_samples_L_tilde * self.num_of_adc_p)).double().to(device)
        for i in range(self.num_of_adc_p):
            for j in range(self.num_samples_L_tilde):
                out[:, i * self.num_samples_L_tilde + j] = x[:, i * params.dense_sampling_L + self.weight[j]]
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
        z = torch.zeros(x.shape[0], x.shape[1]).double().to(device)
        for i in range(self.num_code_words - 1):
            z += self.a[i] * torch.tanh(self.c[i] * (x - self.b[i]))
        return z

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
        z = torch.zeros(x.shape[0], x.shape[1]).double().to(device)
        for i in range(self.num_code_words - 1):
            z += self.a[i] * torch.sign(self.c[i] * (x - self.b[i]))
        return z


class DigitalNetwork(nn.Module):
    def __init__(self, num_of_adc_p, num_samples_L_tilde):
        super(DigitalNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(num_of_adc_p * num_samples_L_tilde, 32, bias=False).double()
        self.fc2 = torch.nn.Linear(32, 2 ** num_transmitted_symbols, bias=False).double()

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


class LoadData(Dataset):
    def __init__(self, data):
        super(LoadData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer(object):

    def __init__(self, train_loader):
        self.train_loader = train_loader

    def train(self, detector, parameters):
        validation_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, validation_size), p=[0.5, 0.5])

        labels_validation = np.sum(0.5 * (validation_data + 1).T * 2 ** np.flip(np.arange(num_transmitted_symbols)),
                                   axis=1).astype(int)

        validation_samples = detector(validation_data, ro_train).T
        validation_set = []

        for i in range(len(validation_samples)):
            validation_set.append([validation_samples[i], labels_validation[i]])
        validation_set = LoadData(validation_set)
        validation_loader = DataLoader(validation_set, batch_size=params.batch_size)
        num_of_adc_p = int(round(2.0 ** parameters['num_of_adc_p']))
        num_samples_L_tilde = int(round(2.0 ** parameters['num_samples_L_tilde']))
        num_code_words = int(round(2.0 ** round(2.0 ** (parameters['num_code_words']))))

        print('p: ', num_of_adc_p)
        print('L: ', num_samples_L_tilde)
        print('code_words: ', num_code_words)
        overall_bits = num_of_adc_p * num_samples_L_tilde * int(round(np.log2(num_code_words)))
        print('overall bits: ', overall_bits)
        net = ADCNet(num_of_adc_p, num_samples_L_tilde, num_code_words,
                     np.max(labels_train), np.max(train_samples)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=params.lr)

        valid_loss_min = np.Inf
        val_net = deepcopy(net)

        for epoch in tqdm(range(params.epochs)):
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            ###################
            # train the model #
            ###################
            net.train()
            for i, data in tqdm(enumerate(self.train_loader)):
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
                train_loss += loss.item() * params.batch_size
            ######################
            # validate the model #
            ######################
            net_val = net
            net_val.eval()
            with torch.no_grad():
                net_val.sampling_layer = HardSamplingLayer(net.sampling_layer.weight, num_of_adc_p,
                                                           num_samples_L_tilde)
                net_val.quantization_layer = HardQuantizationLayer(net.quantization_layer.a, net.quantization_layer.b,
                                                                   net.quantization_layer.c, num_code_words)
                for i, data in tqdm(enumerate(validation_loader)):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = net_val(inputs)
                    # calculate the batch loss
                    loss = criterion(outputs, labels.long())
                    # update validation loss
                    valid_loss += loss.item() * params.batch_size
                train_loss = train_loss / params.train_size
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
        return val_net

    def evaluate(self, parameters, net, alpha):
        BER = []

        num_of_adc_p = int(round(2.0 ** parameters['num_of_adc_p']))
        num_samples_L_tilde = int(round(2.0 ** parameters['num_samples_L_tilde']))
        num_code_words = int(round(2.0 ** round(2.0 ** (parameters['num_code_words']))))
        overall_bits = num_of_adc_p * num_samples_L_tilde * int(round(np.log2(num_code_words)))

        test_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, params.test_size), p=[0.5, 0.5])
        labels_test = np.sum(0.5 * (test_data + 1).T * 2 ** np.flip(np.arange(num_transmitted_symbols)), axis=1).astype(
            int)
        for r, ro in enumerate(np.flip(snr_vals)):

            test_samples = create_received_signal(test_data, ro).T
            test_set = []

            for i in range(len(test_samples)):
                test_set.append([test_samples[i], labels_test[i]])
            test_set = LoadData(test_set)
            test_loader = DataLoader(test_set, batch_size=params.batch_size, num_workers=num_workers)
            best_net = net
            net_test = best_net

            print('p: ', num_of_adc_p)
            print('L: ', num_samples_L_tilde, )
            print('code_words: ', num_code_words)
            print('overall bits: ', overall_bits)

            net_test.eval()
            net_test.sampling_layer = HardSamplingLayer(best_net.sampling_layer.weight, num_of_adc_p,
                                                        num_samples_L_tilde)
            net_test.quantization_layer = HardQuantizationLayer(best_net.quantization_layer.a,
                                                                best_net.quantization_layer.b,
                                                                best_net.quantization_layer.c, num_code_words)
            with torch.no_grad():
                cur_error = []
                for i, data in tqdm(enumerate(test_loader)):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    net_outputs = net_test(inputs)
                    outputs = net_outputs.argmax(dim=1)
                    power = 2 ** np.flip(np.arange(num_transmitted_symbols))
                    est_ber = np.floor((outputs.cpu().numpy()[:, None] % (2 * power)) / power).T
                    est_ber_scale = 2 * est_ber - np.ones((num_transmitted_symbols, params.batch_size))
                    cur_error.append(
                        np.mean(
                            np.mean(est_ber_scale != test_data[:, i * params.batch_size:(i + 1) * params.batch_size],
                                    axis=0)))
                BER.append(sum(cur_error) / len(cur_error))
        return sum(BER) / len(BER) + alpha * overall_bits

    def evaluate_plot(self, parameters, net):
        BER = []
        test_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, params.test_size), p=[0.5, 0.5])
        labels_test = np.sum(0.5 * (test_data + 1).T * 2 ** np.flip(np.arange(num_transmitted_symbols)), axis=1).astype(
            int)
        for r, ro in enumerate(np.flip(snr_vals)):
            test_samples = create_received_signal(test_data, ro).T
            test_set = []

            for i in range(len(test_samples)):
                test_set.append([test_samples[i], labels_test[i]])
            test_set = LoadData(test_set)
            test_loader = DataLoader(test_set, batch_size=params.batch_size)
            best_net = net
            net_test = deepcopy(best_net)

            num_of_adc_p = int(round(2.0 ** parameters['num_of_adc_p']))
            num_samples_L_tilde = int(round(2.0 ** parameters['num_samples_L_tilde']))
            num_code_words = int(round(2.0 ** round(2.0 ** (parameters['num_code_words']))))

            print('p: ', num_of_adc_p)
            print('L: ', num_samples_L_tilde)
            print('code_words: ', num_code_words)
            overall_bits = num_of_adc_p * num_samples_L_tilde * int(round(np.log2(num_code_words)))
            print('overall bits: ', overall_bits)

            net_test.eval()
            net_test.sampling_layer = HardSamplingLayer(best_net.sampling_layer.weight, num_of_adc_p,
                                                        num_samples_L_tilde)
            net_test.quantization_layer = HardQuantizationLayer(best_net.quantization_layer.a,
                                                                best_net.quantization_layer.b,
                                                                best_net.quantization_layer.c, num_code_words)
            with torch.no_grad():
                cur_error = []
                for i, data in tqdm(enumerate(test_loader)):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    net_outputs = net_test(inputs)
                    outputs = net_outputs.argmax(dim=1)
                    power = 2 ** np.flip(np.arange(num_transmitted_symbols))
                    est_ber = np.floor((outputs.cpu().numpy()[:, None] % (2 * power)) / power).T
                    est_ber_scale = 2 * est_ber - np.ones((num_transmitted_symbols, params.batch_size))
                    cur_error.append(
                        np.mean(
                            np.mean(est_ber_scale != test_data[:, i * params.batch_size:(i + 1) * params.batch_size],
                                    axis=0)))
                BER.append(sum(cur_error) / len(cur_error))
        print('The BER for this trial:', BER)
        return BER


def train_evaluate(trainer, parameterization, alpha):
    net = trainer.train(create_received_signal, parameterization)
    return trainer.evaluate(parameterization, net, alpha)


def custom_optimize(lower_bound, upper_bound, alpha, bits_upper_bound):
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "num_of_adc_p", "type": "range", "bounds": [lower_bound["p"], np.log2(upper_bound["p"])]},
            {"name": "num_samples_L_tilde", "type": "range",
             "bounds": [float(np.log2(lower_bound["L"])), float(np.log2(upper_bound["L"]))]},
            {"name": "num_code_words", "type": "range",
             "bounds": [float(np.log2(np.log2(lower_bound["M"]))), float(np.log2(np.log2(upper_bound["M"])))]},
        ],
        evaluation_function=lambda parameterization: train_evaluate(trainer, parameterization, alpha),
        objective_name='Bit Error Rate', minimize=True,
        parameter_constraints=['num_of_adc_p + num_samples_L_tilde + num_code_words <= ' + str(bits_upper_bound)]
        , total_trials=params.num_trials)  # 25
    return best_parameters, values, experiment, model


def hyperparameter_optimization(trainer, alpha):
    if params.settings == "large":
        best_parameters, values, experiment, model = custom_optimize({"p": 0.0, "L": 3.0, "M": 3.0},
                                                                     {"p": 16.0, "L": 16.0, "M": 16.0}, alpha,
                                                                     np.log2(max_bits))
    else:
        best_parameters, values, experiment, model = custom_optimize({"p": 0.0, "L": 3.0, "M": 3.0},
                                                                     {"p": 8.0, "L": 12.0, "M": 12.0}, alpha,
                                                                     np.log2(max_bits))
    net = trainer.train(create_received_signal, best_parameters)
    BER = trainer.evaluate_plot(best_parameters, net)

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


def plot_original_vs_optimized():
    data = sio.loadmat("BER.mat")
    small_alpha = data["alpha1"][0]
    medium_alpha = data["alpha2"][0]
    large_alpha = data["alpha3"][0]
    non_optimized = data["non-optimized"][0]
    snr = np.arange(13)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.plot(snr, small_alpha, color="b", marker="^", markersize=9, linewidth=1, markeredgewidth=0.5, fillstyle="none",
             zorder=10,
             clip_on=False,
             label=r'Meta-Learned $\displaystyle\alpha=0.0005: \{p,\tilde{L},\tilde{M}\}=\{2,4,16\};$ 32 bits'
             )
    plt.plot(snr, medium_alpha, color="r", marker="v", markersize=9, linewidth=1, markeredgewidth=0.5, fillstyle="none",
             zorder=10,
             clip_on=False,
             label=r'Meta-Learned $\displaystyle\alpha=0.005: \{p,\tilde{L},\tilde{M}\}=\{2,3,8\};$ 18 bits')
    plt.plot(snr, large_alpha, "g", marker=">", markersize=9, linewidth=1, markeredgewidth=0.5, fillstyle="none",
             zorder=10, clip_on=False,
             label=r'Meta-Learned $\displaystyle\alpha=0.05: \{p,\tilde{L},\tilde{M}\}=\{1,3,4\};$ 6 bits')
    plt.plot(snr, non_optimized, "k", linewidth=1, linestyle="dashed",
             label=r'Fixed acquisition: $\displaystyle \{p,\tilde{L},\tilde{M}\}=\{1,6,256\};$ 48 bits')
    plt.grid(which="both", linestyle='dashed', zorder=10)
    plt.xlim(0, 12)
    plt.ylim(1e-5, 1)
    plt.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    plt.xlabel("SNR [dB]")
    plt.ylabel("Error rate")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', required=False, default='large',
                        help='large settings n=16 k=8, small settings n=6 k=4')
    parser.add_argument('--num_trials', type=int, default=25, help='num of trials hyperparameter optimization')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dense_sampling_L', type=int, default=20, help='Observed discretized time frame')
    parser.add_argument('--valid_percent', type=float, default=0.3,
                        help='Percentage of training data used for validation')
    parser.add_argument('--train_size', type=int, default=int(1e4), help='train size')
    parser.add_argument('--test_size', type=int, default=int(1e5), help='test size')
    parser.add_argument('--error_variance', type=float, default=0.1, help='error variance for csi uncertainty')
    parser.add_argument('--frame_size', type=int, default=200, help='frame size')
    parser.add_argument('--f_0', type=float, default=1e3, help='f0')
    parser.add_argument('--num_of_snr', type=int, default=13, help='number of snr values to test')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--gaussian_sampling_std', type=float, default=0.4, help='gaussian sampling std')
    params = parser.parse_args()

    ###########################
    # Parmeters Intialization #
    ###########################

    if params.settings == "large":
        num_rx_antenas = 16  # Number of Rx antennas
        num_transmitted_symbols = 8  # Number of transmitted symbols
        max_bits = 150
    else:
        num_rx_antenas = 6  # Number of Rx antennas
        num_transmitted_symbols = 4  # Number of transmitted symbols
        max_bits = 24
    validation_size = int(params.train_size * params.valid_percent)
    BPSK_symbols = [-1, 1]  # BPSK symbols
    w = 2 * np.pi * params.f_0
    snr_vals = 10 ** (-0.1 * np.arange(params.num_of_snr))
    num_workers = 0
    time_vec = np.arange(1, params.dense_sampling_L + 1) / params.dense_sampling_L
    channel_matrix_cos = 1 + 0.5 * np.cos(time_vec)
    noise_vector = 1 + 0.3 * np.cos(1.5 * (np.arange(1, params.dense_sampling_L + 1)) + 0.2)

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

    #################
    # Data Creation #
    #################

    ro_train = snr_vals[0]
    train_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, params.train_size), p=[0.5, 0.5])
    labels_train = np.sum(0.5 * (train_data + 1).T * 2 ** np.flip(np.arange(num_transmitted_symbols)), axis=1).astype(
        int)
    train_samples = create_received_signal(train_data, ro_train).T
    training_set = []

    for i in range(len(train_samples)):
        training_set.append([train_samples[i], labels_train[i]])
    training_set = LoadData(training_set)
    alphas = [5e-4, 5e-3, 5e-2]

    original_parameters = {'num_of_adc_p': np.log2(4 if params.settings == "large" else 1),
                           'num_samples_L_tilde': np.log2(6),
                           'num_code_words': np.log2(np.log2(256))}

    ####################################################
    # Train the model with hyperparameter optimization #
    ####################################################
    train_loader = DataLoader(training_set, batch_size=params.batch_size, num_workers=num_workers)
    trainer = Trainer(train_loader)
    net_original = trainer.train(create_received_signal, original_parameters)
    BER_ORIGINAL = trainer.evaluate_plot(original_parameters, net_original)
    BER_0, best_parameters_0, values_0, experiment_0, model_0 = hyperparameter_optimization(trainer, alphas[0])
    plot_hyperparameters_contours(alphas[0], model_0)
    plot_BER_vs_iterations(alphas[0], experiment_0)

    BER_1, best_parameters_1, values_1, experiment_1, model_1 = hyperparameter_optimization(trainer, alphas[1])
    plot_hyperparameters_contours(alphas[1], model_1)
    plot_BER_vs_iterations(alphas[1], experiment_1)

    BER_2, best_parameters_2, values_2, experiment_2, model_2 = hyperparameter_optimization(trainer, alphas[2])
    plot_hyperparameters_contours(alphas[2], model_2)
    plot_BER_vs_iterations(alphas[2], experiment_2)

    ####################################################
    #   Train the model with original hyperparmeters   #
    ####################################################

    BER = [BER_0, BER_1, BER_2]
    ber_dict = {'alpha1': BER_0, 'alpha2': BER_1, 'alpha3': BER_2, 'non-optimized': BER_ORIGINAL}
    all_ber_dict = {'all': np.array([BER_0, BER_1, BER_2, BER_ORIGINAL])}
    savemat('BER.mat', ber_dict)
    savemat('all_BER.mat', all_ber_dict)
    best_parameters = [best_parameters_0, best_parameters_1, best_parameters_2]
    plot_original_vs_optimized()
