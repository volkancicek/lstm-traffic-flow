import matplotlib.pyplot as plt


def denormalize_data(data, data_mean, data_std):
    return (data * data_std) + data_mean


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def create_time_steps(length):
    return list(range(-length + 1, 1))


def show_plot(plot_data, delta, title):
    label_names = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=label_names[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=label_names[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')

    return plt


class PlotData:

    def __init__(self, data_mean, data_std, model):
        self.mean = data_mean
        self.std = data_std
        self.model = model

    def plot_predictions(self, val_data, title):
        for x, y in val_data:
            plot = show_plot([denormalize_data(x[0][:, 3].numpy(), self.mean[3], self.std[3]),
                              denormalize_data(y[0].numpy(), self.mean[3], self.std[3]),
                              denormalize_data(self.model.predict(x)[0], self.mean[3],
                                               self.std[3])], 1, title)
            plot.show()
