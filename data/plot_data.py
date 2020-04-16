import matplotlib.pyplot as plt


def plot_results(predicted_data, true_data, path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Ground Truth')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(path)
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Ground Truth')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def plot_train_history(history, title, path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.savefig(path)
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


def plot_predictions(val_data, prediction, title):
    for x, y in val_data:
        plot = show_plot([x[0][:, 3], y[0].numpy(), prediction], 1, title)
        plot.show()
