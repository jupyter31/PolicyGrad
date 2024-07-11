import os
import yaml
import logging
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


yaml.add_constructor("!join", join)

def read_config(config_name: str):
    config_folder = os.path.join(os.path.dirname(__file__), '..', 'src', 'config')
    config_path = os.path.join(config_folder, config_name)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def export_plot(ys, ylabel, title, filename):
    """
    Export a plot in filename

    Args:
        ys: (list) of float / int to plot
        filename: (string) directory
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_combined(name, results):
    results = np.array(results)
    xs = np.arange(results.shape[1])
    ys = np.mean(results, axis=0)
    yerrs = stats.sem(results, axis=0)
    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.plot(xs, ys, label=name)


def get_logger(filename):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger
