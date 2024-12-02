import logging

import matplotlib.pyplot as plt

log_level = logging.DEBUG
text_logger = logging.getLogger("magnify")
title_logger = []
plot_logger = []


def debug(text, data=None):
    if log_level > logging.DEBUG:
        return

    if data is None:
        text_logger.debug(text)
    else:
        title_logger.append(text)
        plot_logger.append(data.copy())


def warning(text, data=None):
    if log_level > logging.WARNING:
        return

    if data is None:
        text_logger.debug(text)
    else:
        title_logger.append(text)
        plot_logger.append(data.copy())


def figure():
    num_plots = len(plot_logger)
    fig, axs = plt.subplots(nrows=num_plots, squeeze=False, figsize=(10, 10 * num_plots))
    axs = axs[:, 0]

    for ax, title, data in zip(axs, title_logger, plot_logger):
        ax.set_title(title)
        ax.imshow(data)
    title_logger.clear()
    plot_logger.clear()

    return fig
