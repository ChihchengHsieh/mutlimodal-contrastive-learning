
from matplotlib import pyplot as plt


def plot_losses(
    train_losses,
    val_losses,
    test_losses=None,
):
    loss_keys = train_losses[0].keys()

    fig, subplots = plt.subplots(
        len(loss_keys),
        figsize=(10, 5 * len(loss_keys)),
        dpi=80,
        sharex=True,
    )

    fig.suptitle(f"Training Losses")

    for i, k in enumerate(loss_keys):
        subplots[i].set_title(k)
        subplots[i].plot(
            [loss[k] for loss in train_losses],
            marker="o",
            label="train",
            color="steelblue",
        )

        if k in val_losses[0].keys():
            subplots[i].plot(
                [data[k] for data in val_losses], marker="o", label="val", color="orange"
            )

        if test_losses and k in test_losses[0].keys():
            subplots[i].plot(
                [data[k] for data in test_losses], marker="o", label="test", color="red"
            )

        subplots[i].legend(loc="upper left")

    subplots[-1].set_xlabel("Epoch")
    plt.plot()
    plt.pause(0.01)
