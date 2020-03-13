import numpy as np
import matplotlib.pyplot as plt


def plot_hist(train_hist, epochs, figsize=(15 ,8)):
    x_ax = list(range(epochs))
    train_loss, val_loss, lb_score, lr_hist = zip(*train_hist)
    val_loss = np.array(val_loss)
    lb_score = np.array(lb_score)

    fig, ax1 = plt.subplots(figsize=figsize)

    color = 'tab:red' # For score
    ax1.plot(x_ax, lb_score, color=color)
    ax1.scatter([lb_score.argmax()], [lb_score.max()], c='red', label='')
    ax1.text(lb_score.argmax( ) +.003, lb_score.max( ) +.0003, 'Max score: %5.3f. %s epoch' % (lb_score.max(), lb_score.argmax()), fontsize=10)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('LB Recall score', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.grid(True)

    ax2 = ax1.twinx() # For Loss
    color = 'tab:pink'
    ax2.plot(x_ax, train_loss,
             x_ax, val_loss)
    ax2.scatter([val_loss.argmin()], [val_loss.min()], c='red', label='')
    ax2.text(val_loss.argmin( ) +.03, val_loss.min( ) +.03, 'Min Loss: %5.3f. %s epoch' % (val_loss.min(), val_loss.argmin()), fontsize=10)
    ax2.legend(('Train loss', 'Val loss'), loc='center right')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Log Loss', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx() # For LR
    ax3.spines["right"].set_position(("axes", 1.05))
    color = 'tab:green'
    ax3.plot(x_ax, lr_hist, color=color)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate', color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()
