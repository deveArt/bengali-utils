import numpy as np
from abc import ABCMeta, abstractmethod
import torch
import os
from tqdm.notebook import trange, tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

MODEL_DIR = "./checkpoint/"
MODEL_FILE = MODEL_DIR + "model_weights{epoch}.pth"


def stratified_splits(X: np.ndarray, Y: np.ndarray, val_rate: float, shuffle=True, rand_seed=None):
    mskf = MultilabelStratifiedKFold(n_splits=int(np.ceil(1/val_rate)), shuffle=shuffle, random_state=rand_seed)

    for train_index, val_index in mskf.split(X, Y):
        yield X[train_index], X[val_index], Y[train_index], Y[val_index]


def get_avail_device():
    if torch.cuda.is_available():
        avail_device = torch.device('cuda')
    else:
        avail_device = torch.device('cpu')

    return avail_device


class EarlyStop:
    def __init__(self, patience=5, min_delta=0.001, metric_ascend=True):

        self.patience = patience
        self.bad_epochs = 0
        self.min_delta = min_delta
        self.metric_log = []

        if metric_ascend:
            self.check_fn = lambda m, metric_state: m - self.min_delta > metric_state
        else:
            self.check_fn = lambda m, metric_state: m + self.min_delta < metric_state

    def __call__(self, metric):

        break_train = False

        if self.metric_state is None or self.check_fn(metric, self.metric_state):
            self.bad_epochs = 0
        else:
            if self.bad_epochs >= self.patience:
                break_train = True

            self.bad_epochs += 1

        self.metric_log.append(metric)
        return break_train

    @property
    def best_epoch(self):
        return np.argmax(self.metric_log) if len(self.metric_log) > 0 else None

    @property
    def metric_state(self):
        return np.max(self.metric_log) if len(self.metric_log) > 0 else None

    @property
    def last_better(self) -> bool:
        last_better = False

        if len(self.metric_log) > 1:
            last_better = self.check_fn(self.metric_log[-1], self.metric_log[-2])

        return last_better


class TrainerABC(metaclass=ABCMeta):

    def __init__(self, epochs, patience=5, min_delta=0.00001, metric_ascend=True):
        self.train_hist = []
        self.es = EarlyStop(patience, min_delta, metric_ascend=metric_ascend)
        self.epochs = epochs

        os.makedirs(MODEL_DIR, exist_ok=True)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class TrainerSimple(TrainerABC):

    def __init__(self, epochs, patience=5, min_delta=0.00001, metric_ascend=True):
        super().__init__(epochs, patience, min_delta, metric_ascend)

    def __call__(self, train_ds, val_ds, model, optimizer, scheduler):
        avail_device = get_avail_device()

        for epoch in range(self.epochs):
            train_loss, val_loss, val_score = [], [], []

            model.train()
            # Training
            for batch_id, (batch, label) in enumerate(train_ds):
                batch, label = batch.to(avail_device), label.to(avail_device)
                loss, _ = model.run(batch, label)

                loss.backward()
                train_loss.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            model.eval()
            # Validate
            with torch.no_grad():
                for batch_id, (batch, label) in enumerate(val_ds):
                    batch, label = batch.to(avail_device), label.to(avail_device)
                    loss, score = model.run(batch, label)

                    val_score.append(score)
                    val_loss.append(loss.item())

            ep_train_loss = np.mean(train_loss)
            ep_val_loss = np.mean(val_loss)
            ep_val_score = np.mean(val_score)
            stop_training = self.es(ep_val_score)

            self.train_hist.append((ep_train_loss, ep_val_loss, ep_val_score, scheduler.get_last_lr()[0]))

            postfix = dict(train_loss=ep_train_loss,
                           val_loss=ep_val_loss,
                           lb_score=ep_val_score,
                           best_epoch=self.es.best_epoch)

            print(postfix)

            if stop_training: break
            torch.save(model.state_dict(), MODEL_FILE.format(epoch=epoch))


class TrainerMonitor(TrainerABC):

    def __init__(self, epochs, patience=5, min_delta=0.00001, metric_ascend=True):
        super().__init__(epochs, patience, min_delta, metric_ascend)

    def __call__(self, train_ds, val_ds, model, optimizer, scheduler):
        avail_device = get_avail_device()

        with trange(self.epochs) as monitor:

            for epoch in monitor:
                train_loss, val_loss, val_score = [], [], []
                monitor.set_description("Epoch %s" % epoch)

                model.train()
                # Training
                for batch_id, (batch, label) in enumerate(tqdm(train_ds, position=0, desc="Training", leave=False)):
                    batch, label = batch.to(avail_device), label.to(avail_device)
                    loss, _ = model.run(batch, label)

                    loss.backward()
                    train_loss.append(loss.item())

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                model.eval()
                # Validate
                with torch.no_grad():
                    for batch_id, (batch, label) in enumerate(tqdm(val_ds, position=0, desc="Validation", leave=False)):
                        batch, label = batch.to(avail_device), label.to(avail_device)
                        loss, score = model.run(batch, label)

                        val_score.append(score)
                        val_loss.append(loss.item())

                ep_train_loss = np.mean(train_loss)
                ep_val_loss = np.mean(val_loss)
                ep_val_score = np.mean(val_score)
                stop_training = self.es(ep_val_score)

                self.train_hist.append((ep_train_loss, ep_val_loss, ep_val_score, scheduler.get_last_lr()[0]))

                postfix = dict(train_loss=ep_train_loss,
                               val_loss=ep_val_loss,
                               lb_score=ep_val_score,
                               best_epoch=self.es.best_epoch)

                monitor.set_postfix(**postfix)

                if stop_training: break
                torch.save(model.state_dict(), MODEL_FILE.format(epoch=epoch))