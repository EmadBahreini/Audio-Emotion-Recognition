import torch

class BestModelSaver:
    """Stop training if validation accuracy does not improve after a certain number of epochs."""
    def __init__(self, delta=0, path='best_model.pth'):
        """
        Args:
            delta (float): Minimum change to qualify as an improvement.
            path (str): Filepath to save the best model.
        """

        self.delta = delta
        self.path = path
        self.best_score = None
        self.best_acc = 0.0

    def __call__(self, val_acc, model):
        """
        Args:
            val_acc (float): Current validation accuracy.
            model (torch.nn.Module): Model to save if accuracy improves.
        """
        score = val_acc  # We maximize accuracy, so higher is better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)

        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_acc, model)



    def save_checkpoint(self, val_acc, model):
        """Saves model when validation accuracy improves."""
        torch.save(model.state_dict(), self.path)
        self.best_acc = val_acc


