import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import Accuracy, MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassSpecificity, MulticlassConfusionMatrix
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore
from tabulate import tabulate

class Trainer():
    def __init__(self, **kwargs):
        self.epochs = kwargs["epochs"]
        self.model = kwargs["model"]
        self.train_dataloader = kwargs["train_dataloader"]
        self.valid_dataloader = kwargs["valid_dataloader"]
        self.test_dataloader = kwargs["test_dataloader"]
        
        self.loss_fn_weights = torch.tensor([1/140, 1/160, 1/160]).to(self.model.device)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.15)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.999)
        self.writer = SummaryWriter()

        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        
    def train(self):
        train_accs = []
        train_losses = []

        accuracy = Accuracy(task='multiclass', num_classes=3)

        self.model.train()
        for epoch in range(self.epochs):
            print(Fore.YELLOW + f"Epoch: {(epoch+1):02}/{self.epochs}")
            for batch, (x, y) in enumerate(self.train_dataloader):
                x, y  = x.to(self.model.device), y.to(self.model.device)
                self.optimizer.zero_grad() 
                logits, _, _ = self.model(x)
                self.loss_fn.weights = self.loss_fn_weights
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()
                
                preds = logits.argmax(1)
                acc = accuracy(y.cpu(), preds.cpu())
                
                train_losses.append(loss.item())
                train_accs.append(acc.item())

                if (batch % 10 == 0) or (batch == len(self.train_dataloader)):
                    self.train_loss.append(sum(train_losses)/len(train_losses))
                    self.train_acc.append(sum(train_accs)/len(train_accs))
                    
                    self.writer.add_scalar('train_loss', self.train_loss[-1], epoch * len(self.train_dataloader) + batch)
                    self.writer.add_scalar('train_acc', self.train_acc[-1], epoch * len(self.train_dataloader) + batch)
                    
                    train_losses.clear()
                    train_accs.clear()

                    self.validate(epoch, batch)

                    self.model_report_and_save()

                    self.lr_scheduler.step()

    def validate(self, epoch, batch):
        valid_accs = []
        valid_losses = []

        accuracy = Accuracy(task='multiclass', num_classes=3)

        self.model.eval()
        with torch.no_grad():
            for x, y in self.valid_dataloader:
                x, y  = x.to(self.model.device), y.to(self.model.device)
                logits, _, _ = self.model(x)
                self.loss_fn.weight = None
                loss = self.loss_fn(logits, y)
                preds = logits.argmax(1)
                acc = accuracy(y.cpu(), preds.cpu())
                valid_losses.append(loss.item())
                valid_accs.append(acc.item())
        
        self.valid_loss.append(sum(valid_losses)/len(valid_losses))
        self.valid_acc.append(sum(valid_accs)/len(valid_accs))

        # Why train_dataloader?!!!
        self.writer.add_scalar('valid_loss', self.valid_loss[-1], epoch * len(self.train_dataloader) + batch)
        self.writer.add_scalar('valid_acc', self.valid_acc[-1], epoch * len(self.train_dataloader) + batch)
        
        valid_losses.clear()
        valid_accs.clear()

    def model_report_and_save(self):
        saved = False
        
        if self.model.best_state["loss"]["value"] > self.valid_loss[-1]:
            self.model.save_best_state("loss", self.valid_loss[-1])
            saved = True
        elif self.model.best_state["acc"]["value"] < self.valid_acc[-1]:
            self.model.save_best_state("acc", self.valid_acc[-1])
            saved = True

        if saved:
            print(Fore.GREEN + f"Training Loss(Accuracy): {self.train_loss[-1]:.2f}({self.train_acc[-1]:.2f}), Validation Loss(Accuracy): {self.valid_loss[-1]:.2f}({self.valid_acc[-1]:.2f})")
        else:
            print(Fore.RED + f"Training Loss(Accuracy): {self.train_loss[-1]:.2f}({self.train_acc[-1]:.2f}), Validation Loss(Accuracy): {self.valid_loss[-1]:.2f}({self.valid_acc[-1]:.2f})")

    def test(self, dataloader):
        metrics = {}
        labels = []
        preds = []

        accuracy = MulticlassAccuracy(num_classes=3, average=None)
        f1score = MulticlassF1Score(num_classes=3, average=None)
        sensitivity = MulticlassRecall(num_classes=3, average=None)
        precision = MulticlassPrecision(num_classes=3, average=None)
        specificity = MulticlassSpecificity(num_classes=3, average=None)
        confusion_matrix = MulticlassConfusionMatrix(num_classes=3, normalize="true")
        
        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                logits, _, _ = self.model(x.to(self.model.device))
                preds.append(logits.argmax(1).cpu())
                labels.append(y)

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        metrics["accuracy"] = torch.round(accuracy(preds, labels), decimals=2).reshape(3, 1)
        metrics["sensitivity"] = torch.round(sensitivity(preds, labels), decimals=2).reshape(3, 1)
        metrics["specificity"] = torch.round(specificity(preds, labels), decimals=2).reshape(3, 1)
        metrics["precision"] = torch.round(precision(preds, labels), decimals=2).reshape(3, 1)
        metrics["f1score"] = torch.round(f1score(preds, labels), decimals=2).reshape(3, 1)
        metrics["confusion_matrix"] = confusion_matrix(preds, labels)

        table = {
            "Class": ["CN", "MCI", "AD"],
            "Sensitivity":  metrics["sensitivity"],
            "Specificity":  metrics["specificity"],
            "Precision":  metrics["precision"],
            "F1-Score":  metrics["f1score"],
            "Accuracy": metrics["accuracy"]
        }

        print(tabulate(table, headers="keys", tablefmt="simple_outline", stralign="center", numalign="center", floatfmt=".2f"))
        print(f"F1-Score macro average: {round(metrics["f1score"].mean().item(), 2):.2f}")
        print(f"Accuracy macro average: {round(metrics["accuracy"].mean().item(), 2):.2f}")
        
        confusion_matrix.plot(labels=["CN", "MCI", "AD"])