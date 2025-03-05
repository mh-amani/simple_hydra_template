from lightning import LightningModule
import torch
import hydra

class BaseLightningModule(LightningModule):
    def __init__(self, model, lr):
        super(BaseLightningModule, self).__init__()
        self.model_conf = model
        self.lr = lr

    def setup(self, stage):
        self.model = hydra.utils.instantiate(self.model_conf, _recursive_=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['answer']
        if len(y.size()) == 1:
            y = y.unsqueeze(-1)
        outputs = self(x)
        # Adjust the slicing to match the sequence length of the labels
        logits = outputs['logits'][:, -y.shape[1]:, :]
        loss_fn = torch.nn.CrossEntropyLoss()
        # Permute logits for cross entropy: (batch_size, num_classes, sequence_length)
        loss = loss_fn(logits.permute(0, 2, 1), y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer