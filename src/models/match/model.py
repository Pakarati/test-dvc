from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel


def set_parameter_requires_grad(model, freeze_model=True):
    """
    Fixes pretrained weights if only feature extraction is wanted.
    Args:
        model: Pytorch model class. Pretrained model
        freeze_model: bool. If True, dont retran encoder layers. ->
         Grad = False. If False, retrain layers. Grad = True
    """
    for param in model.parameters():
        param.requires_grad = not freeze_model


class GlotSentenceEmbed(nn.Module):
    def __init__(
        self,
        glot_model: str = "cis-lmu/glot500-base",
        freeze_model: bool = True,
        sonar_embed_size: int = 1024,
        pooling: str = "max",
    ) -> None:
        # vocab size = 401145
        # load model
        self.max_pooling = 1 if pooling == "max" else 0
        super(GlotSentenceEmbed, self).__init__()
        self.device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")
        print(f"Running model on {self.device}")
        self.sent_embed = AutoModel.from_pretrained(glot_model).to(
            self.device
        )  # just use roberta encoder

        # update glot layers grad
        set_parameter_requires_grad(self.sent_embed, freeze_model)

        # set new head to embed to sonar space

        self.sonar_project = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dense",
                        nn.Linear(
                            self.sent_embed.config.hidden_size,
                            self.sent_embed.config.hidden_size,
                        ),
                    ),
                    ("LayerNorm", nn.LayerNorm(self.sent_embed.config.hidden_size)),
                    (
                        "ProjectSonar",
                        nn.Linear(self.sent_embed.config.hidden_size, sonar_embed_size),
                    ),
                ]
            )
        )

    def forward(self, encoded_input):
        inputs, att_mask = encoded_input.input_ids, encoded_input.attention_mask
        if inputs.size(dim=1) == 1:
            inputs = torch.squeeze(inputs, dim=1)  # remove dims 1
            att_mask = torch.squeeze(att_mask, dim=1)
        glot_outputs = self.sent_embed(
            input_ids=inputs, attention_mask=att_mask
        ).last_hidden_state  # returns (batch, # max tokens, #Roberta hidden state)

        # expand mask for broadcasting
        att_mask = att_mask.unsqueeze(dim=2)
        if self.max_pooling:
            # returns (batch, #Roberta hidden state)
            pool_outputs = (
                glot_outputs.float()
                .masked_fill_(att_mask == 0, float("-inf"))
                .max(dim=1)[0]
            )
        else:  # mean
            pool_outputs = glot_outputs.masked_fill_(att_mask == 0, 0).sum(
                dim=1
            ) / att_mask.sum(dim=1)

        outputs = self.sonar_project(
            pool_outputs
        )  # returns (batch, #sonar embeding dim)

        return outputs
