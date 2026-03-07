import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import timm
from timm.layers.classifier import ClassifierHead, NormMlpClassifierHead
from typing import Any, Literal, Optional, Union

logger = getLogger(__name__)

class TimmWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_size: int,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        dropout_p: float = 0.0,
        pool_mode: Literal["gem", "gap", "gem_c", "none"] = "none",
        load_pretrained: bool = False,
        pretrained_weights_path: str = None,
        img_size: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        assert pool_mode == "none" or "vit" not in backbone_name, "pool_mode is not supported for VisionTransformer."
        model_kwargs = {"pretrained": True, "drop_rate": 0.0}
        

        # SwinV2 models have specific window sizes that require compatible image dimensions
        # Let them use their default image size instead of overriding
        if img_size is not None and "swinv2" not in backbone_name.lower():
            model_kwargs["img_size"] = img_size
        elif img_size is not None and "swinv2" in backbone_name.lower():
            logger.warning(
                "Backbone '%s' is a SwinTransformerV2 model. Ignoring img_size=%s and using model's default size to avoid window partitioning errors.",
                backbone_name,
                img_size,
            )

        try:
            self.model = timm.create_model(backbone_name, **model_kwargs)
        except TypeError as err:
            # Some backbones (e.g. EfficientNet variants) do not accept img_size.
            if "img_size" in model_kwargs and "img_size" in str(err):
                logger.warning(
                    "Backbone '%s' does not support img_size=%s; retrying with default image size.",
                    backbone_name,
                    img_size,
                )
                model_kwargs.pop("img_size", None)
                self.model = timm.create_model(backbone_name, **model_kwargs)
            else:
                raise
        
        # Load pretrained weights if specified
        if load_pretrained:
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            self.model.load_state_dict(torch.load(pretrained_weights_path,  weights_only=True), strict=True)
        
        self.num_features = self.model.num_features

        self.reset_if_necessary(pool_mode)
        
         # added proper initialization for more stable training
        self.embedding_layer = get_embedding_layer(
            id=embedding_id, feature_dim=self.num_features, embedding_dim=embedding_size, dropout_p=dropout_p
        )
        self.pool_mode = pool_mode
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        if x.dim() == 3:
            logger.info("Assuming VisionTransformer is used and taking the first token.")
            x = x[:, 0, :]

        x = self.embedding_layer(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_if_necessary(self, pool_mode: Optional[Literal["gem", "gap", "gem_c", "none"]] = None) -> None:
        if pool_mode == "none":
            pool_mode = None
        if (
            hasattr(self.model, "head")
            # NOTE(rob2u): see https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/classifier.py#L73
            and hasattr(self.model.head, "global_pool")
            and pool_mode is not None
        ):
            if isinstance(self.model.head, ClassifierHead):
                self.model.head.global_pool = get_global_pooling_layer(pool_mode, self.model.head.input_fmt)
                self.model.head.fc = nn.Identity()
                self.model.head.drop = nn.Identity()
            elif isinstance(self.model.head, NormMlpClassifierHead):
                logger.warn(
                    "Model uses NormMlpClassifierHead, for which we do not want to change the global_pooling layer."
                )
        elif pool_mode is not None and hasattr(self.model, "global_pool"):
            self.model.reset_classifier(0, "")
            self.model.global_pool = get_global_pooling_layer(pool_mode, self.num_features)
        else:
            logger.info("No pooling layer reset necessary.")

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(
        self, x: torch.Tensor, p: Union[float, torch.Tensor, nn.Parameter] = 3.0, eps: float = 1e-6
    ) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).flatten(1)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )

class GeM_adapted(nn.Module):
    def __init__(
        self, p: float = 3.0, p_shape: Union[tuple[int], int] = 1, eps: float = 1e-6
    ) -> None:  # TODO(rob2u): make p_shape variable (only 1 supported currently)
        super().__init__()
        self.p = nn.Parameter(torch.ones(p_shape) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(
        self, x: torch.Tensor, p: Union[float, torch.Tensor, nn.Parameter] = 3.0, eps: float = 1e-6
    ) -> torch.Tensor:  # TODO(rob2u): find better way instead of multiplying by sign
        batch_size = x.size(0)
        x_pow_mean = x.pow(p).mean((-2, -1))
        return x_pow_mean.abs().pow(1.0 / p).view(batch_size, -1) * (x_pow_mean.view(batch_size, -1).sign()) ** p

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )

class GAP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

class FormatWrapper(nn.Module):
    def __init__(self, pool: nn.Module, format: Literal["NCHW", "NHWC"] = "NCHW") -> None:
        super().__init__()
        assert format in ("NCHW", "NHWC")

        self.pool = pool
        self.format = format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.format == "NHWC":
            x = x.permute(0, 2, 3, 1)
        elif self.format == "NCHW":
            x = x
        else:
            raise ValueError(f"Unknown format {self.format}")
        return self.pool(x)


## Model Setup Helpers
def get_global_pooling_layer(id: str, num_features: int, format: Literal["NCHW", "NHWC"] = "NCHW") -> torch.nn.Module:
    if id == "gem":
        return FormatWrapper(GeM(), format)
    elif id == "gem_c":
        return FormatWrapper(GeM_adapted(p_shape=(num_features)), format)  # TODO(rob2u): test
    elif id == "gap":
        return FormatWrapper(GAP(), format)
    else:
        return nn.Identity()

def get_embedding_layer(id: str, feature_dim: int, embedding_dim: int, dropout_p: float = 0.0) -> torch.nn.Module:
    def init_linear_layer(layer):
        """Helper function to initialize linear layers"""
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        
    if id == "linear":
        embedding_layer = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(embedding_layer)
        return embedding_layer
    elif id == "mlp":
        layer1 = nn.Linear(feature_dim, feature_dim)
        layer2 = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(layer1)
        init_linear_layer(layer2)
        return nn.Sequential(
            layer1,
            nn.ReLU(),
            layer2
        )
    elif "linear_norm_dropout" in id:
        linear_layer = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(linear_layer)
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            linear_layer,
            nn.BatchNorm1d(embedding_dim),
        )
    elif "mlp_norm_dropout" in id:
        layer1 = nn.Linear(feature_dim, feature_dim)
        layer2 = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(layer1)
        init_linear_layer(layer2)
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            layer1,
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            layer2,
            nn.BatchNorm1d(embedding_dim),
        )
    else:
        return nn.Identity()
