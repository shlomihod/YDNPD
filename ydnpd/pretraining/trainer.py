from dataclasses import dataclass
from typing import Optional, Union, Tuple
import warnings

from sklearn.metrics import roc_auc_score
from ydnpd.pretraining.utils import load_data_for_classification
from ydnpd.pretraining.ft_transformer import FTTransformerModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelConfig:
    """Base configuration for the transformer model"""
    dim: int = 32
    dim_out: int = 2
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1
    batch_size: int = 128
    num_epochs: int = 20
    lr: float = 3e-4
    epsilon: Optional[float] = None  # Privacy budget for DP training


@dataclass
class PreTrainConfig:
    """Configuration for pretraining phase"""
    batch_size: int = 4
    num_epochs: int = 3
    lr: float = 3e-4


DataTuple = Tuple  # Type alias for the data tuple type


class TransformerTrainer:
    """Trainer class supporting three modes based on data availability:
    1. Public only (non-private training)
    2. Private only (DP training)
    3. Public + Private (pretrain on public, finetune on private with DP)
    """

    def __init__(
        self,
        config: ModelConfig,
        pretrain_config: Optional[PreTrainConfig] = None,
    ):
        """
        Initialize the trainer with specified configuration.

        Args:
            config: ModelConfig object containing model hyperparameters
            pretrain_config: Optional configuration for pretraining phase
        """
        self.config = config
        self.pretrain_config = pretrain_config or PreTrainConfig()
        self.model = None

    def _create_model(self, dp: bool = False, partial_dp: bool = False) -> FTTransformerModel:
        """Create a new FTTransformerModel instance with specified privacy settings"""
        model_params = {
            "dim": self.config.dim,
            "dim_out": self.config.dim_out,
            "depth": self.config.depth,
            "heads": self.config.heads,
            "attn_dropout": self.config.attn_dropout,
            "ff_dropout": self.config.ff_dropout,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "lr": self.config.lr,
            "load_best_model_when_trained": True,
            "verbose": True,
        }

        if dp:
            if self.config.epsilon is None:
                raise ValueError("epsilon must be specified in config for private training")
            model_params.update({
                "dp": True,
                "epsilon": self.config.epsilon
            })

        if partial_dp:
            model_params.update({
                "partial_dp": True
            })

        return FTTransformerModel(**model_params)

    def train(
        self,
        private_data: Optional[DataTuple] = None,
        public_data: Optional[DataTuple] = None
    ) -> None:
        """
        Train the model based on provided data.
        Mode is automatically determined by data availability:
        - public_data only -> non-private training
        - private_data only -> private training with DP
        - both -> pretrain on public_data, finetune on private_data with DP

        Args:
            private_data: Optional tuple containing private training data
            public_data: Optional tuple containing public training data
        """
        if private_data is None and public_data is None:
            raise ValueError("At least one of private_data or public_data must be provided")

        # Determine training mode based on data availability
        if private_data is None:
            # Public data only - non-private training
            self._train_public(public_data)
        elif public_data is None:
            # Private data only - DP training
            self._train_private(private_data)
        else:
            # Both - pretrain on public, finetune on private with DP
            self._train_pretrain_private(public_data, private_data)

    def _train_public(self, data: DataTuple) -> None:
        """Train model on public data without privacy"""
        X_cat_train, X_cont_train, _, _, y_train, _, cat_cardinalities, _ = data

        self.model = self._create_model(dp=False)
        self.model.fit(
            X_cat_train,
            X_cont_train,
            y_train.flatten(),
            cat_cardinalities,
            X_cont_train.shape[1],
            use_class_weights=True
        )

    def _train_private(self, data: DataTuple) -> None:
        """Train model on private data with differential privacy"""
        X_cat_train, X_cont_train, _, _, y_train, _, cat_cardinalities, _ = data

        self.model = self._create_model(dp=True)
        self.model.fit(
            X_cat_train,
            X_cont_train,
            y_train.flatten(),
            cat_cardinalities,
            X_cont_train.shape[1]
        )

    def _train_pretrain_private(self, public_data: DataTuple, private_data: DataTuple) -> None:
        """Pretrain on public data then finetune with privacy on private data"""
        X_cat_train, X_cont_train, _, _, y_train, _, cat_cardinalities, _ = private_data
        X_cat_pre, X_cont_pre, _, _, y_pre, _, _, _ = public_data

        # Create model with partial DP and pretraining configuration
        self.model = self._create_model(dp=True, partial_dp=True)

        # Set up pretraining configuration
        pretrain_config = {
            'X_cat_pre': X_cat_pre,
            'X_cont_pre': X_cont_pre,
            'y_pre': y_pre,
            'pre_epochs': self.pretrain_config.num_epochs,
            'pre_batch_size': self.pretrain_config.batch_size,
            'pre_lr': self.pretrain_config.lr,
        }
        self.model.partial_pretrain_config = pretrain_config

        # Fit model with pretraining and private finetuning
        self.model.fit(
            X_cat_train,
            X_cont_train,
            y_train.flatten(),
            cat_cardinalities,
            X_cont_train.shape[1]
        )

    def evaluate(self, data: DataTuple) -> dict[str, float]:
        """Evaluate model on validation data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        _, _, X_cat_val, X_cont_val, _, y_val, _, _ = data

        y_pred = self.model.predict_proba(X_cat_val, X_cont_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return {'auc': auc}

    @staticmethod
    def train_and_evaluate(
        config: ModelConfig,
        private_data_pointer: Optional[Union[str, tuple]] = None,
        public_data_pointer: Optional[Union[str, tuple]] = None,
        pretrain_config: Optional[PreTrainConfig] = None,
    ) -> dict[str, float]:
        """Convenience method to train and evaluate in one call"""
        trainer = TransformerTrainer(config, pretrain_config)

        private_data = None if private_data_pointer is None else load_data_for_classification(private_data_pointer)
        public_data = None if public_data_pointer is None else load_data_for_classification(public_data_pointer)

        trainer.train(private_data=private_data, public_data=public_data)

        # Evaluate on private data if available, otherwise public data
        eval_data = private_data if private_data is not None else public_data
        return trainer.evaluate(eval_data)
