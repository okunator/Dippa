import segmentation_models_pytorch as smp
from typing import List

from src.dl.models import (
    Decoder, MultiTaskSegModel, SegHead, TimmUniversalEncoder
)


class HoverNet(MultiTaskSegModel):
    def __init__(
            self,
            encoder_name: str="resnet50",
            encoder_in_channels: int=3,
            encoder_pretrain: bool=True,
            encoder_depth: int=5,
            encoder_freeze: bool=False,
            decoder_type_branch: bool=True,
            decoder_n_layers: int=1,
            decoder_n_blocks: int=2,
            decoder_preactivate: bool=False,
            decoder_upsampling: str="fixed_unpool",
            decoder_weight_init: str="he",
            decoder_short_skips: str=None,
            decoder_channels: List[int]=None,
            activation: str="relu",
            normalization: str="bn",
            weight_standardize: bool=False,
            long_skips: str="unet",
            long_skip_merge_policy: str="sum",
            n_classes_type: int=2,
            model_input_size: int=256
        ) -> None: 
        """
        HoverNet implementation.
        paper: https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045

        Encoders are from the segmentation_models_pytorch library and 
        timm library.
        
        Args:
        -----------
            encoder_name (str, default="resnet50"):
                Name of the encoder. Available encoders from:
                https://github.com/qubvel/segmentation_models.pytorch
            encoder_in_channels (int, default=3):
                Number of input channels in the encoder. Default set for
                RGB images
            encoder_pretrain (bool, default=True):
                imagenet pretrained encoder weights.
            encoder_depth (int, default=5):
                Number of encoder blocks. 
            encoder_freeze (bool, default=False):
                freeze the encoder for training
            decoder_type_branch (bool, default=True):
                Flag whether to include a cell type segmentation 
                branch to the network.
            decoder_n_layers (int, default=1):
                Number of multi-conv blocks inside each level of the
                decoder
            decoder_n_blocks (int, default=2):
                Number of conv blocks inside each multiconv block at 
                every level in the decoder.
            decoder_preactivate (bool, default=False):
                If True, normalization and activation are applied before
                convolution
            decoder_upsampling (str, default="fixed_unpool"):
                The upsampling method. One of: "interp", "max_unpool", 
                transconv", "fixed_unpool" TODO
            decoder_weight_init (str, default="he"):
                weight initialization method. One of: "he", "eoc", 
                "fixup" TODO
            decoder_short_skips (str, default=None):
                The short skip connection style of the decoder. One of: 
                "residual", "dense", None
            decoder_channels (List[int], default=None):
                list of integers for the number of channels in each 
                decoder block. Length of the list has to be equal to 
                `n_blocks`.
            activation (str, default="relu"):
                Activation method. One of: "mish", "swish", "relu"
            normalization (str, default="bn"):
                Normalization method. One of: "bn", "bcn" None
            weight_standardize (bool, default=False):
                Apply weight standardization in conv layers
            long_skips (str, default="unet"):
                The long skip connection style. One of: unet, unet++, 
                unet3+.
            long_skip_merge_policy (str, default="sum"):
                How to merge the features in long skips. One of: "sum", 
                "cat"
            n_classes_type (int, default=2):
                Number of cell type classes in the dataset. Type decoder
                branch is set to output this number of classes. If 
                `type_branch` is `False`, this is ignored.
            model_input_size (int, default=256):
                The input image size of the model. Assumes that input 
                images are square patches i.e. H == W.
        """
        if decoder_channels is not None:
            assert len(decoder_channels) == encoder_depth, (
                f"encoder dept: {encoder_depth} != len(decoder_channels):",
                f"{len(decoder_channels)}" 
            )
        super(HoverNet, self).__init__()

        # encoder args
        self.encoder_in_channels = encoder_in_channels
        self.encoder_name = encoder_name
        self.encoder_pretrain = encoder_pretrain
        self.encoder_depth = encoder_depth
        self.encoder_weights = "imagenet" if self.encoder_pretrain else None
        self.encoder_freeze = encoder_freeze

        # Decoder args
        self.decoder_type_branch = decoder_type_branch
        self.decoder_weight_init = decoder_weight_init
        self.decoder_n_layers = decoder_n_layers
        self.decoder_n_blocks = decoder_n_blocks
        self.decoder_preactivate = decoder_preactivate
        self.decoder_short_skips = decoder_short_skips
        self.decoder_upsampling = decoder_upsampling

        self.decoder_channels = decoder_channels
        if decoder_channels is None:
            self.decoder_channels = [256, 128, 64, 32, 16]

        self.long_skips = long_skips
        self.merge_policy = long_skip_merge_policy

        # module args
        self.activation = activation
        self.normalization = normalization
        self.weight_standardize = weight_standardize

        # model input size
        self.model_input_size = model_input_size

        # set smp or timm encoder
        if self.encoder_name.startswith("tu-"):
            name = self.encoder_name[3:]
            self.encoder = TimmUniversalEncoder(
                name,
                depth=self.encoder_depth,
                weights=self.encoder_pretrain,
                pretrained=True,
            )
        else:
            self.encoder = smp.encoders.get_encoder(
                self.encoder_name,
                in_channels=self.encoder_in_channels,
                depth=self.encoder_depth,
                weights=self.encoder_weights
            )

        # nuclei segmentation branch
        self.inst_decoder = Decoder(
            encoder_channels=list(self.encoder.out_channels),
            decoder_channels=self.decoder_channels,
            same_padding=True,
            batch_norm=self.normalization,
            activation=self.activation,
            weight_standardize=self.weight_standardize,
            n_layers=self.decoder_n_layers,
            n_blocks=self.decoder_n_blocks,
            preactivate=self.decoder_preactivate,
            up_sampling=self.decoder_upsampling,
            short_skip=self.decoder_short_skips,
            long_skip=self.long_skips,
            long_skip_merge_policy=self.merge_policy,
            model_input_size=self.model_input_size
        )

        self.inst_seg_head = SegHead(
            in_channels=self.decoder_channels[-1],
            out_channels=2,
            kernel_size=1
        )

        # nuclei gradient map regeression branch 
        aux_out_channels = 2
        self.aux_decoder = Decoder(
            encoder_channels=list(self.encoder.out_channels),
            decoder_channels=self.decoder_channels,
            same_padding=True,
            batch_norm=self.normalization,
            activation=self.activation,
            weight_standardize=self.weight_standardize,
            n_layers=self.decoder_n_layers,
            n_blocks=self.decoder_n_blocks,
            preactivate=self.decoder_preactivate,
            up_sampling=self.decoder_upsampling,
            short_skip=self.decoder_short_skips,
            long_skip=self.long_skips,
            long_skip_merge_policy=self.merge_policy,
            model_input_size=self.model_input_size
        )

        self.aux_seg_head = SegHead(
            in_channels=self.decoder_channels[-1],
            out_channels=aux_out_channels,
            kernel_size=1
        )

        # cell type classification branch
        self.type_decoder = None
        self.type_seg_head = None
        if self.decoder_type_branch:
            self.type_decoder = Decoder(
                encoder_channels=list(self.encoder.out_channels),
                decoder_channels=self.decoder_channels,
                same_padding=True,
                batch_norm=self.normalization,
                activation=self.activation,
                weight_standardize=self.weight_standardize,
                n_layers=self.decoder_n_layers,
                n_blocks=self.decoder_n_blocks,
                preactivate=self.decoder_preactivate,
                up_sampling=self.decoder_upsampling,
                short_skip=self.decoder_short_skips,
                long_skip=self.long_skips,
                long_skip_merge_policy=self.merge_policy,
                model_input_size=self.model_input_size
            )

            self.type_seg_head = SegHead(
                in_channels=self.decoder_channels[-1],
                out_channels=n_classes_type,
                kernel_size=1
            )

        self.name = f"hovernet-{self.encoder_name}"
        
        # init decoder weights
        self.initialize()
            
        # freeze encoder if specified
        if self.encoder_freeze:
            self.freeze_encoder()