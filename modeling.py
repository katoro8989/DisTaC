import open_clip
import torch
import os

from timm.data.transforms_factory import transforms_imagenet_train

import utils 


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            self.name, self.pretrained = args.model.split("__pretrained__")
        elif "__init__" in args.model:
            print("Using random initialization.")
            self.name, self.pretrained = args.model.split("__init__")[0], None
        else:
            self.name = args.model
            self.pretrained = "openai"

        self.cache_dir = os.environ["HF_HOME"]

        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            self.name, pretrained=self.pretrained, cache_dir=self.cache_dir
            )
        
        if args.use_timm_aug:
            first = self.train_preprocess.transforms[0]
            img_size = first.size
            self.train_preprocess = transforms_imagenet_train(
                    img_size=img_size,
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def get_intermediate_features(self, images, image_indices):
        assert self.model is not None
        return self.model.forward_intermediates(image=images, image_indices=image_indices, normalize=False)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename, pretrained="openai", aug_cfg=None):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu")
        return cls.load_from_state_dict(model_name, pretrained, state_dict, aug_cfg)
        
    @classmethod
    def load_from_state_dict(cls, model_name, pretrained, state_dict, aug_cfg=None):
        cache_dir = os.environ["HF_HOME"]
        model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir
        )
        model.load_state_dict(state_dict, strict=False)
        
        # __new__でインスタンスを生成
        instance = cls.__new__(cls)
        # 明示的にtorch.nn.Moduleの初期化処理を実行する
        torch.nn.Module.__init__(instance)
        
        # 属性の設定
        instance.model = model
        if aug_cfg is not None and aug_cfg["use_timm"]:
            first = train_preprocess.transforms[0]
            img_size = first.size
            instance.train_preprocess = transforms_imagenet_train(
                    img_size=img_size,
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
        else:
            instance.train_preprocess = train_preprocess
        instance.val_preprocess = val_preprocess
        instance.name = model_name
        instance.pretrained = pretrained
        instance.cache_dir = cache_dir
        return instance


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename: str) -> "ClassificationHead":
        """
        先に save() で保存したファイルを読み込んで、
        normalize フラグ と weights/biases を復元します。
        """
        print(f"Loading classification head from {filename}")
        payload = torch.load(filename, map_location="cpu")
        normalize = payload.get("normalize", False)

        weights = payload["weight"]
        biases  = payload.get("bias", None)
        return cls(normalize, weights, biases)




class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def get_intermediate_features(self, inputs, image_indices):
        features = self.image_encoder.get_intermediate_features(inputs, image_indices)
        features["image_features"] = self.classification_head(features["image_features"])
        return features

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
