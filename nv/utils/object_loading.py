from torch.utils.data import DataLoader

import nv.datasets
from nv.collate_fn.collate import Collator
from nv.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    assert "val" not in configs["data"], 'Expected to make validation set from part of train'
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        ds = params["dataset"]
        dataset = configs.init_obj(ds, nv.datasets)

        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
        else:
            raise Exception()

        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=Collator(configs.config["mel_spec"]),
            shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
        )
        dataloaders[split] = dataloader

    return dataloaders
