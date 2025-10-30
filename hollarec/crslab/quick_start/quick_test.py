from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system


def test_dataset(config):
    """A quick test for dataset loading."""
    # Load dataset
    dataset = get_dataset(config, config["tokenize"], restore=False, save=False)
    side_data = dataset.side_data
    vocab = dataset.vocab

    print(f"Train samples: {len(dataset.train_data)}")
    print(f"Valid samples: {len(dataset.valid_data)}")
    print(f"Test samples: {len(dataset.test_data)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"data sample: {dataset.train_data[30]}")
    # print(f'Number of items: {side_data["n_entity"]}')

def test_dataloader(config):
    """A quick test for dataloader loading."""
    # Load dataset
    dataset = get_dataset(config, config["tokenize"], restore=False, save=False)
    vocab = dataset.vocab

    # Load dataloader
    dataloader = get_dataloader(config, dataset.train_data, vocab)
    dataloader
    for batch in dataloader.get_conv_data(batch_size=4, shuffle=False):
        print(f"Batch data sample: {batch}")
        break
    



# if __name__ == "__main__":
def run_quick_test(config):
    # Load config
    # config = Config("config/crs/hollarec/redial.yaml")

    # Test dataset
    test_dataset(config)
    # Test dataloader
    test_dataloader(config)
