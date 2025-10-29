from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system

def test_dataset(config):
    """A quick test for dataset loading."""
    # Load dataset
    dataset = get_dataset(config, config['tokenize'], restore=False, save=False)
    side_data = dataset.side_data
    vocab = dataset.vocab

    print(f'Train samples: {len(dataset.train_data)}')
    print(f'Valid samples: {len(dataset.valid_data)}')
    print(f'Test samples: {len(dataset.test_data)}')
    print(f'Vocabulary size: {len(vocab)}')
    print(f'Number of items: {side_data["n_entity"]}')

if __name__ == "__main__":
    # Load config
    config = Config('config/crs/hollarec/redial.yaml')
    
    # Test dataset
    test_dataset(config)