from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system
from crslab.data.dataset import ReDialDataset2
from loguru import logger

def flatten(ll):
    result = []
    for l in ll:
        result.extend(l)
    return result

def test_dataset(config):
    """A quick test for dataset loading."""
    # Load dataset
    # dataset = get_dataset(config, config["tokenize"], restore=False, save=False)
    dataset = ReDialDataset2(config, config["tokenize"], restore=False, save=False)
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
    logger.info(f"ind2movie len: {len(vocab['ind2movie'])} sample:")
    print(vocab['ind2movie'][122411])
    # Load dataloader
    dataloader = get_dataloader(config, dataset.train_data, vocab)

    mm_node_coverage = {
        'txt': 0,
        'img': 0,
        'vdo': 0,
        'ado': 0
    }
    mm_node_coverage_dist = {
        'txt': [],
        'img': [],
        'vdo': [],
        'ado': []
    }
    total_convs = {'txt': 0, 'img': 0, 'vdo': 0, 'ado': 0}
    for conv_dict in dataset.train_data:
        target_node_count = len(conv_dict['context_movies'])
        for m in ['txt', 'img', 'vdo', 'ado']:
            related_nodes_count = len(flatten(conv_dict['related_movies'][m]))
            unq_related_nodes_count = len(list(set(flatten(conv_dict['related_movies'][m]))))
            if unq_related_nodes_count > 0:
                mm_node_coverage[m] += related_nodes_count - unq_related_nodes_count
                total_convs[m] += 1
                mm_node_coverage_dist[m].append(related_nodes_count - unq_related_nodes_count)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    for i, m in enumerate(['txt', 'img', 'vdo', 'ado']):
        mm_node_coverage[m] /= total_convs[m]
        logger.info(f"Average {m} related node coverage: {mm_node_coverage[m]:.4f}")
        plt.subplot(2, 2, i+1)
        plt.hist(mm_node_coverage_dist[m], bins=30, alpha=0.7, color='blue')
        plt.title(f"{m} related node coverage distribution")
        plt.xlabel("Number of covered nodes")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
def run_quick_test(config):
    # Load config
    # config = Config("config/crs/hollarec/redial.yaml")
    # Test dataset
    # test_dataset(config)
    # test_dataloader(config)
    test_dataset(config)


    logger.info("*** RUN QUICK TEST SUCCESSFULLY! ***")
