import hydra
import logging
from omegaconf import OmegaConf

# Se up logging for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def app(cfg):

    log.info("Starting the app")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    app()
