import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def app(cfg):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    app()
