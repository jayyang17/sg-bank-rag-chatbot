from src.utils.common import read_yaml
from src.logging.logger import logging
from src.config.config_entity import *

class ConfigurationManager:
    def __init__(self, config_filepath = Path('config.yaml')):
        self.config = read_yaml(config_filepath)
        logging.info("config read")
    
    def get_path_config(self) -> PathConfig:
        cfg = self.config["path"]
        
        path_config = PathConfig(
            raw_dir = cfg['raw_dir'],
            processed_dir=cfg['processed_dir'],
            output_dir=cfg["output_dir"]
        )
        
        return path_config

    def get_retrieval_config(self) -> RetrievalConfig:
        cfg = self.config["retrieval"]
        return RetrievalConfig(
            embedding_model=cfg["embedding_model"],
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            vector_store_path=cfg["vector_store_path"]
        )

    def get_llm_config(self) -> LLMConfig:
        cfg = self.config["llm"]
        return LLMConfig(
            provider=cfg["provider"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"]
        )

    def get_prompt_config(self) -> PromptConfig:
        return PromptConfig(prompt_template=self.config["prompt_template"])

    def get_telegram_config(self) -> TelegramConfig:
        return TelegramConfig(token=self.config["telegram"]["token"])

    def get_flask_config(self) -> FlaskConfig:
        return FlaskConfig(port=self.config["flask"]["port"])

    def get_aws_config(self) -> AWSConfig:
        return AWSConfig(s3_bucket=self.config["aws"]["s3_bucket"])

if __name__ == "__main__":
    cfg = ConfigurationManager()
    raw_dir = cfg.get_path_config().raw_dir
    print(raw_dir)
    

